"""Retrieval-only adapter for SWE-bench Lite.

SWE-bench measures whether a model can *fix* an issue, which needs a model
provider and a per-instance test harness.  This adapter measures the part the
harness owns end to end and can run without a key: given the issue text, does
the retriever surface the files the gold patch touches?  For each instance it
checks out ``base_commit``, builds the index over that snapshot, ranks files
with the hybrid retriever, and reports file Recall@k / Recall@1 / MRR against
the patch's files.

    python3 -m eval.swebench_retrieval --limit 120 --out swebench-lite.json

Instances come from the HF datasets-server rows API unless ``--instances``
points at a JSON file shaped like the rows (repo, base_commit,
problem_statement, patch).  Repos are cloned once into ``--cache`` (full
clones, since base_commit need not be on a shallow tip) and reused.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.index import RepoIndex  # noqa: E402
from harness.retrieval import HybridRetriever  # noqa: E402

ROWS_API = ("https://datasets-server.huggingface.co/rows"
            "?dataset=princeton-nlp/SWE-bench_Lite&config=default&split=test")
_DIFF = re.compile(r"^diff --git a/(\S+) b/\S+", re.M)


def parse_patch_files(patch: str) -> list[str]:
    """Files a gold patch touches, in the order the diff lists them."""
    out: list[str] = []
    for path in _DIFF.findall(patch or ""):
        if path not in out:
            out.append(path)
    return out


def _fetch(limit: int, page: int = 100) -> list[dict]:
    out: list[dict] = []
    offset = 0
    while len(out) < limit:
        length = min(page, limit - len(out))
        url = f"{ROWS_API}&offset={offset}&length={length}"
        req = urllib.request.Request(url, headers={"User-Agent": "rbg-harness"})
        with urllib.request.urlopen(req, timeout=60) as fh:
            data = json.load(fh)
        rows = data.get("rows", [])
        if not rows:
            break
        out.extend(r["row"] for r in rows)
        offset += len(rows)
    return out[:limit]


def load_instances(source: str | None, limit: int) -> list[dict]:
    if source and os.path.exists(source):
        with open(source, encoding="utf-8") as fh:
            data = json.load(fh)
        instances = data["instances"] if isinstance(data, dict) else data
        return instances[:limit]
    return _fetch(limit)


def _git(args: list[str], cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def ensure_repo(repo: str, base_commit: str, cache: str) -> str:
    """Return a checkout of ``repo`` at ``base_commit``.

    A local directory is used as-is; anything else is cloned into ``cache``
    and checked out.  Errors propagate so the caller can record and skip.
    """
    if os.path.isdir(repo):
        path = os.path.abspath(repo)
    else:
        path = os.path.join(cache, repo.replace("/", "__"))
        if not os.path.isdir(path):
            os.makedirs(cache, exist_ok=True)
            clone = subprocess.run(["git", "clone", "--quiet",
                                    f"https://github.com/{repo}.git", path],
                                   capture_output=True, text=True)
            if clone.returncode != 0:
                raise RuntimeError(f"clone {repo} failed: {clone.stderr.strip()}")
    checkout = _git(["checkout", "--quiet", "--force", base_commit], path)
    if checkout.returncode != 0:
        raise RuntimeError(f"checkout {base_commit[:8]} failed: {checkout.stderr.strip()}")
    return path


def _index_cache_path(cache: str, repo: str, commit: str) -> str:
    return os.path.join(cache, "indexes", f"{repo.replace('/', '__')}@{commit}")


def load_or_build_index(path: str, repo: str, commit: str, cache: str) -> RepoIndex:
    """The parsed index for a checked-out commit, reused across runs.

    Parsing dominates a retrieval-only SWE-bench run (~50 s/instance), and the
    same instances are re-scored whenever the encoder or fusion changes.
    ``ensure_repo`` has already forced the tree to ``commit``, so the commit is
    a valid key; a pickle whose revision disagrees is rebuilt, not trusted.
    """
    fp = _index_cache_path(cache, repo, commit)
    try:
        with open(fp, "rb") as fh:
            idx = pickle.load(fh)
        if getattr(idx, "revision", None) == commit:
            return idx
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError):
        pass
    idx = RepoIndex.build(path, revision=commit)
    try:
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        tmp = fp + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(idx, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, fp)
    except OSError:
        pass
    return idx


def score_instance(instance: dict, cache: str, k: int, use_index_cache: bool = True) -> dict:
    instance_id = instance.get("instance_id", instance.get("repo", "?"))
    gold = parse_patch_files(instance.get("patch", ""))
    row = {"instance_id": instance_id, "repo": instance.get("repo"),
           "base_commit": instance.get("base_commit"), "gold_files": gold}
    if not gold:
        row.update({"error": "no files in patch"})
        return row
    try:
        path = ensure_repo(instance["repo"], instance["base_commit"], cache)
    except Exception as exc:  # noqa: BLE001 - recorded, not raised, per instance
        row.update({"error": str(exc)})
        return row
    try:
        if use_index_cache:
            idx = load_or_build_index(path, instance["repo"],
                                      instance["base_commit"], cache)
        else:
            idx = RepoIndex.build(path, revision=instance["base_commit"])
        hits = HybridRetriever(idx).retrieve(instance["problem_statement"], limit=k)
    except Exception as exc:  # noqa: BLE001
        row.update({"error": f"retrieve: {exc}"})
        return row
    seen: list[str] = []
    for h in hits:
        if h.path not in seen:
            seen.append(h.path)
    found = sum(1 for f in gold if f in seen)
    row.update({
        "retrieved": seen[:k],
        "recall_at_k": round(found / len(gold), 4),
        "recall_at_1": 1.0 if seen and seen[0] in gold else 0.0,
        "mrr": round(max((1.0 / (seen.index(f) + 1) for f in gold if f in seen), default=0.0), 4),
    })
    return row


def _report(rows: list[dict], total: int, k: int) -> dict:
    scored = [r for r in rows if "error" not in r]
    n = len(scored)

    def mean(key: str) -> float:
        return round(sum(r[key] for r in scored) / n, 4) if n else 0.0

    return {
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "split": "test",
        "k": k,
        "instances": total,
        "scored": n,
        "errors": len(rows) - n,
        "metrics": {"recall_at_k": mean("recall_at_k"),
                    "recall_at_1": mean("recall_at_1"),
                    "mrr": mean("mrr")},
        "results": sorted(rows, key=lambda r: r["instance_id"]),
    }


def run_report(instances: list[dict], cache: str, k: int = 10,
               progress=None, seed: list[dict] | None = None,
               checkpoint=None, use_index_cache: bool = True) -> dict:
    """Score every instance; ``seed`` rows are reused (resume) and ``checkpoint``
    is called with the accumulated rows after each new one, so a mid-run kill
    keeps the instances already scored."""
    rows = list(seed or [])
    done = {r["instance_id"] for r in rows}
    for i, instance in enumerate(instances, 1):
        if progress:
            progress(i, len(instances), instance.get("instance_id", "?"))
        if instance.get("instance_id") in done:
            continue
        rows.append(score_instance(instance, cache, k, use_index_cache))
        if checkpoint:
            checkpoint(_report(rows, len(instances), k))
    return _report(rows, len(instances), k)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", default="",
                    help="JSON file of instances, or empty to fetch SWE-bench Lite")
    ap.add_argument("--limit", type=int, default=120, help="instances to score (>=100)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/rbg-swebench"))
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="",
                    help="semantic encoder device (cuda/cpu); empty = auto (CUDA if present)")
    ap.add_argument("--no-index-cache", action="store_true",
                    help="re-parse every tree instead of reusing the commit-keyed cache")
    args = ap.parse_args(argv)
    if args.device:
        os.environ["RBG_EMBEDDING_DEVICE"] = args.device

    instances = load_instances(args.instances or None, args.limit)
    print(f"loaded {len(instances)} instances", file=sys.stderr)

    def progress(i, total, iid):
        print(f"[{i}/{total}] {iid}", file=sys.stderr)

    # Resume from a partial --out so a kill (this machine is memory-tight) does
    # not re-clone and re-index everything already scored.
    seed = None
    if args.out and os.path.exists(args.out):
        try:
            with open(args.out, encoding="utf-8") as fh:
                seed = json.load(fh).get("results") or []
            print(f"resuming, {len(seed)} instances already scored", file=sys.stderr)
        except (OSError, ValueError):
            seed = None

    def checkpoint(report):
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)

    report = run_report(instances, args.cache, args.k, progress=progress,
                        seed=seed, checkpoint=checkpoint,
                        use_index_cache=not args.no_index_cache)
    m = report["metrics"]
    print(f"scored {report['scored']}/{report['instances']} (errors {report['errors']})")
    print(f"recall@{args.k}={m['recall_at_k']} recall@1={m['recall_at_1']} mrr={m['mrr']}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote {args.out}")
    return 0 if report["scored"] else 2


if __name__ == "__main__":
    sys.exit(main())
