"""Measure fault-localization ranking against known bug locations.

For each task, run the buggy repository's tests under line coverage, compute
Ochiai over failing vs passing spectra, and check whether a line changed by the
reference fix appears in the top 1, 5, and 10 suspects.  This evaluates the
localizer, not the coding model; no LLM or training is involved.

    python -m eval.fault_localization_eval --suite eval/suites/smoke --out fl.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

from eval.benchmark import load_suite
from rbg.fault_localization import collect, rank_lines
from rbg.training import wilson_ci
from rbg.verify import Sandbox


def _changed_lines(old: str, new: str) -> tuple[set[int], set[str]]:
    """Old-file line anchors and edit kinds for a reference fix.

    Insert-only fixes have no removed line; anchor them to the following
    existing old line (or the preceding line at EOF) so they can still be
    scored against coverage of the buggy code.
    """
    lines: set[int] = set()
    kinds: set[str] = set()
    old_lines = old.splitlines()
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(None, old_lines,
                                                        new.splitlines()).get_opcodes():
        if tag == "insert":
            kinds.add("insertion")
            lines.add(max(1, min(i1 + 1, len(old_lines) or 1)))
        elif tag == "delete":
            kinds.add("deletion")
            lines.update(range(i1 + 1, i2 + 1))
        elif tag == "replace":
            kinds.add("replacement")
            lines.update(range(i1 + 1, i2 + 1))
    return lines, kinds


def _mean_ci(values: list[float]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    half = 1.96 * math.sqrt(var / len(values))
    return [round(max(0.0, mean - half), 4), round(min(1.0, mean + half), 4)]


def _regions(lines: set[tuple[str, int]]) -> list[tuple[str, int, int]]:
    by_path: dict[str, list[int]] = {}
    for path, line in lines:
        by_path.setdefault(path, []).append(line)
    out: list[tuple[str, int, int]] = []
    for path, nums in by_path.items():
        nums = sorted(set(nums))
        start = prev = nums[0]
        for n in nums[1:]:
            if n == prev + 1:
                prev = n
                continue
            out.append((path, start, prev))
            start = prev = n
        out.append((path, start, prev))
    return out


def _recall(regions, ranked, k: int, *, files: bool) -> float:
    top = ranked[:k]
    if files:
        wanted = {path for path, _a, _b in regions}
        found = {path for path, _line in top} & wanted
        return len(found) / len(wanted) if wanted else 0.0
    found = 0
    for path, start, end in regions:
        if any(rp == path and start <= line <= end for rp, line in top):
            found += 1
    return found / len(regions) if regions else 0.0


def _ground_truth(task: dict) -> tuple[set[tuple[str, int]], str]:
    lines: set[tuple[str, int]] = set()
    kinds: set[str] = set()
    changed_files = 0
    for path, old in (task.get("files") or {}).items():
        new = (task.get("fix") or {}).get(path)
        if new is None or old == new:
            continue
        file_lines, file_kinds = _changed_lines(old, new)
        if file_lines:
            changed_files += 1
        lines.update((path, line) for line in file_lines)
        kinds |= file_kinds
    if changed_files > 1:
        kind = "multi-file"
    elif kinds == {"insertion"}:
        kind = "insertion-only"
    elif kinds == {"deletion"}:
        kind = "deletion-only"
    elif len(lines) > 1:
        kind = "multi-line"
    else:
        kind = "replacement"
    return lines, kind


def evaluate_task(task: dict, root: str) -> dict:
    expected = list(task.get("expect_pass") or [])
    passing = list(task.get("must_keep_passing") or [])
    truth, derived = _ground_truth(task)
    kind = task.get("category") or derived
    if not truth or not expected:
        return {"id": task["id"], "category": kind,
                "skipped": "no reference lines or failing tests"}
    coverage = collect(root, Sandbox(root), expected + passing)
    rows = rank_lines(coverage, expected, limit=10)
    ranked = [(r["path"], r["line"]) for r in rows]
    regions = _regions(truth)
    return {
        "id": task["id"], "category": kind,
        "n_failing": len(expected), "n_passing": len(passing),
        "has_passing_spectrum": bool(passing and any(coverage.get(t) for t in passing)),
        "truth": sorted(truth), "regions": [list(r) for r in regions],
        "top_hits": {k: any(t in ranked[:k] for t in truth) for k in (1, 5, 10)},
        "file_recall": {k: round(_recall(regions, ranked, k, files=True), 4)
                        for k in (1, 5, 10)},
        "region_recall": {k: round(_recall(regions, ranked, k, files=False), 4)
                          for k in (1, 5, 10)},
        "top10": [{"path": p, "line": n, "score": rows[i]["score"]}
                  for i, (p, n) in enumerate(ranked)],
    }


def run_suite(suite_dir: str, limit: int = 0) -> dict:
    suite = load_suite(suite_dir)
    tasks = suite["tasks"][:limit] if limit else suite["tasks"]
    workdir = tempfile.mkdtemp(prefix="rbg-fl-")
    results: list[dict] = []
    try:
        for task in tasks:
            root = os.path.join(workdir, task["id"])
            Path(root).mkdir(parents=True, exist_ok=True)
            for rel, content in (task.get("files") or {}).items():
                path = Path(root, rel)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            try:
                results.append(evaluate_task(task, root))
            except Exception as exc:  # a collection failure is a result, not a crash
                results.append({"id": task["id"], "error": f"{type(exc).__name__}: {exc}"})
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    usable = [r for r in results if "top_hits" in r]
    n = len(usable)
    by_cat: dict[str, list[dict]] = {}
    for row in usable:
        by_cat.setdefault(row["category"], []).append(row)
    return {
        "suite": suite.get("name", os.path.basename(suite_dir)),
        "n_tasks": len(results), "n_evaluated": n,
        "n_skipped": len(results) - n,
        "skipped": [r["id"] for r in results if "top_hits" not in r],
        "metrics": {
            **{f"hit_at_{k}": round(
                sum(1 for r in usable if r["top_hits"][k]) / n, 4)
               if n else 0.0 for k in (1, 5, 10)},
            **{f"file_recall_at_{k}": round(
                sum(r["file_recall"][k] for r in usable) / n, 4)
               if n else 0.0 for k in (1, 5, 10)},
            **{f"region_recall_at_{k}": round(
                sum(r["region_recall"][k] for r in usable) / n, 4)
               if n else 0.0 for k in (1, 5, 10)},
            "hit_at_1_ci": wilson_ci(
                sum(1 for r in usable if r["top_hits"][1]), n),
            "hit_at_5_ci": wilson_ci(
                sum(1 for r in usable if r["top_hits"][5]), n),
            "file_recall_at_5_ci": _mean_ci(
                [r["file_recall"][5] for r in usable]),
            "region_recall_at_5_ci": _mean_ci(
                [r["region_recall"][5] for r in usable]),
        },
        "no_passing_spectrum": [r["id"] for r in usable
                                if not r["has_passing_spectrum"]],
        "per_category": {
            cat: {"n": len(rows), **{
                f"hit_at_{k}": round(
                    sum(1 for r in rows if r["top_hits"][k]) / len(rows), 4)
                for k in (1, 5, 10)}, **{
                "hit_at_5_ci": wilson_ci(
                    sum(1 for r in rows if r["top_hits"][5]), len(rows)),
                "region_recall_at_5_ci": _mean_ci(
                    [r["region_recall"][5] for r in rows])}, **{
                f"file_recall_at_{k}": round(
                    sum(r["file_recall"][k] for r in rows) / len(rows), 4)
                for k in (1, 5, 10)}, **{
                f"region_recall_at_{k}": round(
                    sum(r["region_recall"][k] for r in rows) / len(rows), 4)
                for k in (1, 5, 10)}}
            for cat, rows in sorted(by_cat.items())
        },
        "tasks": results,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rbg-fl-eval")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-hit-at-1", type=float, default=0.75,
                    help="fail when Hit@1 drops below this baseline")
    ap.add_argument("--min-evaluated", type=int, default=0,
                    help="0 requires every selected task to be evaluated")
    ap.add_argument("--min-file-recall-at-5", type=float, default=0.0)
    ap.add_argument("--min-region-recall-at-5", type=float, default=0.0)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="permit skipped tasks (diagnostic runs only)")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    result = run_suite(args.suite, args.limit)
    text = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    failures: list[str] = []
    required = args.min_evaluated or result["n_tasks"]
    if not args.allow_incomplete and result["n_evaluated"] < required:
        failures.append(f"evaluated {result['n_evaluated']}/{required} tasks")
    hit1 = result["metrics"]["hit_at_1"]
    if hit1 < args.min_hit_at_1:
        failures.append(f"Hit@1 {hit1:.3f} < baseline {args.min_hit_at_1:.3f}")
    file5 = result["metrics"]["file_recall_at_5"]
    if file5 < args.min_file_recall_at_5:
        failures.append(
            f"file recall@5 {file5:.3f} < baseline {args.min_file_recall_at_5:.3f}")
    region5 = result["metrics"]["region_recall_at_5"]
    if region5 < args.min_region_recall_at_5:
        failures.append(
            f"region recall@5 {region5:.3f} < baseline "
            f"{args.min_region_recall_at_5:.3f}")
    if failures:
        print("fault-localization gate failed: " + "; ".join(failures),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
