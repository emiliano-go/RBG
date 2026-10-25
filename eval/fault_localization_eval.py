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
import os
import shutil
import sys
import tempfile
from pathlib import Path

from eval.benchmark import load_suite
from rbg.fault_localization import collect, rank_lines
from rbg.verify import Sandbox


def _changed_lines(old: str, new: str) -> set[int]:
    """Line numbers in the *old* file that the fix replaces."""
    removed: set[int] = set()
    old_lines = old.splitlines()
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(None, old_lines,
                                                        new.splitlines()).get_opcodes():
        if tag in ("replace", "delete"):
            removed.update(range(i1 + 1, i2 + 1))
    return removed


def _ground_truth(task: dict) -> set[tuple[str, int]]:
    lines: set[tuple[str, int]] = set()
    for path, old in (task.get("files") or {}).items():
        new = (task.get("fix") or {}).get(path)
        if new is None:
            continue
        for line in _changed_lines(old, new):
            lines.add((path, line))
    return lines


def evaluate_task(task: dict, root: str) -> dict:
    expected = list(task.get("expect_pass") or [])
    passing = list(task.get("must_keep_passing") or [])
    if not passing:
        passing = [t for t in expected if t not in set(expected)]
    truth = _ground_truth(task)
    if not truth or not expected:
        return {"id": task["id"], "skipped": "no reference lines or failing tests"}
    coverage = collect(root, Sandbox(root), expected + passing)
    rows = rank_lines(coverage, expected, limit=10)
    ranked = [(r["path"], r["line"]) for r in rows]
    return {
        "id": task["id"], "n_failing": len(expected), "n_passing": len(passing),
        "truth": sorted(truth),
        "top_hits": {k: any(t in ranked[:k] for t in truth) for k in (1, 5, 10)},
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
    return {
        "suite": suite.get("name", os.path.basename(suite_dir)),
        "n_tasks": len(results), "n_evaluated": n,
        "metrics": {
            f"hit_at_{k}": round(sum(1 for r in usable if r["top_hits"][k]) / n, 4)
            if n else 0.0 for k in (1, 5, 10)
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
    if failures:
        print("fault-localization gate failed: " + "; ".join(failures),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
