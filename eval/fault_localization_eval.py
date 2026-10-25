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


def _changed_lines(old: str, new: str) -> tuple[set[int], set[str]]:
    """Old-file line anchors and edit kinds for a reference fix.

    Insert-only fixes have no removed line; anchor them to the preceding old
    line so they can still be scored against coverage of the buggy code.
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
    if not passing:
        passing = [t for t in expected if t not in set(expected)]
    truth, derived = _ground_truth(task)
    kind = task.get("category") or derived
    if not truth or not expected:
        return {"id": task["id"], "category": kind,
                "skipped": "no reference lines or failing tests"}
    coverage = collect(root, Sandbox(root), expected + passing)
    rows = rank_lines(coverage, expected, limit=10)
    ranked = [(r["path"], r["line"]) for r in rows]
    return {
        "id": task["id"], "category": kind,
        "n_failing": len(expected), "n_passing": len(passing),
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
    by_cat: dict[str, list[dict]] = {}
    for row in usable:
        by_cat.setdefault(row["category"], []).append(row)
    return {
        "suite": suite.get("name", os.path.basename(suite_dir)),
        "n_tasks": len(results), "n_evaluated": n,
        "n_skipped": len(results) - n,
        "skipped": [r["id"] for r in results if "top_hits" not in r],
        "metrics": {
            f"hit_at_{k}": round(sum(1 for r in usable if r["top_hits"][k]) / n, 4)
            if n else 0.0 for k in (1, 5, 10)
        },
        "per_category": {
            cat: {"n": len(rows), **{
                f"hit_at_{k}": round(
                    sum(1 for r in rows if r["top_hits"][k]) / len(rows), 4)
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
