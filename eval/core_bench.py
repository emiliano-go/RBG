"""CORE-Bench-style issue-to-edit retrieval evaluation.

CORE-Bench (EMNLP 2026) found classic snippet benchmarks overstate agentic
retrieval.  This adapter consumes a small exported JSONL where each row is
``{"query": "...", "relevant_paths": ["src/a.py", ...]}`` and scores whether the
retriever surfaces a file that must change.

    python -m eval.core_bench --repo /path/to/repo --data cases.jsonl --k 10
"""

from __future__ import annotations

import argparse
import json

from harness.index import RepoIndex
from harness.retrieval import HybridRetriever


def load_cases(path: str) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("query") and row.get("relevant_paths"):
                    cases.append(row)
    return cases


def evaluate(repo: str, cases: list[dict], k: int = 10) -> dict:
    idx = RepoIndex.build(repo)
    retr = HybridRetriever(idx)
    recall = 0
    rr = 0.0
    per_case = []
    for case in cases:
        expected = set(case["relevant_paths"])
        hits = retr.retrieve(case["query"], limit=k)
        paths = [h.path for h in hits]
        hit = next((i for i, p in enumerate(paths, 1) if p in expected), None)
        recall += 1 if hit else 0
        rr += (1.0 / hit) if hit else 0.0
        per_case.append({"query": case["query"], "hit_rank": hit,
                         "paths": paths[:k]})
    n = len(cases) or 1
    return {"repo": repo, "cases": len(cases), f"recall_at_{k}": round(recall / n, 4),
            "mrr": round(rr / n, 4), "per_case": per_case}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="core_bench")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    result = evaluate(args.repo, load_cases(args.data), args.k)
    text = json.dumps(result, indent=2)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
