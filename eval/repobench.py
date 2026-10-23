"""Private merged-PR benchmark adapter.

Mine merged PRs into ``{"query": title+body, "relevant_paths": changed_files}``
rows (any exporter works), then score the retriever on them.  RepoBench's
value is that the tasks are private, so they cannot be in a model's training
data.
"""

from __future__ import annotations

import argparse
import json

from eval.core_bench import evaluate, load_cases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="repobench")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", required=True,
                    help="JSONL with query + relevant_paths")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    result = evaluate(args.repo, load_cases(args.data), args.k)
    result["benchmark"] = "repobench"
    text = json.dumps(result, indent=2)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
