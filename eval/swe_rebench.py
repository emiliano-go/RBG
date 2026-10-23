"""SWE-rebench / issue-to-file retrieval adapter.

Expects exported rows ``{"query": issue_text, "relevant_paths": [...]}``.  The
continuous SWE-rebench release is preferred over SWE-bench Verified because it
is refreshed and less contaminated.
"""

from __future__ import annotations

import argparse
import json

from eval.core_bench import evaluate, load_cases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="swe_rebench")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    result = evaluate(args.repo, load_cases(args.data), args.k)
    result["benchmark"] = "swe-rebench"
    text = json.dumps(result, indent=2)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
