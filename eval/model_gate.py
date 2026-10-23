"""Compare a candidate retrieval config against a baseline.

    python -m eval.model_gate --baseline base.json --candidate cand.json

Each file is the output of ``eval.core_bench`` / ``eval.swe_rebench`` /
``eval.retrieval_ablation``.  A candidate is promoted only when its recall/MRR
do not regress beyond ``--tolerance`` and its latency stays under ``--p95``.
"""

from __future__ import annotations

import argparse
import json


def _metric(data: dict, *names, default=0.0) -> float:
    for name in names:
        for key, value in data.items():
            if key == name or key.startswith(name):
                if isinstance(value, (int, float)):
                    return float(value)
    return default


def gate(baseline: dict, candidate: dict, tolerance: float = 0.02,
         p95_ms: float = 2000.0) -> dict:
    base_recall = _metric(baseline, "recall_at_10", "recall_at_5", "recall_at_1")
    cand_recall = _metric(candidate, "recall_at_10", "recall_at_5", "recall_at_1")
    base_mrr = _metric(baseline, "mrr")
    cand_mrr = _metric(candidate, "mrr")
    latency = _metric(candidate, "p95_ms")
    failures = []
    if cand_recall + tolerance < base_recall:
        failures.append("recall_regression")
    if cand_mrr + tolerance < base_mrr:
        failures.append("mrr_regression")
    if latency and latency > p95_ms:
        failures.append("latency_budget")
    return {"ok": not failures, "failures": failures,
            "baseline": {"recall": base_recall, "mrr": base_mrr},
            "candidate": {"recall": cand_recall, "mrr": cand_mrr,
                          "p95_ms": latency}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="model_gate")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--p95", type=float, default=2000.0)
    args = ap.parse_args(argv)
    verdict = gate(json.load(open(args.baseline)), json.load(open(args.candidate)),
                   args.tolerance, args.p95)
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
