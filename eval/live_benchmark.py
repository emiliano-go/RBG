"""Measure a real generative model end to end.

Unlike ``eval.benchmark``, nothing here is an oracle: the model receives the
task plus retrieved context and writes the edits itself.  Results are therefore
run-level evidence (resolution, regressions, verification coverage, latency,
tokens, optional cost), not a mechanism ablation.

    RBG_LLM_API_KEY=... RBG_LLM_MODEL=gpt-4o-mini \
      python -m eval.live_benchmark --suite eval/suites/smoke --repeats 3 --out live.json

Repeat runs on held-out repositories before treating any difference as real.
The suite format is the same as ``eval.benchmark``; ``fix`` is ignored.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile

from eval.benchmark import BenchConfig, _mean, load_suite, run_task
from rbg.providers import OpenAICompatLLM
from rbg.training import wilson_ci


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ordered = sorted(xs)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return round(ordered[idx], 4)


def run_live(suite_dir: str, *, model: OpenAICompatLLM, repeats: int = 1,
             use_retrieval: bool = True, use_plan_audit: bool = True,
             max_iterations: int = 3, task_ids: list[str] | None = None,
             price_in: float = 0.0, price_out: float = 0.0,
             fault_localization: bool = False) -> dict:
    suite = load_suite(suite_dir)
    tasks = [t for t in suite["tasks"] if not task_ids or t["id"] in task_ids]
    config = BenchConfig("live", use_retrieval=use_retrieval,
                         use_plan_audit=use_plan_audit,
                         max_iterations=max_iterations)
    old_fl = os.environ.get("RBG_FAULT_LOCALIZATION")
    if fault_localization:
        os.environ["RBG_FAULT_LOCALIZATION"] = "1"
    workdir = tempfile.mkdtemp(prefix="rbg-live-")
    runs: list[dict] = []
    try:
        for repeat in range(repeats):
            for task in tasks:
                before = dict(model.usage)
                row = run_task(task, config, workdir, model=model)
                row.update({
                    "repeat": repeat,
                    "prompt_tokens": model.usage["prompt_tokens"] - before["prompt_tokens"],
                    "completion_tokens": (model.usage["completion_tokens"]
                                          - before["completion_tokens"]),
                })
                runs.append(row)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        if fault_localization:
            if old_fl is None:
                os.environ.pop("RBG_FAULT_LOCALIZATION", None)
            else:
                os.environ["RBG_FAULT_LOCALIZATION"] = old_fl

    n = len(runs)
    resolved = sum(1 for r in runs if r["resolved"])
    regressed = sum(1 for r in runs if r["regressed"])
    verified = sum(1 for r in runs if r["status"] == "verified")
    walls = [r["wall_s"] for r in runs]
    prompt = sum(r["prompt_tokens"] for r in runs)
    completion = sum(r["completion_tokens"] for r in runs)
    cost = (prompt / 1_000_000 * price_in) + (completion / 1_000_000 * price_out)
    return {
        "kind": "live-model",
        "suite": suite.get("name", os.path.basename(suite_dir)),
        "model": model.model,
        "base_url": model.base_url,
        "repeats": repeats,
        "fault_localization": fault_localization,
        "n": n,
        "metrics": {
            "resolution_rate": round(resolved / n, 4) if n else 0.0,
            "resolution_rate_ci": wilson_ci(resolved, n),
            "regression_rate": round(regressed / n, 4) if n else 0.0,
            "verification_coverage": round(verified / n, 4) if n else 0.0,
            "mean_wall_s": _mean(walls),
            "p50_wall_s": _pct(walls, 0.5),
            "p95_wall_s": _pct(walls, 0.95),
            "mean_iterations": _mean([r["iterations"] for r in runs]),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cost_usd": round(cost, 6) if (price_in or price_out) else None,
        },
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rbg-live-benchmark")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--tasks", default="", help="comma-separated task ids (holdout subset)")
    ap.add_argument("--no-retrieval", action="store_true")
    ap.add_argument("--no-plan-audit", action="store_true")
    ap.add_argument("--max-iterations", type=int, default=3)
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--price-in", type=float, default=0.0, help="USD per 1M prompt tokens")
    ap.add_argument("--price-out", type=float, default=0.0, help="USD per 1M completion tokens")
    ap.add_argument("--fault-localization", action="store_true",
                    help="enable Ochiai evidence + repair attempts (A/B this flag)")
    args = ap.parse_args(argv)

    model = OpenAICompatLLM(model=args.model or None, base_url=args.base_url or None)
    if not model.api_key and "localhost" not in model.base_url and "127.0.0.1" not in model.base_url:
        print("RBG_LLM_API_KEY is required for a hosted endpoint")
        return 2
    result = run_live(
        args.suite, model=model, repeats=max(1, args.repeats),
        use_retrieval=not args.no_retrieval,
        use_plan_audit=not args.no_plan_audit,
        max_iterations=args.max_iterations,
        task_ids=[t for t in args.tasks.split(",") if t],
        price_in=args.price_in, price_out=args.price_out,
        fault_localization=args.fault_localization)
    text = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
