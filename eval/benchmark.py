"""A small, deterministic benchmark for the control plane.

The benchmark measures the *harness*, not a model: each task ships an oracle
edit (what a perfect main LLM would write), so results are reproducible with no
network and no API key.  A config turns components on and off, and the suite is
run once per config; ``rbg compare`` diffs two result files.

    rbg eval --suite eval/suites/smoke --config baseline --out baseline.json
    rbg eval --suite eval/suites/smoke --config full     --out full.json
    rbg compare baseline.json full.json

Metric groups: task resolution, regressions in kept-passing tests, verification
coverage, and wall time, each with n; retrieval quality (Recall@k / MRR) when
retrieval is on.  Learned-model and cost metrics are out of scope here (no
model tokens are spent).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass

from harness.orchestrator import EditOp, Orchestrator
from harness.retrieval import HybridRetriever
from harness.index import RepoIndex
from harness.training import wilson_ci


@dataclass(frozen=True)
class BenchConfig:
    name: str
    use_retrieval: bool = True
    use_reranker: bool = False
    use_plan_audit: bool = True
    context_limit: int = 15
    max_iterations: int = 3


CONFIGS = {
    "baseline": BenchConfig("baseline", use_retrieval=False, use_plan_audit=False),
    "+retrieval": BenchConfig("+retrieval", use_retrieval=True, use_plan_audit=False),
    "+plan_audit": BenchConfig("+plan_audit", use_retrieval=True, use_plan_audit=True),
    "full": BenchConfig("full", use_retrieval=True, use_reranker=True, use_plan_audit=True),
}


class OracleLLM:
    """Stands in for the main model with the task's known-correct edit."""

    def __init__(self, fix: dict[str, str]) -> None:
        self.fix = fix

    def plan(self, task: str, context) -> str:
        files = ", ".join(sorted(self.fix))
        return f"1. Edit {files}.\n2. Run the tests to confirm the fix."

    def edit(self, task: str, plan: str, context, diagnostics: str = "") -> list[EditOp]:
        return [EditOp(path, content) for path, content in self.fix.items()]


def load_suite(suite_dir: str) -> dict:
    path = suite_dir if suite_dir.endswith(".json") else os.path.join(suite_dir, "suite.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _outcomes(report) -> dict[str, str]:
    status: dict[str, str] = {}
    for check in report.checks_run:
        if check.get("name") in ("tests", "full_suite"):
            for o in check.get("outcomes", []):
                status[o["test_id"]] = o["status"]
    return status


def _retrieval_quality(root: str, query: str, expected: list[str], k: int) -> tuple[float, float]:
    idx = RepoIndex.build(root)
    hits = HybridRetriever(idx).retrieve(query, limit=k)
    seen = []
    for h in hits:
        if h.path not in seen:
            seen.append(h.path)
    if not expected:
        return 1.0, 1.0
    recall = sum(1 for p in expected if p in seen) / len(expected)
    mrr = 0.0
    for p in expected:
        if p in seen:
            mrr = max(mrr, 1.0 / (seen.index(p) + 1))
    return recall, mrr


def _write_repo(root: str, files: dict[str, str]) -> None:
    for rel, content in files.items():
        full = os.path.join(root, rel)
        os.makedirs(os.path.dirname(full) or root, exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(content)


def run_task(task: dict, config: BenchConfig, workdir: str) -> dict:
    root = os.path.join(workdir, task["id"])
    os.makedirs(root, exist_ok=True)
    _write_repo(root, task["files"])

    recall, mrr = (0.0, 0.0)
    if config.use_retrieval:
        recall, mrr = _retrieval_quality(root, task["task"], task.get("expected_files", []),
                                         config.context_limit)

    orch = Orchestrator(
        root, OracleLLM(task.get("fix", {})),
        use_retrieval=config.use_retrieval, use_reranker=config.use_reranker,
        use_plan_audit=config.use_plan_audit, context_limit=config.context_limit,
        max_iterations=config.max_iterations)
    started = time.perf_counter()
    report = orch.run(task["task"], task_id=f"bench-{task['id']}")
    wall = time.perf_counter() - started

    status = _outcomes(report)
    resolved = all(status.get(t) == "passed" for t in task.get("expect_pass", []))
    regressed = any(status.get(t) != "passed" for t in task.get("must_keep_passing", []))
    return {
        "id": task["id"], "resolved": resolved, "regressed": regressed,
        "status": report.status, "iterations": report.iterations,
        "wall_s": round(wall, 4), "context_chunks": report.context_used,
        "recall_at_k": round(recall, 4), "mrr": round(mrr, 4),
        "files_changed": sorted(report.files_changed),
    }


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def run_suite(suite_dir: str, config_name: str) -> dict:
    config = CONFIGS[config_name]
    suite = load_suite(suite_dir)
    workdir = tempfile.mkdtemp(prefix="rbg-bench-")
    try:
        tasks = [run_task(t, config, workdir) for t in suite["tasks"]]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    n = len(tasks)
    resolved = sum(1 for t in tasks if t["resolved"])
    regressed = sum(1 for t in tasks if t["regressed"])
    verified = sum(1 for t in tasks if t["status"] == "verified")
    return {
        "suite": suite.get("name", os.path.basename(suite_dir)),
        "config": config_name,
        "config_values": asdict(config),
        "n": n,
        "metrics": {
            "resolution_rate": round(resolved / n, 4) if n else 0.0,
            "resolution_rate_ci": wilson_ci(resolved, n),
            "regression_rate": round(regressed / n, 4) if n else 0.0,
            "verification_coverage": round(verified / n, 4) if n else 0.0,
            "mean_wall_s": _mean([t["wall_s"] for t in tasks]),
            "mean_iterations": _mean([t["iterations"] for t in tasks]),
            "mean_context_chunks": _mean([t["context_chunks"] for t in tasks]),
            "recall_at_k": _mean([t["recall_at_k"] for t in tasks]),
            "mrr": _mean([t["mrr"] for t in tasks]),
        },
        "tasks": tasks,
    }


def compare(a: dict, b: dict) -> dict:
    """Per-metric delta ``b - a`` for two result documents."""
    out: dict[str, dict] = {}
    for key in sorted(set(a.get("metrics", {})) & set(b.get("metrics", {}))):
        av, bv = a["metrics"][key], b["metrics"][key]
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            out[key] = {"a": round(av, 4), "b": round(bv, 4), "delta": round(bv - av, 4)}
        else:
            out[key] = {"a": av, "b": bv}
    return {"a": {"config": a.get("config"), "suite": a.get("suite")},
            "b": {"config": b.get("config"), "suite": b.get("suite")},
            "metrics": out}


def _print_compare(diff: dict) -> None:
    print(f"a={diff['a']['config']}  b={diff['b']['config']}")
    for key, v in diff["metrics"].items():
        if "delta" in v:
            print(f"  {key:24s} {v['a']:>10} -> {v['b']:>10}   ({v['delta']:+.4f})")
        else:
            print(f"  {key:24s} {v['a']} -> {v['b']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rbg-benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("eval", help="run a suite under one config")
    ev.add_argument("--suite", required=True)
    ev.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ev.add_argument("--out", default="")
    cp = sub.add_parser("compare", help="diff two result files")
    cp.add_argument("a")
    cp.add_argument("b")
    args = ap.parse_args(argv)
    if args.cmd == "eval":
        result = run_suite(args.suite, args.config)
        text = json.dumps(result, indent=2)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
        print(text)
        return 0
    with open(args.a, encoding="utf-8") as fh:
        a = json.load(fh)
    with open(args.b, encoding="utf-8") as fh:
        b = json.load(fh)
    _print_compare(compare(a, b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
