"""A live-benchmark run with a deterministic stand-in model (no network)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import live_benchmark  # noqa: E402
from rbg.orchestrator import EditOp  # noqa: E402

SUITE = {"name": "fixture", "tasks": [{
    "id": "discount",
    "task": "fix apply_discount so a percentage discount is applied correctly",
    "expected_files": ["shop.py"],
    "files": {
        "shop.py": "def apply_discount(price, percent):\n    return price - percent\n\n\ndef total(prices):\n    return sum(prices)\n",
        "test_shop.py": "from shop import apply_discount, total\n\n\ndef test_apply_discount():\n    assert apply_discount(100, 10) == 90.0\n\n\ndef test_total():\n    assert total([1, 2, 3]) == 6\n",
    },
    "expect_pass": ["test_shop.py::test_apply_discount"],
    "must_keep_passing": ["test_shop.py::test_total"],
}]}


class FixLLM:
    model = "fixture-model"
    base_url = "http://localhost/v1"
    api_key = "unused"

    def __init__(self):
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def plan(self, task, context):
        return "1. Edit shop.py.\n2. Run the tests."

    def edit(self, task, plan, context, diagnostics=""):
        self.usage["prompt_tokens"] += 100
        self.usage["completion_tokens"] += 50
        self.usage["total_tokens"] += 150
        return [EditOp("shop.py",
                       "def apply_discount(price, percent):\n"
                       "    return price * (1 - percent / 100.0)\n\n\n"
                       "def total(prices):\n    return sum(prices)\n")]


def test_live_benchmark_reports_run_level_metrics(monkeypatch):
    monkeypatch.setattr(live_benchmark, "load_suite", lambda _: SUITE)
    model = FixLLM()
    out = live_benchmark.run_live("ignored", model=model, repeats=2,
                                  price_in=1.0, price_out=2.0)
    assert out["kind"] == "live-model" and out["n"] == 2
    assert out["metrics"]["resolution_rate"] == 1.0
    assert out["metrics"]["regression_rate"] == 0.0
    assert out["metrics"]["p95_wall_s"] >= out["metrics"]["p50_wall_s"]
    assert out["metrics"]["cost_usd"] == 0.0004


def test_compare_with_adds_metric_deltas(monkeypatch, tmp_path):
    import json
    monkeypatch.setattr(live_benchmark, "load_suite", lambda _: SUITE)
    base = {"metrics": {"resolution_rate": 0.0, "cost_usd": 0.0}}
    path = tmp_path / "base.json"
    path.write_text(json.dumps(base))
    out = live_benchmark.run_live("ignored", model=FixLLM(), repeats=1)
    diff = live_benchmark.compare_metrics(base, out)
    assert diff["resolution_rate"]["delta"] == 1.0
