"""Command-line entry point and an end-to-end demo.

``python -m harness retrieve <root> <query>``  - hybrid retrieval
``python -m harness index <root>``            - repository index summary
``python -m harness models``                  - train the GBDT on a toy task
``python -m harness demo``                    - full state-machine run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

from .index import RepoIndex
from .models import GBDT, PlattScaler, Policy, TestSelector
from .orchestrator import EditOp, Orchestrator, Report
from .retrieval import HybridRetriever

# --------------------------------------------------------------------------- #
# The main coding LLM, stood in by a scripted edit for the demo.  In real use
# this is the interactive model; it receives only the retrieved context.
# --------------------------------------------------------------------------- #
FIXED_SHOP = '''\
def apply_discount(price, percent):
    return price * (1 - percent / 100.0)


def total(prices):
    return sum(prices)
'''

BUGGY_SHOP = '''\
def apply_discount(price, percent):
    return price - percent


def total(prices):
    return sum(prices)
'''

TEST_SHOP = '''\
from shop import apply_discount, total


def test_apply_discount():
    assert apply_discount(100, 10) == 90.0


def test_total():
    assert total([1, 2, 3]) == 6
'''


class ScriptedLLM:
    """Represents the single generative model.  ``plan`` and ``edit`` only see
    the bounded context the harness assembled."""

    def plan(self, task, context):
        return (
            "1. Edit shop.py: fix apply_discount to multiply by (1 - percent/100).\n"
            "2. Run test_shop.py to confirm the discount and total behave.\n"
            f"Goal: {task}"
        )

    def edit(self, task, plan, context, diagnostics=""):
        return [EditOp("shop.py", FIXED_SHOP)]


def _make_demo_repo(root: str) -> None:
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "shop.py"), "w") as fh:
        fh.write(BUGGY_SHOP)
    with open(os.path.join(root, "test_shop.py"), "w") as fh:
        fh.write(TEST_SHOP)


def cmd_index(root: str) -> None:
    idx = RepoIndex.build(root)
    print(json.dumps({
        "root": idx.root, "revision": idx.revision, "files": idx.files,
        "chunks": len(idx.chunks), "symbols": [s.name for s in idx.symbols],
    }, indent=2))


def cmd_retrieve(root: str, query: str) -> None:
    idx = RepoIndex.build(root)
    hits = HybridRetriever(idx).retrieve(query, limit=8)
    print(json.dumps([h.to_dict() for h in hits], indent=2))


def cmd_models() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = ((X[:, 0] + X[:, 1] ** 2 - 0.5) > 0).astype(float)
    split = 320
    model = GBDT(n_estimators=60, learning_rate=0.3, max_depth=3, loss="logloss").fit(X[:split], y[:split])
    pred = (model.predict(X[split:]) > 0.5).astype(float)
    acc = float((pred == y[split:]).mean())
    cal = PlattScaler().fit(model.predict(X[:split]), y[:split])
    ece = PlattScaler.ece([cal(float(s)) for s in model.predict(X[split:])], y[split:])
    print(json.dumps({"gbdt_test_accuracy": round(acc, 4), "calibrated_ece": round(ece, 4),
                      "feature_importance": model.feature_importance().tolist()}, indent=2))


def cmd_demo() -> int:
    root = tempfile.mkdtemp(prefix="harness-demo-")
    _make_demo_repo(root)
    print(f"demo repo: {root}")

    policy = Policy()
    selector = TestSelector()
    orch = Orchestrator(root, ScriptedLLM(), policy=policy, selector=selector)
    task = "fix the discount calculation so a 10 percent discount applies correctly"
    report: Report = orch.run(task, task_id="demo-1")
    print(json.dumps(report.to_dict(), indent=2))
    print("ledger entries:", len(orch.ledger.entries))
    print("commands executed:", [(c["name"], c["exit_code"]) for c in report.checks_run])
    return 0 if report.success else 1


def cmd_replay(root: str, transcript: str, task: str, in_place: bool = False) -> int:
    from .providers import ReplayLLM
    work = root
    if not in_place:
        work = os.path.join(tempfile.mkdtemp(prefix="rbg-replay-"), "repo")
        shutil.copytree(root, work)
    orch = Orchestrator(work, ReplayLLM.from_file(transcript))
    report: Report = orch.run(task, task_id="replay")
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.success else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("index", "retrieve"):
        p = sub.add_parser(name)
        p.add_argument("root")
        if name == "retrieve":
            p.add_argument("query")
    sub.add_parser("models")
    sub.add_parser("demo")
    p = sub.add_parser("replay")
    p.add_argument("root")
    p.add_argument("transcript")
    p.add_argument("task")
    p.add_argument("--in-place", action="store_true")
    for name in ("eval", "compare"):
        p = sub.add_parser(name)
        p.add_argument("args", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    if args.cmd in ("eval", "compare"):
        from eval.benchmark import main as bench_main
        return bench_main([args.cmd, *args.args])
    if args.cmd == "index":
        cmd_index(args.root)
    elif args.cmd == "retrieve":
        cmd_retrieve(args.root, args.query)
    elif args.cmd == "models":
        cmd_models()
    elif args.cmd == "demo":
        return cmd_demo()
    elif args.cmd == "replay":
        return cmd_replay(args.root, args.transcript, args.task, args.in_place)
    return 0


if __name__ == "__main__":
    sys.exit(main())
