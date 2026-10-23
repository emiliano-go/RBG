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


def cmd_install(args) -> int:
    from .install import TARGETS, InstallError, broad_roots, install

    try:
        records = install(
            args.target,
            scope="project" if args.project else "user",
            directory=args.dir or os.getcwd(),
            roots=args.root,
            dry_run=args.dry_run,
            uninstall=args.uninstall,
        )
    except InstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not records:
        print("error: no target selected", file=sys.stderr)
        return 1
    for rec in records:
        if rec.get("path"):
            print(f"{rec['target']:<9} {rec['status']:<11} {rec['path']}")
        else:
            print(f"{rec['target']:<9} {rec['status']:<11} ({rec.get('reason')})")
    if args.dry_run:
        print("dry run: no files written")
    if not args.uninstall:
        roots = records[0].get("roots") or []
        print("allowed roots: " + (", ".join(roots) if roots else "(none)"))
        broad = broad_roots(roots)
        if broad:
            print("warning: these roots include HOME or a parent: " + ", ".join(broad))
        print("restart your agent client to load rbg-mcp")
    if not args.target and not args.dry_run:
        print(f"pass --target {'|'.join(TARGETS)} to choose explicitly")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("eval", "compare"):
        from eval.benchmark import main as bench_main
        return bench_main(argv)
    ap = argparse.ArgumentParser(prog="harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("index", "retrieve"):
        p = sub.add_parser(name)
        p.add_argument("root")
        if name == "retrieve":
            p.add_argument("query")
    sub.add_parser("models")
    p = sub.add_parser("validate-models")
    p.add_argument("root")
    p.add_argument("--min-test", type=int, default=20)
    sub.add_parser("demo")
    p = sub.add_parser("replay")
    p.add_argument("root")
    p.add_argument("transcript")
    p.add_argument("task")
    p.add_argument("--in-place", action="store_true")
    p = sub.add_parser("install", help="register the MCP server with agent harnesses")
    p.add_argument("--target", action="append", metavar="NAME",
                   help="harness to configure (repeatable); default auto-detect")
    p.add_argument("--project", action="store_true",
                   help="write a project-local config instead of the user config")
    p.add_argument("--dir", help="project directory (default: cwd)")
    p.add_argument("--root", action="append",
                   help="repository root the server may access (repeatable); "
                        "required for user scope, defaults to the project dir")
    p.add_argument("--dry-run", action="store_true", help="print what would be written")
    p.add_argument("--uninstall", action="store_true", help="remove the rbg entry")
    args = ap.parse_args(argv)
    if args.cmd == "index":
        cmd_index(args.root)
    elif args.cmd == "retrieve":
        cmd_retrieve(args.root, args.query)
    elif args.cmd == "models":
        cmd_models()
    elif args.cmd == "validate-models":
        from .training import validate_and_promote
        verdict = validate_and_promote(args.root, min_test=args.min_test)
        print(json.dumps(verdict, indent=2))
        return 0 if verdict["promoted"] else 1
    elif args.cmd == "demo":
        return cmd_demo()
    elif args.cmd == "replay":
        return cmd_replay(args.root, args.transcript, args.task, args.in_place)
    elif args.cmd == "install":
        return cmd_install(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
