"""Farm a real execution ledger, then gate the learned risk head on it.

The benchmark runs each task in its own throwaway root, so its ledger entries
never accumulate.  This script writes every suite task into *one* root and runs
each task under two configs, interleaved, so the ledger gets both outcomes:

* ``blind-baseline`` - no retrieval, so the model sees nothing and fails
* ``full``          - retrieval on and the oracle fix, so it resolves

The resulting temporal test slice has both classes, which is what
:func:`harness.training.validate_and_promote` needs.

    python3 -m eval.ledger_farm --suite eval/suites/curated --tasks 50
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.benchmark import CONFIGS, BlindLLM, OracleLLM, _write_repo, load_suite  # noqa: E402
from harness.orchestrator import Orchestrator  # noqa: E402
from harness.training import train_from_ledger, validate_and_promote  # noqa: E402


def _reset(root: str) -> None:
    """Clear the working tree but keep the root, so the per-root ledger (which
    lives outside the tree) survives while each run sees only its own task."""
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def farm(suite_dir: str, root: str, configs: tuple[str, ...],
         limit: int | None, quiet: bool = False) -> dict:
    suite = load_suite(suite_dir)
    tasks = suite["tasks"][:limit] if limit else suite["tasks"]
    os.makedirs(root, exist_ok=True)

    for task in tasks:
        for name in configs:
            _reset(root)  # another task's broken modules would fail this run
            _write_repo(root, task["files"])
            cfg = CONFIGS[name]
            model = (BlindLLM(task.get("fix", {})) if cfg.blind
                     else OracleLLM(task.get("fix", {})))
            orch = Orchestrator(
                root, model,
                use_retrieval=cfg.use_retrieval, use_reranker=cfg.use_reranker,
                use_plan_audit=cfg.use_plan_audit, context_limit=cfg.context_limit,
                max_iterations=cfg.max_iterations)
            report = orch.run(task["task"], task_id=f"farm-{task['id']}-{name}")
            if not quiet:
                print(f"{task['id']:40s} {name:14s} "
                      f"{report.status:10s} success={report.success}")

    models = train_from_ledger(root)
    return {
        "root": root,
        "tasks": len(tasks),
        "configs": list(configs),
        "metrics": models.metrics,
        "validated": models.validated,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ledger-farm")
    ap.add_argument("--suite", default="eval/suites/curated")
    ap.add_argument("--root", default="")
    ap.add_argument("--configs", nargs="+", default=["blind-baseline", "full"])
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--min-test", type=int, default=20)
    ap.add_argument("--keep", action="store_true",
                    help="keep a temporary root instead of deleting it")
    ap.add_argument("--force", action="store_true",
                    help="allow farming into a non-empty --root (it is wiped per run)")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root) if args.root else tempfile.mkdtemp(prefix="rbg-farm-")
    if args.root and os.path.isdir(root) and os.listdir(root) and not args.force:
        ap.error(f"refusing to wipe non-empty root {root!r}; pass --force if intended")
    try:
        out = farm(args.suite, root, tuple(args.configs), args.tasks or None)
        verdict = validate_and_promote(root, min_test=args.min_test)
        out["validation"] = verdict
        print(json.dumps(out, indent=2))
        return 0 if verdict["promoted"] else 1
    finally:
        if not args.root and not args.keep:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
