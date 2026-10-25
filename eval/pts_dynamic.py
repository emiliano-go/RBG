"""Dynamic ground truth for predictive test selection (PTS).

The static benchmark (`eval/effectiveness.py`) labels a test as "revealing" a
symbol when the test's text mentions that symbol, then scores the selector on
whether it finds tests that mention the symbol; the label and the feature are
the same signal, so it reads a near-perfect MRR that means nothing.

This module instead asks the question the tool actually answers: *which tests
fail when this symbol changes?*  For each sampled top-level function it injects
a ``raise`` as the first statement, runs the suite, and takes as ground truth
the tests that fail on the mutated tree but passed on the clean one.  The label
is behavioural, so a test that genuinely exercises the function counts whether
or not its name or body repeats the symbol.

    python3 -m eval.pts_dynamic <repo> [--limit 20] [--timeout 120] [--out j.json]

Bounded and best-effort: it samples a few functions and skips any whose file
cannot be mutated or whose suite cannot be collected; a repo without a runnable
suite simply yields an empty report rather than an error.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbg.index import RepoIndex, is_test_chunk  # noqa: E402
from rbg.models import TestSelector  # noqa: E402
from rbg.retrieval import tokenize  # noqa: E402

PROBE = "rbg-probe"


def mutate_source(src: str, func_name: str, lineno: int) -> str | None:
    """Return ``src`` with ``raise RuntimeError("rbg-probe")`` as the first
    statement of ``func_name`` (after any docstring), or ``None`` if the
    function cannot be located or is already a stub/empty body.

    Textual insertion, not ``ast.unparse``, so comments and formatting survive
    and line numbers below the edit stay close enough for readable diffs.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            if getattr(node, "lineno", None) == lineno or target is None:
                target = node
                if getattr(node, "lineno", None) == lineno:
                    break
    if target is None or not target.body:
        return None
    first = target.body[0]
    # Insert after a leading docstring so the function still has one.
    if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str) and len(target.body) > 1):
        first = target.body[1]
    indent = " " * first.col_offset
    lines = src.splitlines(keepends=True)
    at = first.lineno - 1
    if at < 0 or at > len(lines):
        return None
    lines.insert(at, f"{indent}raise RuntimeError({PROBE!r})\n")
    return "".join(lines)


def _parse_junit(path: str) -> dict[str, bool]:
    """{test node id -> passed}. node id mirrors pytest: path::name."""
    out: dict[str, bool] = {}
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return out
    for case in root.iter("testcase"):
        cls = case.get("classname", "")
        name = case.get("name", "")
        path = cls.replace(".", "/") + ".py" if cls else ""
        tid = f"{path}::{name}" if path else name
        failed = any(c.tag in ("failure", "error") for c in case)
        out[tid] = not failed
    return out


def run_suite(cwd: str, timeout_s: int) -> dict[str, bool]:
    """Run pytest once in ``cwd``; return {test id -> passed}. Empty on failure."""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as fh:
        xml = fh.name
    try:
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
             f"--junitxml={xml}"],
            cwd=cwd, timeout=timeout_s,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return _parse_junit(xml)
    except (subprocess.TimeoutExpired, OSError):
        return {}
    finally:
        try:
            os.unlink(xml)
        except OSError:
            pass


def select_probe_symbols(index: RepoIndex, limit: int, seed: int = 0):
    """Deterministic sample of top-level functions to mutate."""
    seen: set[tuple[str, str]] = set()
    out = []
    for s in index.symbols:
        if s.kind != "function":
            continue
        key = (s.path, s.name)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    out.sort(key=lambda s: (s.path, s.start_line))
    random.Random(seed).shuffle(out)
    return out[:limit]


def dynamic_ground_truth(root: str, index: RepoIndex, limit: int = 20,
                         timeout_s: int = 120, seed: int = 0) -> list[dict]:
    """For each sampled symbol, the tests that break when it is mutated."""
    work = tempfile.mkdtemp(prefix="rbg-pts-")
    try:
        copy = os.path.join(work, "repo")
        shutil.copytree(root, copy, ignore=shutil.ignore_patterns(".git", ".venv",
                                                                  "__pycache__", "*.pyc"))
        clean = run_suite(copy, timeout_s)
        passing = {t for t, ok in clean.items() if ok}
        if not passing:
            return []
        probes = []
        for s in select_probe_symbols(index, limit, seed):
            src_path = os.path.join(copy, s.path)
            try:
                with open(src_path, encoding="utf-8") as fh:
                    src = fh.read()
            except OSError:
                continue
            mutated = mutate_source(src, s.name, s.start_line)
            if mutated is None:
                continue
            with open(src_path, "w", encoding="utf-8") as fh:
                fh.write(mutated)
            after = run_suite(copy, timeout_s)
            with open(src_path, "w", encoding="utf-8") as fh:
                fh.write(src)
            failing = sorted(t for t in passing if after.get(t) is False)
            probes.append({"symbol": s.name, "path": s.path,
                           "ground_truth_tests": failing})
        return probes
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _tests_for(index: RepoIndex) -> list[dict]:
    defined = {s.name.lower() for s in index.symbols}
    tests = []
    for c in index.chunks:
        if is_test_chunk(c):
            tests.append({"test_id": f"{c.path}::{c.symbol or c.path}", "duration": 1.0,
                          "history": [], "symbols": sorted(set(tokenize(c.content)) & defined)})
    return tests


def score_selection(probes: list[dict], tests: list[dict], seed: int = 0,
                    k: int = 5) -> dict:
    """Recall@k / recall_at_third / MRR of the selector vs alphabetical vs random
    on the dynamic ground truth.  Symbols whose mutation broke no *passing*
    test carry no signal and are dropped."""
    selector = TestSelector()
    ids = sorted(t["test_id"] for t in tests)
    budget = max(1, len(ids) // 3)
    rng = random.Random(seed)
    agg = {n: {"r5": 0.0, "third": 0.0, "mrr": 0.0, "n": 0} for n in ("selector", "alpha", "random")}
    for p in probes:
        gt = set(p["ground_truth_tests"])
        if not gt:
            continue
        orders = {
            "selector": [r.test_id for r in selector.rank(tests, [p["symbol"]])],
            "alpha": ids,
            "random": rng.sample(ids, len(ids)),
        }
        for name, order in orders.items():
            a = agg[name]
            a["n"] += 1
            a["r5"] += sum(1 for t in order[:k] if t in gt) / len(gt)
            a["third"] += sum(1 for t in order[:budget] if t in gt) / len(gt)
            first = next((i for i, t in enumerate(order) if t in gt), None)
            a["mrr"] += 1.0 / (first + 1) if first is not None else 0.0
    out = {}
    for name, a in agg.items():
        n = a["n"] or 1
        out[name] = {"recall@k": round(min(a["r5"] / n, 1.0), 4),
                     "recall_at_third": round(min(a["third"] / n, 1.0), 4),
                     "mrr": round(a["mrr"] / n, 4), "n": a["n"]}
    return out


def run(repo: str, limit: int = 20, timeout_s: int = 120, seed: int = 0) -> dict:
    index = RepoIndex.build(repo)
    probes = dynamic_ground_truth(repo, index, limit, timeout_s, seed)
    scored = [p for p in probes if p["ground_truth_tests"]]
    return {
        "repo": os.path.abspath(repo), "revision": index.revision,
        "probed": len(probes), "with_signal": len(scored),
        "budget_tests": max(1, len(_tests_for(index)) // 3),
        "scores": score_selection(scored, _tests_for(index), seed),
        "probes": scored,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="dynamic (mutate-and-run) PTS ground truth")
    ap.add_argument("repo")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    report = run(args.repo, args.limit, args.timeout, args.seed)
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
