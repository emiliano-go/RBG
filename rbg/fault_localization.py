"""Spectrum-based fault localization (Ochiai).

Execution coverage turns observed failures into an interpretable location
signal: lines executed by failing tests but rarely by passing ones are
suspicious.  This is deterministic; no training set is required.  It feeds
retrieval and test selection, and never changes a verification verdict.

Coverage is collected by running Python tests with a small in-tree pytest
plugin (stdlib ``sys.settrace``) when ``RBG_FAULT_LOCALIZATION=1``; other
languages can supply the same ``{test_id: {(path, line), ...}}`` map.
"""

from __future__ import annotations

import json
import math
import os
import tempfile

PLUGIN = "rbg_coverage_plugin.py"


class CoverageUnavailable(RuntimeError):
    """The coverage run produced no artifact; do not treat it as empty evidence."""

_PLUGIN_SOURCE = r'''
import json
import os
import sys

_DATA = {}
_CUR = "<import>"
_ROOT = os.path.realpath(os.environ.get("RBG_COVERAGE_ROOT", os.getcwd()))


def _trace(frame, event, arg):
    if event != "line" or _CUR is None:
        return _trace
    name = frame.f_code.co_filename
    if name.startswith("<"):
        return _trace
    if not os.path.isabs(name):
        name = os.path.abspath(name)
    name = os.path.realpath(name)
    if not name.startswith(_ROOT + os.sep):
        return _trace
    rel = os.path.relpath(name, _ROOT).replace(os.sep, "/")
    if rel.startswith(".rbg-cov-") or rel == "rbg_coverage_plugin.py":
        return _trace
    if rel.startswith(("tests/", "test_")) or "/test_" in rel:
        return _trace
    _DATA.setdefault(_CUR, set()).add((rel, frame.f_lineno))
    return _trace


def pytest_configure(config):
    # Collection imports modules before pytest_runtest_setup runs.  Start the
    # tracer here and attribute import-time lines to a synthetic spectrum that
    # the collector merges into every test.
    sys.settrace(_trace)


def pytest_runtest_setup(item):
    global _CUR
    _CUR = item.nodeid
    sys.settrace(_trace)


def pytest_runtest_teardown(item, nextitem):
    global _CUR
    sys.settrace(None)
    _CUR = None


def pytest_sessionfinish(session, exitstatus):
    sys.settrace(None)
    path = os.environ.get("RBG_COVERAGE_FILE")
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({k: sorted(v) for k, v in _DATA.items()}, fh)
'''


def ochiai(failing: list[set[tuple[str, int]]], passing: list[set[tuple[str, int]]],
           ) -> dict[tuple[str, int], float]:
    """Ochiai suspiciousness for every line seen in either spectrum."""
    ef: dict[tuple[str, int], int] = {}
    ep: dict[tuple[str, int], int] = {}
    for cov in failing:
        for line in cov:
            ef[line] = ef.get(line, 0) + 1
    for cov in passing:
        for line in cov:
            ep[line] = ep.get(line, 0) + 1
    nf = len(failing)
    out: dict[tuple[str, int], float] = {}
    for line in set(ef) | set(ep):
        a, b = ef.get(line, 0), ep.get(line, 0)
        out[line] = 0.0 if a == 0 else a / math.sqrt((a + (nf - a)) * (a + b))
    return out


def rank_lines(coverage: dict[str, set[tuple[str, int]]], failing_ids: list[str],
               limit: int = 50) -> list[dict]:
    failing = [coverage.get(t, set()) for t in failing_ids]
    passing = [cov for t, cov in coverage.items() if t not in set(failing_ids)]
    scores = ochiai(failing, passing)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    n = len(failing_ids)
    return [{"path": path, "line": line, "score": round(score, 4),
             "failing_hits": sum(1 for c in failing if (path, line) in c),
             "passing_hits": sum(1 for c in passing if (path, line) in c),
             "n_failing": n}
            for (path, line), score in ordered]


def symbol_ranking(lines: list[dict], index, limit: int = 20) -> list[dict]:
    """Aggregate line scores into symbols/chunks for retrieval and selection."""
    best: dict[tuple[str, str], dict] = {}
    for row in lines:
        for chunk in index.chunks:
            if chunk.path != row["path"]:
                continue
            if not (chunk.start_line <= row["line"] <= chunk.end_line):
                continue
            key = (chunk.path, chunk.symbol or f"line:{row['line']}")
            cur = best.get(key)
            if cur is None or row["score"] > cur["score"]:
                best[key] = {"path": chunk.path, "symbol": chunk.symbol,
                             "start_line": chunk.start_line, "end_line": chunk.end_line,
                             "line": row["line"], "score": row["score"]}
    return sorted(best.values(), key=lambda r: -r["score"])[:limit]


def write_plugin(directory: str) -> str:
    path = os.path.join(directory, PLUGIN)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_PLUGIN_SOURCE)
    return path


def collect(root: str, sandbox, test_ids: list[str]) -> dict[str, set[tuple[str, int]]]:
    """Run ``test_ids`` under the coverage plugin and return their line spectra.

    Instrumentation lives in a unique temp directory inside the repository (so
    the sandbox can see it) and never reuses a repository filename.  A missing
    artifact is an explicit :class:`CoverageUnavailable`, distinct from a
    legitimate run with no covered source lines.
    """
    if not test_ids:
        return {}
    scratch = tempfile.mkdtemp(prefix=".rbg-cov-", dir=root)
    rel_scratch = os.path.relpath(scratch, root)
    write_plugin(scratch)
    host_out = os.path.join(scratch, "coverage.json")
    try:
        res = sandbox.run(
            [sandbox.python, "-m", "pytest", *test_ids, "-q", "-p", PLUGIN[:-3],
             "--no-header"],
            cwd=root,
            env={"PYTHONPATH": rel_scratch, "RBG_COVERAGE_ROOT": ".",
                 "RBG_COVERAGE_FILE": f"{rel_scratch}/coverage.json"},
            timeout_s=300)
        if not os.path.exists(host_out):
            raise CoverageUnavailable(
                f"coverage artifact missing (pytest exit {res.exit_code})")
        with open(host_out, encoding="utf-8") as fh:
            raw = json.load(fh)
        imported = {(str(p), int(n)) for p, n in raw.pop("<import>", [])}
        out = {str(t): set() for t in test_ids}
        for t, lines in raw.items():
            out.setdefault(str(t), set()).update(
                (str(p), int(n)) for p, n in lines)
        if imported:
            for lines in out.values():
                lines.update(imported)
        return out
    finally:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)
