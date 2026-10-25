"""Ochiai scoring and Python line-coverage collection."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbg import fault_localization as fl  # noqa: E402
from rbg.verify import Sandbox  # noqa: E402


def test_ochiai_ranks_failing_only_lines_highest():
    failing = [{("a.py", 1), ("a.py", 2)}, {("a.py", 2)}]
    passing = [{("a.py", 1)}]
    scores = fl.ochiai(failing, passing)
    assert scores[("a.py", 2)] == 1.0
    assert scores[("a.py", 1)] < scores[("a.py", 2)]
    assert scores[("a.py", 3)] if ("a.py", 3) in scores else True


def test_rank_lines_reports_test_spectrum_counts():
    coverage = {"t_fail": {("m.py", 3)}, "t_pass": {("m.py", 1)}}
    rows = fl.rank_lines(coverage, ["t_fail"])
    top = rows[0]
    assert top["path"] == "m.py" and top["line"] == 3
    assert top["failing_hits"] == 1 and top["passing_hits"] == 0


def test_python_coverage_collects_failing_and_passing_lines(tmp_path):
    if fl.os.environ.get("RBG_ALLOW_UNISOLATED") != "1":
        from rbg import isolation
        if isolation.available_backend() is None:
            import pytest
            pytest.skip("no isolation backend on this host")
    (tmp_path / "m.py").write_text(
        "def add(a, b):\n"
        "    total = a - b\n"
        "    return total\n")
    (tmp_path / "test_m.py").write_text(
        "from m import add\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n"
        "def test_ok():\n"
        "    assert add(1, 1) == 0\n")
    sb = Sandbox(str(tmp_path))
    cov = fl.collect(str(tmp_path), sb, ["test_m.py::test_add", "test_m.py::test_ok"])
    assert "test_m.py::test_add" in cov and "test_m.py::test_ok" in cov
    assert ("m.py", 2) in cov["test_m.py::test_add"]
    rows = fl.rank_lines(cov, ["test_m.py::test_add"])
    assert rows and rows[0]["path"] == "m.py"


def test_collector_does_not_clobber_repository_files(tmp_path):
    if fl.os.environ.get("RBG_ALLOW_UNISOLATED") != "1":
        from rbg import isolation
        if isolation.available_backend() is None:
            import pytest
            pytest.skip("no isolation backend on this host")
    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    (tmp_path / "test_m.py").write_text("from m import f\n\ndef test_f():\n    assert f() == 1\n")
    (tmp_path / "_rbg_coverage_plugin.py").write_text("sentinel plugin\n")
    (tmp_path / ".rbg-coverage.json").write_text("sentinel output\n")
    fl.collect(str(tmp_path), Sandbox(str(tmp_path)), ["test_m.py::test_f"])
    assert (tmp_path / "_rbg_coverage_plugin.py").read_text() == "sentinel plugin\n"
    assert (tmp_path / ".rbg-coverage.json").read_text() == "sentinel output\n"
