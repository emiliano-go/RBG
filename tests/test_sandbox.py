"""Containment and sandbox-boundary checks (spec: security boundary).

Run: ``python3 -m pytest tests -q``
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import isolation  # noqa: E402
from harness.isolation import IsolationUnavailable  # noqa: E402
from harness.paths import resolve_within  # noqa: E402
from harness.verify import Sandbox  # noqa: E402


def test_relative_path_resolves_under_root(tmp_path):
    assert resolve_within(str(tmp_path), "a/b.py") == str(tmp_path / "a" / "b.py")


def test_absolute_path_inside_root_ok(tmp_path):
    assert resolve_within(str(tmp_path), str(tmp_path / "x.py")) == str(tmp_path / "x.py")


def test_new_nonexistent_file_allowed(tmp_path):
    got = resolve_within(str(tmp_path), "new/dir/created.py")
    assert got == str(tmp_path / "new" / "dir" / "created.py")


def test_sibling_prefix_is_rejected(tmp_path):
    root = tmp_path / "repo"
    escape = tmp_path / "repo-escape"
    root.mkdir()
    escape.mkdir()
    with pytest.raises(PermissionError):
        resolve_within(str(root), str(escape / "secret.py"))


def test_parent_traversal_is_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    with pytest.raises(PermissionError):
        resolve_within(str(root), "../outside.py")


def test_symlink_escape_is_rejected(tmp_path):
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "target.py").write_text("x = 1\n")
    link = root / "link.py"
    try:
        os.symlink(outside / "target.py", link)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(PermissionError):
        resolve_within(str(root), "link.py")


def test_symlink_inside_root_is_allowed(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "real.py").write_text("x = 1\n")
    link = root / "alias.py"
    try:
        os.symlink(root / "real.py", link)
    except OSError:
        pytest.skip("symlinks unavailable")
    assert resolve_within(str(root), "alias.py") == str(root / "real.py")


# -- environment allowlist -------------------------------------------------- #
def test_unnamed_secret_is_stripped(tmp_path, monkeypatch):
    sb = Sandbox(str(tmp_path))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "leak")
    assert "AWS_ACCESS_KEY_ID" not in sb._env()


def test_safe_vars_are_kept(tmp_path, monkeypatch):
    sb = Sandbox(str(tmp_path))
    monkeypatch.setenv("HOME", "/home/x")
    assert sb._env().get("HOME") == "/home/x"


def test_allowlist_extension_is_honoured(tmp_path, monkeypatch):
    sb = Sandbox(str(tmp_path))
    monkeypatch.setenv("HARNESS_ENV_ALLOW", "MY_VAR")
    monkeypatch.setenv("MY_VAR", "ok")
    assert sb._env().get("MY_VAR") == "ok"


def test_sandbox_rejects_shell_string(tmp_path):
    sb = Sandbox(str(tmp_path))
    with pytest.raises(TypeError):
        sb.run("echo hi")


# -- isolation -------------------------------------------------------------- #
def test_isolation_is_applied_when_backend_exists(tmp_path):
    if isolation.available_backend() is None:
        pytest.skip("no isolation backend on this host")
    res = Sandbox(str(tmp_path)).run(["python3", "-c", "print('hi')"])
    assert res.isolated is True
    assert res.isolation in ("bwrap", "nsjail")
    assert res.network_disabled is True
    assert res.ok and "hi" in res.stdout


def test_run_refuses_without_backend_or_hatch(tmp_path, monkeypatch):
    monkeypatch.setattr(isolation, "available_backend", lambda: None)
    monkeypatch.delenv("HARNESS_ALLOW_UNISOLATED", raising=False)
    with pytest.raises(IsolationUnavailable):
        Sandbox(str(tmp_path)).run(["python3", "-c", "print('hi')"])


def test_unisolated_hatch_marks_result(tmp_path, monkeypatch):
    monkeypatch.setattr(isolation, "available_backend", lambda: None)
    monkeypatch.setenv("HARNESS_ALLOW_UNISOLATED", "1")
    res = Sandbox(str(tmp_path)).run(["python3", "-c", "print('hi')"])
    assert res.ok and res.isolated is False and res.isolation == "none"
