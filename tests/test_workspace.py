"""Patch-lifecycle checks: isolated start state and a durable exported diff."""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbg.workspace import Workspace  # noqa: E402


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                          text=True, check=True)


def test_non_git_root_is_copied_not_edited_in_place(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    ws = Workspace(str(root))
    work = Path(ws.setup())
    assert ws.isolated and os.path.realpath(work) != os.path.realpath(root)
    (work / "a.py").write_text("x = 2\n")
    ws.cleanup()
    assert (root / "a.py").read_text() == "x = 1\n"
    assert "--- a/a.py" in ws.patch and "+x = 2" in ws.patch


def test_git_worktree_sees_uncommitted_and_untracked_start_state(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    (root / "b.py").write_text("b = 1\n")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "-c", "user.email=t@e", "-c", "user.name=t", "commit", "-qm", "base")
    (root / "b.py").write_text("b = 2\n")          # unstaged
    (root / "c.py").write_text("c = 1\n")          # untracked
    ws = Workspace(str(root))
    work = Path(ws.setup())
    assert (work / "b.py").read_text() == "b = 2\n"
    assert (work / "c.py").read_text() == "c = 1\n"
    (work / "a.py").write_text("x = 3\n")
    ws.cleanup()
    assert (root / "a.py").read_text() == "x = 1\n"
    assert (root / "b.py").read_text() == "b = 2\n"
    assert "+x = 3" in ws.patch
    # The exported patch is the harness edit only, not the caller's dirty state.
    assert "+b = 2" not in ws.patch and "c.py" not in ws.patch


def test_failed_worktree_does_not_fall_back_to_the_checkout(monkeypatch, tmp_path):
    import subprocess

    from rbg import workspace

    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "-c", "user.email=t@e", "-c", "user.name=t", "commit", "-qm", "base")
    real = workspace._git

    def failing(cwd, *args, **kwargs):
        if args[:2] == ("worktree", "add"):
            return subprocess.CompletedProcess([], 1, b"", b"no worktree")
        return real(cwd, *args, **kwargs)

    monkeypatch.setattr(workspace, "_git", failing)
    import pytest
    with pytest.raises(RuntimeError):
        workspace.Workspace(str(root)).setup()
    assert (root / "a.py").read_text() == "x = 1\n"
