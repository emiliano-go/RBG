"""An isolated place to apply edits and run checks.

The orchestrator edits files and runs the test suite.  Doing that in the
caller's checkout can leave a failed attempt's changes behind, so every run
works in a disposable copy:

* a git checkout gets a throwaway worktree at ``HEAD`` with the caller's
  unstaged, staged, and untracked files applied, so the repair sees the real
  starting state;
* anything else is copied to a temporary directory.

There is no silent in-place fallback.  ``cleanup()`` exports the complete
unified diff before discarding the workspace, and :class:`Workspace.patch`
stays readable afterwards so a verified repair is never thrown away.
"""

from __future__ import annotations

import difflib
import os
import shutil
import subprocess
import tempfile

_IGNORE = {".git", ".hg", ".svn", ".venv", "venv", "node_modules",
           "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".rbg"}


def _git(root: str, *args: str, input: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", root, *args], capture_output=True,
                          input=input, text=False)


def _is_toplevel_git(root: str) -> bool:
    try:
        r = _git(root, "rev-parse", "--show-toplevel")
    except OSError:
        return False
    return r.returncode == 0 and os.path.realpath(r.stdout.decode().strip()) == os.path.realpath(root)


def _copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if os.path.islink(src):
        os.symlink(os.readlink(src), dst)
    else:
        shutil.copy2(src, dst)


def _copy_untracked(root: str, dest: str) -> None:
    names = _git(root, "ls-files", "--others", "--exclude-standard", "-z").stdout
    for raw in names.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode(errors="surrogateescape")
        if any(part in _IGNORE for part in rel.split(os.sep)):
            continue
        src = os.path.join(root, rel)
        if not os.path.lexists(src):
            continue
        _copy(src, os.path.join(dest, rel))


def _apply_start_state(root: str, dest: str) -> None:
    diff = _git(root, "diff", "--binary", "HEAD").stdout
    if diff:
        _git(dest, "apply", "--binary", "--whitespace=nowarn", input=diff)
    _copy_untracked(root, dest)


def _snapshot(root: str, dest: str) -> None:
    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {n for n in names if n in _IGNORE}
    shutil.copytree(root, dest, ignore=ignore, symlinks=True, dirs_exist_ok=True)


def _plain_patch(before: str, after: str) -> str:
    parts: list[str] = []
    for base, _dirs, files in os.walk(before):
        for name in files:
            src = os.path.join(base, name)
            rel = os.path.relpath(src, before)
            if any(part in _IGNORE for part in rel.split(os.sep)):
                continue
            dst = os.path.join(after, rel)
            if not os.path.exists(dst):
                old, new, label = _read(src), "", rel
                parts.append(_unified(old, new, label, deleted=True))
            else:
                old, new = _read(src), _read(dst)
                if old != new:
                    parts.append(_unified(old, new, rel))
    for base, _dirs, files in os.walk(after):
        for name in files:
            dst = os.path.join(base, name)
            rel = os.path.relpath(dst, after)
            if any(part in _IGNORE for part in rel.split(os.sep)):
                continue
            if not os.path.exists(os.path.join(before, rel)):
                parts.append(_unified("", _read(dst), rel, added=True))
    return "".join(parts)


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _unified(old: str, new: str, rel: str, *, added: bool = False,
             deleted: bool = False) -> str:
    a = "/dev/null" if added else f"a/{rel}"
    b = "/dev/null" if deleted else f"b/{rel}"
    old_lines = [] if added else old.splitlines(keepends=True)
    new_lines = [] if deleted else new.splitlines(keepends=True)
    return "".join(difflib.unified_diff(old_lines, new_lines, fromfile=a, tofile=b))


class Workspace:
    """``setup()`` returns the directory to edit/verify in; ``cleanup()``
    exports ``patch`` and discards it.  ``isolated`` is always true: the
    caller's checkout is never the edit target."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.path = self.root
        self.isolated = False
        self.patch = ""
        self._tmp: str | None = None
        self._baseline: str | None = None
        self._base_tree: str | None = None

    def setup(self) -> str:
        if _is_toplevel_git(self.root):
            return self._setup_git()
        return self._setup_copy()

    def _setup_git(self) -> str:
        tmp = tempfile.mkdtemp(prefix="rbg-wt-")
        if _git(self.root, "worktree", "add", "--detach", tmp, "HEAD").returncode != 0:
            shutil.rmtree(tmp, ignore_errors=True)
            raise RuntimeError("could not create an isolated git worktree")
        try:
            _apply_start_state(self.root, tmp)
            add = _git(tmp, "add", "-A")
            if add.returncode != 0:
                raise RuntimeError("could not record the workspace baseline")
            tree = _git(tmp, "write-tree").stdout.decode().strip()
            if not tree:
                raise RuntimeError("could not record the workspace baseline")
        except Exception:
            _git(self.root, "worktree", "remove", "--force", tmp)
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        self._base_tree = tree
        self._tmp, self.path, self.isolated = tmp, tmp, True
        # Reuse the checkout's virtualenv/deps: git-ignored, so absent here.
        for name in (".venv", "venv", "node_modules"):  # ponytail: symlink, not copy
            src = os.path.join(self.root, name)
            dst = os.path.join(tmp, name)
            if os.path.isdir(src) and not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                except OSError:
                    pass
        return tmp

    def _setup_copy(self) -> str:
        tmp = tempfile.mkdtemp(prefix="rbg-copy-")
        baseline = tempfile.mkdtemp(prefix="rbg-base-")
        try:
            _snapshot(self.root, baseline)
            _snapshot(self.root, tmp)
            for name in (".venv", "venv", "node_modules"):
                src = os.path.join(self.root, name)
                dst = os.path.join(tmp, name)
                if os.path.isdir(src) and not os.path.exists(dst):
                    try:
                        os.symlink(src, dst)
                    except OSError:
                        pass
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(baseline, ignore_errors=True)
            raise
        self._tmp, self._baseline = tmp, baseline
        self.path, self.isolated = tmp, True
        return tmp

    def _deps(self) -> list[str]:
        return [n for n in (".venv", "venv", "node_modules")
                if os.path.islink(os.path.join(self.path, n))]

    def cleanup(self) -> None:
        if not self._tmp:
            return
        try:
            if self._base_tree:
                for name in self._deps():
                    path = os.path.join(self.path, name)
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                _git(self.path, "add", "-A")
                self.patch = _git(self.path, "diff", "--cached", "--binary",
                                  self._base_tree).stdout.decode(errors="replace")
            elif self._baseline:
                self.patch = _plain_patch(self._baseline, self.path)
        finally:
            if self._base_tree:
                _git(self.root, "worktree", "remove", "--force", self._tmp)
            shutil.rmtree(self._tmp, ignore_errors=True)
            if self._baseline:
                shutil.rmtree(self._baseline, ignore_errors=True)
            self._tmp = self._baseline = self._base_tree = None
            self.path = self.root
            self.isolated = False
