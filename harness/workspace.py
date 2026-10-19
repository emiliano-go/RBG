"""A disposable place to apply edits and run checks.

The orchestrator edits files and runs the test suite; doing that in the
caller's checkout leaves a failed attempt's changes behind.  When ``root`` is
the top of a git work tree we do the work in a throwaway worktree and remove
it afterwards, so the caller's tree is never mutated.  Otherwise (no git, or
``root`` is not the repository top) we fall back to editing in place.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile


def _git(root: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)


def _is_toplevel_git(root: str) -> bool:
    try:
        r = _git(root, "rev-parse", "--show-toplevel")
    except OSError:
        return False
    return r.returncode == 0 and os.path.realpath(r.stdout.strip()) == os.path.realpath(root)


class Workspace:
    """``setup()`` returns the directory to edit/verify in; ``cleanup()``
    discards it.  Known by :attr:`isolated` whether a throwaway worktree was
    used."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.path = self.root
        self.isolated = False
        self._tmp: str | None = None

    def setup(self) -> str:
        if not _is_toplevel_git(self.root):
            return self.path
        tmp = tempfile.mkdtemp(prefix="rbg-wt-")
        if _git(self.root, "worktree", "add", "--detach", tmp, "HEAD").returncode != 0:
            shutil.rmtree(tmp, ignore_errors=True)
            return self.path
        # Reuse the checkout's virtualenv/deps: git-ignored, so absent here.
        for name in (".venv", "venv", "node_modules"):  # ponytail: symlink, not copy
            src = os.path.join(self.root, name)
            dst = os.path.join(tmp, name)
            if os.path.isdir(src) and not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                except OSError:
                    pass
        self._tmp, self.path, self.isolated = tmp, tmp, True
        return tmp

    def cleanup(self) -> None:
        if not self._tmp:
            return
        _git(self.root, "worktree", "remove", "--force", self._tmp)
        shutil.rmtree(self._tmp, ignore_errors=True)
        self._tmp = None
        self.path = self.root
        self.isolated = False
