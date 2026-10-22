"""Collect the working-tree change set without asking the agent.

The drift audit must not depend on the model reporting what it changed. This
module reads git directly: tracked modifications, staged changes, deletions,
renames, and untracked files. When the root is not a git checkout it returns
an empty set with ``git=False`` so the caller can report the limitation
instead of treating "no evidence" as "no drift".
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field


@dataclass
class ChangedSet:
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    renamed: list[tuple[str, str]] = field(default_factory=list)
    untracked: list[str] = field(default_factory=list)
    git: bool = True
    base: str = "HEAD"
    error: str = ""

    def files(self) -> list[str]:
        out = set(self.added) | set(self.modified) | set(self.deleted)
        out |= set(self.untracked)
        for old, new in self.renamed:
            out.add(old)
            out.add(new)
        return sorted(p for p in out if p)

    def live(self) -> list[str]:
        """Files that exist after the change (renames count their new path)."""
        out = set(self.added) | set(self.modified) | set(self.untracked)
        out |= {new for _old, new in self.renamed}
        return sorted(p for p in out if p)

    def to_dict(self) -> dict:
        return {
            "git": self.git,
            "base": self.base,
            "added": self.added,
            "modified": self.modified,
            "deleted": self.deleted,
            "renamed": [{"old": old, "new": new} for old, new in self.renamed],
            "untracked": self.untracked,
            "files": self.files(),
            "error": self.error,
        }


def _run(root: str, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "-C", root, *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(
            ["git", "-C", root, *args], 1, "", str(exc)
        )


def _norm(path: str) -> str:
    return path.replace("\\", "/").removeprefix("./")


def _parse_name_status(text: str) -> tuple[list, list, list, list]:
    added, modified, deleted, renamed = [], [], [], []
    parts = [p for p in text.split("\0") if p]
    index = 0
    while index < len(parts):
        status = parts[index]
        index += 1
        if not status:
            continue
        code = status[0]
        if code in ("R", "C"):
            if index + 1 >= len(parts):
                break
            old, new = _norm(parts[index]), _norm(parts[index + 1])
            index += 2
            renamed.append((old, new))
        else:
            if index >= len(parts):
                break
            path = _norm(parts[index])
            index += 1
            if code == "A":
                added.append(path)
            elif code == "D":
                deleted.append(path)
            else:
                modified.append(path)
    return added, modified, deleted, renamed


def git_changed(root: str, base: str = "HEAD") -> ChangedSet:
    """Tracked and untracked changes relative to *base*."""

    root = os.path.abspath(root)
    probe = _run(root, "rev-parse", "--is-inside-work-tree")
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return ChangedSet(git=False, base=base)
    # Resolve the revision before it reaches the command line: a base that
    # starts with "-" would otherwise be read as a git option.
    resolved = _run(root, "rev-parse", "--verify", "--end-of-options",
                    f"{base}^{{commit}}")
    if resolved.returncode != 0:
        resolved = _run(root, "rev-parse", "--verify", "--end-of-options",
                        "HEAD^{commit}")
    revision = (resolved.stdout or "").strip() if resolved.returncode == 0 else ""
    error = ""
    if not revision:
        diff = subprocess.CompletedProcess([], 1, "", "no commit")
        error = "no commit to diff against"
    else:
        diff = _run(root, "diff", "--name-status", "-z", revision)
        if diff.returncode != 0:
            error = (diff.stderr or "git diff failed").strip()[:200]
    added, modified, deleted, renamed = _parse_name_status(diff.stdout or "")
    untracked_out = _run(root, "ls-files", "--others", "--exclude-standard", "-z")
    untracked = sorted(_norm(p) for p in (untracked_out.stdout or "").split("\0") if p)
    return ChangedSet(
        added=sorted(set(added)),
        modified=sorted(set(modified)),
        deleted=sorted(set(deleted)),
        renamed=sorted(set(renamed)),
        untracked=untracked,
        git=True,
        base=revision or base,
        error=error,
    )


def symbols_in(changed: ChangedSet, index) -> list[str]:
    """Best-effort symbols reachable from the changed files.

    Without a pre-change index this is the current symbol set for those
    files, which is enough for the "symbol changed but not declared" check.
    """

    files = set(changed.live())
    if not index or not files:
        return []
    return sorted({s.name for s in index.symbols if _norm(s.path) in files})
