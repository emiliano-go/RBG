"""Path containment: the single way to turn an untrusted path into an absolute
one inside a repository root.

A bare ``str.startswith`` check is wrong twice over: a sibling such as
``/work/repo-escape`` shares the prefix of ``/work/repo``, and ``abspath`` does
not resolve symlink targets.  Resolve both sides with ``realpath`` and compare
with ``commonpath`` instead, so neither ``..`` nor a symlink can leave the root.
"""

from __future__ import annotations

import os


def resolve_within(root: str, path: str) -> str:
    """Absolute path for ``path`` under ``root``, or ``PermissionError``.

    ``path`` may be relative (joined to ``root``) or absolute.  Symlinks are
    resolved before the containment check; a path that resolves at or inside
    the root is returned, anything else is rejected.
    """
    root_real = os.path.realpath(os.path.abspath(root))
    candidate = path if os.path.isabs(path) else os.path.join(root_real, path)
    target = os.path.realpath(candidate)
    try:
        within = os.path.commonpath([root_real, target]) == root_real
    except ValueError:  # e.g. different drives on Windows
        within = False
    if not within:
        raise PermissionError(f"path escapes root: {path!r}")
    return target
