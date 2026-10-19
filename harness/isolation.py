"""OS-level isolation for sandboxed commands (spec 5.1, 10).

Real isolation is *required*: markers in the environment are not a boundary.
Each backend is probed once with a trivial command; a backend that cannot
actually start is treated as unavailable rather than trusted.  When no backend
works, :func:`select` raises unless ``HARNESS_ALLOW_UNISOLATED=1`` is set, in
which case the command runs bare and the result is marked unisolated so it can
never count as verified.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field

_BACKENDS = ("bwrap", "nsjail")
_probe_cache: dict[str, bool] = {}


class IsolationUnavailable(RuntimeError):
    """No usable isolation backend and the unisolated escape hatch is off."""


@dataclass(frozen=True)
class Isolation:
    backend: str            # "bwrap" | "nsjail" | "none"
    isolated: bool
    network_disabled: bool
    prefix: tuple[str, ...] = field(default_factory=tuple)


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no")


def _wrap(backend: str, cwd: str, network: bool) -> list[str]:
    if backend == "bwrap":
        argv = [
            shutil.which("bwrap") or "bwrap",
            "--ro-bind", "/", "/",
            "--dev", "/dev",
            "--proc", "/proc",
            "--bind", "/tmp", "/tmp",
            "--die-with-parent",
            "--new-session",
        ]
        if not network:
            argv.append("--unshare-net")
        return [*argv, "--"]
    # nsjail: recipe is probed before use, so a wrong flag set fails closed.
    argv = [
        shutil.which("nsjail") or "nsjail",
        "--quiet", "--mode", "o", "--chroot", "/", "--cwd", cwd,
        "--bindmount_ro", "/:/", "--tmpfs", "/tmp",
    ]
    if network:
        argv.append("--disable_clone_newnet")
    return [*argv, "--"]


def _probe(backend: str) -> bool:
    if backend in _probe_cache:
        return _probe_cache[backend]
    binary = shutil.which(backend)
    if not binary:
        _probe_cache[backend] = False
        return False
    argv = [*_wrap(backend, os.getcwd(), network=True), "true"]
    try:
        ok = subprocess.run(argv, capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        ok = False
    _probe_cache[backend] = ok
    return ok


def available_backend() -> str | None:
    """First backend that both exists and passes its probe, else ``None``."""
    for backend in _BACKENDS:
        if _probe(backend):
            return backend
    return None


def select(network: bool) -> Isolation:
    backend = available_backend()
    if backend:
        return Isolation(backend, True, not network,
                         tuple(_wrap(backend, os.getcwd(), network)))
    if _truthy("HARNESS_ALLOW_UNISOLATED"):
        return Isolation("none", False, False, ())
    raise IsolationUnavailable(
        "no usable isolation backend (tried bwrap, nsjail); refusing to run. "
        "Set HARNESS_ALLOW_UNISOLATED=1 for development only.")
