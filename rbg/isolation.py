"""OS-level isolation for sandboxed commands (spec 5.1, 10).

Real isolation is *required*: markers in the environment are not a boundary.
Each backend is probed once with a command that asserts the boundary (writes
outside the worktree fail, host home is hidden, the network namespace is empty,
and the PID namespace is new); a backend that cannot prove those properties is
treated as unavailable.  When no backend works, :func:`select` raises unless
``RBG_ALLOW_UNISOLATED=1`` is set, in which case the command runs bare and
the result is marked unisolated so it can never count as verified.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

WORKSPACE = "/tmp/workspace"  # /tmp is a writable tmpfs, so the mountpoint exists

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
    workspace: str = WORKSPACE


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no")


def _under(path: str, parent: str) -> bool:
    path, parent = os.path.realpath(path), os.path.realpath(parent)
    return path == parent or path.startswith(parent.rstrip(os.sep) + os.sep)


# Every command sees the candidate repository at this fixed path (under the
# writable /tmp tmpfs, since bwrap cannot create a mountpoint at /).  Host home
# directories are hidden unconditionally, so a repo under /home cannot expose
# the rest of /home through read-only access.


# Host data directories that are not needed to run code.  A read-only bind of
# ``/`` still exposes their contents, and command output can carry it back to the
# model, so mask them before adding the workspace and explicit dependencies.
_HOST_MASK = ("/var", "/mnt", "/media", "/srv", "/boot", "/opt", "/run")
# Also hidden by explicit tmpfs mounts, so a kept runtime beneath them must be
# rebound after the mask.
_WORKSPACE_MASKS = ("/tmp", "/home", "/root")


def _wrap(backend: str, network: bool, root: str,
          ro_binds: tuple[tuple[str, str], ...] = (),
          keep: tuple[str, ...] = ()) -> list[str]:
    root = os.path.realpath(root)
    if backend == "bwrap":
        argv = [
            shutil.which("bwrap") or "bwrap",
            "--ro-bind", "/", "/",
            "--dev", "/dev",
            "--proc", "/proc",
            "--size", "268435456", "--tmpfs", "/tmp",
            "--tmpfs", "/root",
            "--tmpfs", "/home",
            "--ro-bind-try", "/etc", "/etc",
            "--die-with-parent",
            "--new-session",
            "--cap-drop", "ALL",
        ]
        for path in _HOST_MASK:
            if os.path.exists(path):
                argv += ["--tmpfs", path]
        # Re-expose only the interpreter runtime that lives under a masked path
        # (hosted CI puts Python in /opt); the rest of the mask stays empty.
        for path in keep:
            if any(_under(path, masked) for masked in (*_HOST_MASK, *_WORKSPACE_MASKS)):
                argv += ["--ro-bind", path, path]
        # The candidate repo is the only writable host path inside the sandbox;
        # a project venv is mounted read-only at a stable sandbox path.
        for src, dst in ro_binds:
            argv += ["--ro-bind", src, dst]
        argv += ["--bind", root, WORKSPACE]
        argv += ["--unshare-all"]
        if network:
            argv.append("--share-net")
        return [*argv, "--"]
    # nsjail: recipe is probed before use, so a wrong flag set fails closed.
    argv = [
        shutil.which("nsjail") or "nsjail",
        "--quiet", "--mode", "o", "--chroot", "/",
        "--bindmount_ro", "/:/", "--tmpfs", "/tmp",
        "--tmpfs", "/root", "--tmpfs", "/home",
        "--rlimit_as", "4096", "--rlimit_cpu", "60", "--rlimit_nproc", "512",
        "--rlimit_fsize", "64", "--rlimit_nofile", "512",
        "--time_limit", "60",
        "--bindmount", f"{root}:{WORKSPACE}",
    ]
    for path in _HOST_MASK:
        if os.path.exists(path):
            argv += ["--tmpfs", path]
    for path in keep:
        if any(_under(path, masked) for masked in (*_HOST_MASK, *_WORKSPACE_MASKS)):
            argv += ["--bindmount_ro", f"{path}:{path}"]
    for src, dst in ro_binds:
        argv += ["--bindmount_ro", f"{src}:{dst}"]
    if not network:
        argv.append("--disable_clone_newnet")
    return [*argv, "--"]


_PROBE = r"""
set -e
host_pidns="$1"
# A write outside the worktree must fail (the root is mounted read-only).
if touch /usr/.rbg-probe-$$ 2>/dev/null; then exit 11; fi
# Host homes are hidden; a probe run must not see the caller's keys.
if [ -e "$HOME/.ssh" ]; then exit 12; fi
# Offline probe: no interface other than loopback.
if awk 'NR>2 && $1 != "lo:" { found=1 } END { exit found ? 0 : 1 }' /proc/net/dev; then
  exit 13
fi
# A new PID namespace means our pid namespace inode differs from the host's.
now=$(stat -Lc %i /proc/self/ns/pid)
if [ "$now" = "$host_pidns" ]; then exit 14; fi
exit 0
"""


def _probe(backend: str) -> bool:
    if backend in _probe_cache:
        return _probe_cache[backend]
    binary = shutil.which(backend)
    if not binary:
        _probe_cache[backend] = False
        return False
    root = tempfile.mkdtemp(prefix="rbg-probe-")
    try:
        host_pidns = os.stat("/proc/self/ns/pid").st_ino
        argv = [*_wrap(backend, network=False, root=root),
                shutil.which("sh") or "/bin/sh", "-c", _PROBE, "rbg-probe",
                str(host_pidns)]
        ok = subprocess.run(argv, capture_output=True, timeout=15,
                            env={**os.environ, "HOME": "/nonexistent"}).returncode == 0
    except (OSError, subprocess.SubprocessError):
        ok = False
    finally:
        shutil.rmtree(root, ignore_errors=True)
    _probe_cache[backend] = ok
    return ok


def available_backend() -> str | None:
    """First backend that exists and proves its boundary, else ``None``."""
    for backend in _BACKENDS:
        if _probe(backend):
            return backend
    return None


def select(network: bool, root: str | None = None,
           ro_binds: tuple[tuple[str, str], ...] = (),
           keep: tuple[str, ...] = ()) -> Isolation:
    backend = available_backend()
    workdir = os.path.realpath(root or os.getcwd())
    if backend:
        return Isolation(backend, True, not network,
                         tuple(_wrap(backend, network, workdir, ro_binds, keep)))
    if _truthy("RBG_ALLOW_UNISOLATED"):
        return Isolation("none", False, False, ())
    raise IsolationUnavailable(
        "no usable isolation backend (tried bwrap, nsjail); refusing to run. "
        "Set RBG_ALLOW_UNISOLATED=1 for development only.")
