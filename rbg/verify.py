"""Sandboxed tool execution, verification gates, and the structured ledger.

Everything the main LLM proposes runs through :class:`Sandbox`, which applies
timeouts, resource caps, and a scrubbed environment (spec 5.1, 10).  Results
are hard evidence: exit codes and parsed test outcomes, never model claims
(spec 3.2, 6.11).  :class:`Ledger` writes the versioned JSONL envelope from
spec section 7.
"""

from __future__ import annotations

import os
import re
import resource
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .checks import Toolchain, compile_command, detect, runner_argv
from .isolation import Isolation
from .isolation import select as select_isolation
from .paths import resolve_within

SCHEMA_VERSION = 1
VENV_MOUNT = "/tmp/venv"
# An allowlist, not a denylist: a secret is leaked by any variable we forget to
# name, so only a known-safe set of variables reaches a sandboxed command.
_ENV_ALLOW = ("PATH", "HOME", "LANG", "TMPDIR", "TEMP", "TMP", "USER",
              "LOGNAME", "VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")
_ENV_ALLOW_PREFIX = ("LC_", "PYTHON")


@dataclass
class CommandResult:
    command: str
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False
    isolation: str = "none"
    isolated: bool = False
    network_disabled: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict:
        return {
            "command": self.command, "cwd": self.cwd, "exit_code": self.exit_code,
            "duration_ms": self.duration_ms, "timed_out": self.timed_out,
            "isolation": self.isolation, "isolated": self.isolated,
            "network_disabled": self.network_disabled,
            "stdout_tail": self.stdout[-2000:], "stderr_tail": self.stderr[-2000:],
        }


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))


class Sandbox:
    """Runs commands with least privilege: no inherited secrets, capped time
    and CPU/file resources."""

    def __init__(self, root: str, timeout_s: int = 60, network: bool = False) -> None:
        self.root = os.path.abspath(root)
        self.timeout_s = timeout_s
        self.network = network
        self.python = self._resolve_python(self.root)
        self.venv = self._venv_root(self.python)
        self.runtime = self._runtime_prefixes(self.python, self.venv)
        self.python_sandbox = f"{VENV_MOUNT}/bin/python" if self.venv else self.python
        self.log: list[CommandResult] = []
        self._isolation: Isolation | None = None

    def isolation(self) -> Isolation:
        """Selected once per sandbox; raises if no backend and no escape hatch."""
        if self._isolation is None:
            binds = ((self.venv, VENV_MOUNT),) if self.venv else ()
            keep = tuple(p for p in self.runtime if p != self.venv)
            self._isolation = select_isolation(self.network, self.root, binds,
                                               keep=keep)
        return self._isolation

    @staticmethod
    def _runtime_prefixes(python: str, venv: str | None) -> tuple[str, ...]:
        """Host runtime directories the interpreter needs, even if masked.

        Hosted CI keeps Python under ``/opt``.  /opt is masked for privacy, so
        the base prefix (and its shared libraries) must be re-exposed at the
        same path, while unrelated /opt contents stay hidden.
        """
        paths: set[str] = set()
        if venv:
            paths.add(os.path.realpath(venv))
            cfg = os.path.join(venv, "pyvenv.cfg")
            try:
                with open(cfg, encoding="utf-8") as fh:
                    for line in fh:
                        key, _, value = line.partition("=")
                        if key.strip() == "home" and value.strip():
                            home = os.path.realpath(value.strip())
                            paths.add(os.path.dirname(home))
            except OSError:
                pass
        exe = python if os.path.exists(python) else (shutil.which(python) or python)
        # Follow the symlink chain and keep every prefix it names: a venv's
        # python often points at a versioned directory that itself symlinks to
        # the patch release, and the first target must exist inside the sandbox
        # before the loader can follow the next hop.
        current = os.path.abspath(exe)
        seen: set[str] = set()
        while os.path.islink(current) and current not in seen:
            seen.add(current)
            target = os.readlink(current)
            if not os.path.isabs(target):
                target = os.path.join(os.path.dirname(current), target)
            current = os.path.normpath(target)
            parent = os.path.dirname(os.path.dirname(current))
            if os.path.isdir(parent):
                paths.add(parent)
        real = os.path.realpath(exe)
        if os.path.basename(os.path.dirname(real)) == "bin":
            paths.add(os.path.dirname(os.path.dirname(real)))
        return tuple(sorted(p for p in paths if os.path.isdir(p)))

    @staticmethod
    def _venv_root(python: str) -> str | None:
        """Identify a virtualenv from the executable path and its pyvenv.cfg.

        Do not resolve the executable first: ``venv/bin/python`` is usually a
        symlink to the base interpreter, and realpath would identify the base
        prefix instead of the project environment.  Resolve only the discovered
        environment root (which may itself be a symlink).
        """
        if python == "python3" or not os.path.exists(python):
            return None
        candidate = os.path.abspath(python)
        bin_dir = os.path.dirname(candidate)
        if os.path.basename(bin_dir) != "bin":
            return None
        root = os.path.dirname(bin_dir)
        if not os.path.isfile(os.path.join(root, "pyvenv.cfg")):
            return None
        return os.path.realpath(root)

    @staticmethod
    def _resolve_python(root: str) -> str:
        """Prefer the project's own virtualenv so checks run with its deps."""
        for candidate in (".venv/bin/python", "venv/bin/python", ".venv/bin/python3"):
            path = os.path.join(root, candidate)
            if os.path.exists(path):
                return path
        return "python3"

    def _env(self) -> dict[str, str]:
        allow = set(_ENV_ALLOW)
        allow.update(e for e in os.environ.get("RBG_ENV_ALLOW", "").split(",") if e)
        env = {}
        for k, v in os.environ.items():
            if k in allow or any(k.startswith(p) for p in _ENV_ALLOW_PREFIX):
                env[k] = v
        env["RBG_SANDBOX"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # The repository is mounted read-only; compile checks still want to write
        # bytecode, so redirect it beneath the writable /tmp the sandbox exposes.
        pycache = os.path.join(tempfile.gettempdir(), "rbg-pycache")
        try:
            os.makedirs(pycache, exist_ok=True)
        except OSError:
            pass
        env["PYTHONPYCACHEPREFIX"] = pycache
        if self.venv:
            # A host venv is mounted at a stable path; its original location may
            # be hidden (or outside the workspace), so rewrite the pointer.
            env["VIRTUAL_ENV"] = VENV_MOUNT
            env["PATH"] = VENV_MOUNT + "/bin:" + env.get("PATH", "")
            env.pop("PYTHONHOME", None)
        if not self.network:
            # advisory markers; a real deployment would add a netns/firewall
            env["NO_NETWORK"] = "1"
        return env

    def _translate(self, arg: str, iso: Isolation) -> str:
        if arg == self.python or arg == self.python_sandbox:
            return self.python_sandbox
        if self.venv and arg.startswith(self.venv + os.sep):
            return VENV_MOUNT + arg[len(self.venv):]
        if arg.startswith(self.root + os.sep):
            return iso.workspace + arg[len(self.root):]
        return arg

    def run(self, argv: list[str], cwd: str | None = None, timeout_s: int | None = None,
            env: dict[str, str] | None = None) -> CommandResult:
        if not argv or isinstance(argv, str):
            raise TypeError("Sandbox.run takes a non-empty argv list, never a shell string")
        cwd = resolve_within(self.root, cwd or self.root)
        iso = self.isolation()
        prefix = list(iso.prefix)
        if iso.isolated:
            # The repo is mounted at a fixed path; translate the host cwd and
            # any absolute argv paths that point into the workspace or venv.
            rel = os.path.relpath(cwd, self.root)
            sandbox_cwd = iso.workspace if rel == "." else f"{iso.workspace}/{rel}"
            flag = "--chdir" if iso.backend == "bwrap" else "--cwd"
            prefix[-1:-1] = [flag, sandbox_cwd]
            argv = [self._translate(arg, iso) for arg in argv]
        full = [*prefix, *argv]
        command = " ".join(argv)
        started = time.monotonic()
        timed_out = False
        try:
            run_env = self._env()
            run_env.update(env or {})
            proc = subprocess.run(
                full, shell=False, cwd=cwd, env=run_env,
                capture_output=True, text=True, timeout=timeout_s or self.timeout_s,
                preexec_fn=_limits,
            )
            code, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            code = 124
            out = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            err = (exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        result = CommandResult(command, cwd, code, out, err,
                               int((time.monotonic() - started) * 1000), timed_out,
                               iso.backend, iso.isolated, iso.network_disabled)
        self.log.append(result)
        return result


@dataclass
class TestOutcome:
    __test__ = False

    test_id: str
    status: str  # passed | failed | error
    duration_s: float = 0.0
    message: str = ""


_PYTEST_LINE = re.compile(r"^(?P<id>[\w./:\\-]+::[\w:\[\].\-]+)\s+(?P<status>PASSED|FAILED|ERROR|SKIPPED)")
_PYTEST_SUMMARY = re.compile(r"^(?P<status>PASSED|FAILED|ERROR|SKIPPED)\s+(?P<id>[\w./:\\-]+::[\w:\[\].\-]+)")
_PYTEST_SKIP_LINE = re.compile(r"^SKIPPED\s+\[\d+\]\s+(?P<id>[^:]+):(?P<line>\d+)")
_STATUS = {"PASSED": "passed", "FAILED": "failed", "ERROR": "error", "SKIPPED": "skipped"}
_PYTEST_DURATION = re.compile(
    r"^(?P<duration>\d+(?:\.\d+)?)s\s+(?:call|setup|teardown|collect)\s+"
    r"(?P<id>[\w./:\\-]+::[\w:\[\].\-]+)")


def parse_pytest_durations(output: str) -> dict[str, float]:
    """Per-test wall time from ``pytest --durations=0`` (last phase wins)."""
    out: dict[str, float] = {}
    for line in output.splitlines():
        m = _PYTEST_DURATION.match(line.strip())
        if m:
            out[m.group("id")] = float(m.group("duration"))
    return out


def parse_pytest(output: str) -> list[TestOutcome]:
    outcomes: list[TestOutcome] = []
    for line in output.splitlines():
        line = line.strip()
        m = _PYTEST_LINE.match(line) or _PYTEST_SUMMARY.match(line)
        if m:
            outcomes.append(TestOutcome(m.group("id"), _STATUS.get(m.group("status"), "unknown")))
            continue
        # pytest -rA reports skips as ``SKIPPED [1] test_x.py:2: reason`` with
        # no test id, so an all-skipped suite would otherwise look like an
        # unparsed run.
        skip = _PYTEST_SKIP_LINE.match(line)
        if skip:
            outcomes.append(TestOutcome(skip.group("id"), "skipped"))
    return outcomes


class Verifier:
    """Conventional verification gates (spec 5.1, 6.6-6.9).  Commands come from
    the detected :mod:`rbg.checks` toolchain, so a non-Python repository is
    checked with its own tools rather than being silently untouched."""

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox
        self._toolchain: Toolchain | None = None

    def toolchain(self) -> Toolchain:
        if self._toolchain is None:
            self._toolchain = detect(self.sandbox.root, python=self.sandbox.python)
        return self._toolchain

    def compile_check(self, paths: list[str]) -> CommandResult | None:
        """None when the project has no compile step (an unverified gate, not a
        green one)."""
        cmd = compile_command(self.toolchain(), self.sandbox.python, paths)
        return self.sandbox.run(cmd) if cmd else None

    def typecheck_available(self) -> bool:
        """True when the toolchain has a type checker, so a missing checker is
        reported as unverified rather than silently scored clean."""
        if not self.toolchain().is_python:
            return False
        return self.sandbox.run([self.sandbox.python, "-c", "import mypy"]).ok

    def typecheck(self, paths: list[str] | None = None) -> CommandResult:
        targets = list(paths) if paths else ["."]
        return self.sandbox.run(
            [self.sandbox.python, "-m", "mypy", *targets, "--ignore-missing-imports"])

    def run_tests(self, test_paths: list[str] | None = None, timeout_s: int = 120) -> tuple[CommandResult, list[TestOutcome]]:
        cmd = runner_argv(self.toolchain(), self.sandbox.python, test_paths)
        if not cmd:
            # No applicable test runner: a synthetic "nothing collected" result
            # (exit 5), so the tests gate is recorded unverified, never passed.
            return CommandResult("(no test command for this project)", self.sandbox.root,
                                 5, "", "", 0), []
        res = self.sandbox.run(cmd, timeout_s=timeout_s)
        text = res.stdout + "\n" + res.stderr
        outcomes = parse_pytest(text)
        durations = parse_pytest_durations(text)
        for outcome in outcomes:
            if outcome.test_id in durations:
                outcome.duration_s = durations[outcome.test_id]
        return res, outcomes


@dataclass
class Ledger:
    """Append-only trace store (spec 8.1, 7), backed by the repo's libSQL
    database.  libSQL is the only store; there is no JSONL fallback."""

    store: object
    entries: list[dict] = field(default_factory=list)

    def record(self, *, component: str, task_id: str, repo_id: str, revision: str,
               model_name: str, result: dict, reason_codes: list[str] | None = None,
               confidence: float | None = None, latency_ms: int = 0,
               inputs_hash: str = "") -> dict:
        entry = {
            "task_id": task_id, "repo_id": repo_id, "revision": revision,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": component, "model_name": model_name,
            "schema_version": SCHEMA_VERSION, "inputs_hash": inputs_hash,
            "result": result, "confidence": confidence,
            "reason_codes": reason_codes or [], "latency_ms": latency_ms,
        }
        self.entries.append(entry)
        self.store.append_ledger(entry)
        return entry
