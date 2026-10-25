"""Detect a repository's language and the commands that compile and test it.

The harness must not run Python tooling against a non-Python tree and call the
silence "verified".  :func:`detect` inspects marker files and returns a
:class:`Toolchain`; an empty command list means *not applicable*, so the caller
records the gate as unverified instead of passing it.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Toolchain:
    language: str
    compile: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    test_globs: tuple[str, ...] = ()

    @property
    def is_python(self) -> bool:
        return self.language == "python"


def _present(root: str, *names: str) -> bool:
    return any(os.path.exists(os.path.join(root, n)) for n in names)


def _top_level_has(root: str, suffix: str) -> bool:
    try:
        return any(f.endswith(suffix) for f in os.listdir(root))
    except OSError:
        return False


def detect(root: str, python: str = "python3") -> Toolchain:
    """First matching language by marker file, else ``unknown``."""
    if _present(root, "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt") \
            or _top_level_has(root, ".py"):
        return Toolchain("python", test_globs=("test_*.py", "*_test.py"))
    if _present(root, "package.json"):
        return Toolchain("node",
                         compile=["npx", "--no-install", "tsc", "--noEmit"],
                         test=["npm", "test", "--silent"],
                         test_globs=("*.test.js", "*.spec.js", "*.test.ts"))
    if _present(root, "Cargo.toml"):
        return Toolchain("rust", compile=["cargo", "check", "--quiet"],
                         test=["cargo", "test", "--quiet"])
    if _present(root, "go.mod"):
        return Toolchain("go", compile=["go", "build", "./..."],
                         test=["go", "test", "./..."])
    if _present(root, "Makefile", "makefile"):
        return Toolchain("make", test=["make", "test"])
    return Toolchain("unknown")


def compile_command(toolchain: Toolchain, python: str, paths: list[str] | None) -> list[str]:
    if toolchain.is_python:
        if paths:
            return [python, "-m", "py_compile", *paths]
        return [python, "-m", "compileall", "-q", "."]
    return list(toolchain.compile)


def runner_argv(toolchain: Toolchain, python: str, selected: list[str] | None) -> list[str]:
    if toolchain.is_python:
        return [python, "-m", "pytest", *(selected or []), "-q", "-rA", "--no-header", "--durations=0"]
    return list(toolchain.test)


def binary_available(cmd: list[str]) -> bool:
    return bool(cmd) and shutil.which(cmd[0]) is not None
