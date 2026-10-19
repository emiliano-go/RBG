"""Runnable checks for language detection and command construction."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.checks import (  # noqa: E402
    Toolchain, binary_available, compile_command, detect, runner_argv,
)


def test_detect_python(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    assert detect(str(tmp_path)).language == "python"


def test_detect_python_by_source_file(tmp_path):
    (tmp_path / "mod.py").write_text("x = 1\n")
    assert detect(str(tmp_path)).is_python


def test_detect_other_languages(tmp_path):
    for marker, lang in (("package.json", "node"), ("Cargo.toml", "rust"),
                         ("go.mod", "go"), ("Makefile", "make")):
        d = tmp_path / lang
        d.mkdir()
        (d / marker).write_text("")
        assert detect(str(d)).language == lang


def test_unknown_language_has_no_commands(tmp_path):
    tc = detect(str(tmp_path))
    assert tc.language == "unknown"
    assert compile_command(tc, "python3", None) == []
    assert runner_argv(tc, "python3", None) == []


def test_python_commands_use_paths_and_selection():
    tc = Toolchain("python")
    assert compile_command(tc, "python3", ["a.py", "b.py"]) == \
        ["python3", "-m", "py_compile", "a.py", "b.py"]
    assert compile_command(tc, "python3", []) == ["python3", "-m", "compileall", "-q", "."]
    assert runner_argv(tc, "python3", ["t::x"]) == \
        ["python3", "-m", "pytest", "t::x", "-q", "-rA", "--no-header"]


def test_non_python_commands_ignore_paths():
    tc = Toolchain("go", compile=["go", "build", "./..."], test=["go", "test", "./..."])
    assert compile_command(tc, "python3", ["x.py"]) == ["go", "build", "./..."]
    assert runner_argv(tc, "python3", ["x"]) == ["go", "test", "./..."]


def test_binary_available():
    assert binary_available(["sh"]) is True
    assert binary_available(["definitely-not-a-binary-xyz"]) is False
    assert binary_available([]) is False
