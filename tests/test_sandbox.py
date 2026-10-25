"""Containment and sandbox-boundary checks (spec: security boundary).

Run: ``python3 -m pytest tests -q``
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbg import isolation  # noqa: E402
from rbg.isolation import IsolationUnavailable  # noqa: E402
from rbg.paths import resolve_within  # noqa: E402
from rbg.verify import Sandbox  # noqa: E402


def test_relative_path_resolves_under_root(tmp_path):
    assert resolve_within(str(tmp_path), "a/b.py") == str(tmp_path / "a" / "b.py")


def test_absolute_path_inside_root_ok(tmp_path):
    assert resolve_within(str(tmp_path), str(tmp_path / "x.py")) == str(tmp_path / "x.py")


def test_new_nonexistent_file_allowed(tmp_path):
    got = resolve_within(str(tmp_path), "new/dir/created.py")
    assert got == str(tmp_path / "new" / "dir" / "created.py")


def test_sibling_prefix_is_rejected(tmp_path):
    root = tmp_path / "repo"
    escape = tmp_path / "repo-escape"
    root.mkdir()
    escape.mkdir()
    with pytest.raises(PermissionError):
        resolve_within(str(root), str(escape / "secret.py"))


def test_parent_traversal_is_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    with pytest.raises(PermissionError):
        resolve_within(str(root), "../outside.py")


def test_symlink_escape_is_rejected(tmp_path):
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "target.py").write_text("x = 1\n")
    link = root / "link.py"
    try:
        os.symlink(outside / "target.py", link)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(PermissionError):
        resolve_within(str(root), "link.py")


def test_symlink_inside_root_is_allowed(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "real.py").write_text("x = 1\n")
    link = root / "alias.py"
    try:
        os.symlink(root / "real.py", link)
    except OSError:
        pytest.skip("symlinks unavailable")
    assert resolve_within(str(root), "alias.py") == str(root / "real.py")


# - environment allowlist -------------------------------------------------- #
def test_unnamed_secret_is_stripped(tmp_path, monkeypatch):
    sb = Sandbox(str(tmp_path))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "leak")
    assert "AWS_ACCESS_KEY_ID" not in sb._env()


def test_safe_vars_are_kept(tmp_path, monkeypatch):
    sb = Sandbox(str(tmp_path))
    monkeypatch.setenv("HOME", "/home/x")
    assert sb._env().get("HOME") == "/home/x"


def test_allowlist_extension_is_honoured(tmp_path, monkeypatch):
    sb = Sandbox(str(tmp_path))
    monkeypatch.setenv("RBG_ENV_ALLOW", "MY_VAR")
    monkeypatch.setenv("MY_VAR", "ok")
    assert sb._env().get("MY_VAR") == "ok"


def test_sandbox_rejects_shell_string(tmp_path):
    sb = Sandbox(str(tmp_path))
    with pytest.raises(TypeError):
        sb.run("echo hi")


# - isolation -------------------------------------------------------------- #
def test_isolation_is_applied_when_backend_exists(tmp_path):
    if isolation.available_backend() is None:
        pytest.skip("no isolation backend on this host")
    res = Sandbox(str(tmp_path)).run(["python3", "-c", "print('hi')"])
    assert res.isolated is True
    assert res.isolation in ("bwrap", "nsjail")
    assert res.network_disabled is True
    assert res.ok and "hi" in res.stdout


def test_run_refuses_without_backend_or_hatch(tmp_path, monkeypatch):
    monkeypatch.setattr(isolation, "available_backend", lambda: None)
    monkeypatch.delenv("RBG_ALLOW_UNISOLATED", raising=False)
    with pytest.raises(IsolationUnavailable):
        Sandbox(str(tmp_path)).run(["python3", "-c", "print('hi')"])


def test_unisolated_hatch_marks_result(tmp_path, monkeypatch):
    monkeypatch.setattr(isolation, "available_backend", lambda: None)
    monkeypatch.setenv("RBG_ALLOW_UNISOLATED", "1")
    res = Sandbox(str(tmp_path)).run(["python3", "-c", "print('hi')"])
    assert res.ok and res.isolated is False and res.isolation == "none"


# - fixed workspace mount -------------------------------------------------- #
def _bwrap_only():
    if isolation.available_backend() is None:
        pytest.skip("no isolation backend on this host")


def test_repo_is_mounted_at_the_fixed_workspace_path(tmp_path):
    _bwrap_only()
    res = Sandbox(str(tmp_path)).run(["sh", "-c", "pwd"])
    assert res.ok and res.stdout.strip() == isolation.WORKSPACE


def test_host_home_is_always_hidden(tmp_path):
    _bwrap_only()
    res = Sandbox(str(tmp_path)).run(
        ["sh", "-c", 'test -z "$(ls -A /home)" && test ! -e "$HOME/.ssh"'])
    assert res.ok, res.stderr


def test_symlink_to_outside_repo_is_not_readable(tmp_path):
    _bwrap_only()
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("leak")
    try:
        os.symlink(outside / "secret.txt", root / "link.txt")
    except OSError:
        pytest.skip("symlinks unavailable")
    res = Sandbox(str(root)).run(["cat", "link.txt"])
    assert not res.ok and "leak" not in res.stdout


def test_subdirectory_cwd_is_translated(tmp_path):
    _bwrap_only()
    (tmp_path / "sub").mkdir()
    res = Sandbox(str(tmp_path)).run(["pwd"], cwd="sub")
    assert res.ok and res.stdout.strip() == f"{isolation.WORKSPACE}/sub"


# - project virtualenvs under the fixed mount ------------------------------ #
def _make_venv(path):
    import subprocess
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(path)],
                   check=True, capture_output=True)


def _install_probe(venv, name):
    site = next((p for p in (venv / "lib").glob("python*/site-packages")), None)
    assert site is not None, f"no site-packages in {venv}"
    value = "inside" if name == "inside_probe" else "outside"
    (site / f"{name}.py").write_text(f"VALUE = {value!r}\n")


def test_project_venv_is_mounted_and_executes(tmp_path):
    _bwrap_only()
    root = tmp_path / "repo"
    root.mkdir()
    _make_venv(root / ".venv")
    _install_probe(root / ".venv", "inside_probe")
    sb = Sandbox(str(root))
    res = sb.run([sb.python_sandbox, "-c",
                  "import sys, inside_probe; print(sys.prefix, inside_probe.VALUE)"])
    assert res.ok, res.stderr
    assert res.stdout.strip() == "/tmp/venv inside"


def test_venv_outside_workspace_is_mounted(tmp_path):
    _bwrap_only()
    root = tmp_path / "repo"
    venv = tmp_path / "shared-venv"
    root.mkdir()
    _make_venv(venv)
    os.symlink(venv, root / ".venv")
    _install_probe(venv, "outside_probe")
    sb = Sandbox(str(root))
    res = sb.run([sb.python, "-c",
                  "import sys, outside_probe; print(sys.prefix, outside_probe.VALUE)"])
    assert res.ok, res.stderr
    assert res.stdout.strip() == "/tmp/venv outside"


def test_project_under_home_with_venv_executes(tmp_path):
    _bwrap_only()
    home = os.path.expanduser("~")
    if not os.access(home, os.W_OK):
        pytest.skip("home is not writable")
    root = tmp_path if str(tmp_path).startswith(home) else None
    if root is None:
        import tempfile
        root = __import__("pathlib").Path(tempfile.mkdtemp(prefix="rbg-home-", dir=home))
    (root / "repo").mkdir(exist_ok=True)
    repo = root / "repo"
    _make_venv(repo / ".venv")
    sb = Sandbox(str(repo))
    res = sb.run([sb.python, "-c", "import sys; print(sys.prefix)"])
    assert res.ok, res.stderr
    assert res.stdout.strip() == "/tmp/venv"


def test_runtime_under_a_masked_path_is_rebound(monkeypatch, tmp_path):
    mask = tmp_path / "opt"
    keep = mask / "python" / "x64"
    keep.mkdir(parents=True)
    monkeypatch.setattr(isolation, "_HOST_MASK", (str(mask),))
    argv = isolation._wrap("bwrap", False, str(tmp_path), (), (str(keep),))
    text = " ".join(argv)
    assert f"--tmpfs {mask}" in text
    assert f"--ro-bind {keep} {keep}" in text
    assert text.index(f"--tmpfs {mask}") < text.index(f"--ro-bind {keep} {keep}")


def test_host_data_directories_are_masked(tmp_path):
    _bwrap_only()
    res = Sandbox(str(tmp_path)).run(
        ["sh", "-c", "test -z \"$(ls -A /var 2>/dev/null)\" && "
                     "test -z \"$(ls -A /srv 2>/dev/null)\" && "
                     "test -z \"$(ls -A /run 2>/dev/null)\""])
    assert res.ok, res.stderr


def test_opt_exposes_only_interpreter_runtime_subtrees(tmp_path):
    _bwrap_only()
    sb = Sandbox(str(tmp_path))
    res = sb.run(["sh", "-c", "ls -A /opt 2>/dev/null || true"])
    assert res.ok, res.stderr
    allowed = [p for p in sb.runtime if p.startswith("/opt/")]
    for entry in [e for e in res.stdout.splitlines() if e.strip()]:
        top = f"/opt/{entry}"
        assert any(p == top or p.startswith(top + "/") for p in allowed), \
            f"unexpected /opt entry visible: {entry}"
