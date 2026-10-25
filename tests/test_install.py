"""Client config wiring: merge, idempotence, uninstall, and safety."""

import json
import os
import sys
from pathlib import Path

import pytest

from rbg.install import InstallError, config_path, install, server_command


def _json(path):
    return json.loads(path.read_text())


def test_project_json_targets_merge_with_existing_config(tmp_path):
    (tmp_path / "opencode.json").write_text(json.dumps({"theme": "dark"}))
    install(["opencode"], scope="project", directory=tmp_path, roots=[tmp_path])

    data = _json(tmp_path / "opencode.json")
    assert data["theme"] == "dark"
    entry = data["mcp"]["rbg"]
    assert entry["type"] == "local"
    assert (entry["command"][0].endswith("rbg-mcp")
            or entry["command"][-2:] == ["-m", "rbg.mcp_server"])
    assert entry["environment"]["RBG_ALLOWED_ROOTS"] == str(tmp_path.resolve())


def test_claude_project_config(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    install(["claude"], scope="project", directory=tmp_path, roots=[a, b])
    entry = _json(tmp_path / ".mcp.json")["mcpServers"]["rbg"]
    argv = [entry["command"], *entry["args"]]
    assert argv[0].endswith("rbg-mcp") or argv[-2:] == ["-m", "rbg.mcp_server"]
    assert entry["env"]["RBG_ALLOWED_ROOTS"].split(os.pathsep) == [str(a), str(b)]


def test_codex_toml_is_idempotent_and_replaces_managed_block(tmp_path):
    install(["codex"], scope="project", directory=tmp_path, roots=[tmp_path])
    install(["codex"], scope="project", directory=tmp_path, roots=[tmp_path])

    text = (tmp_path / ".codex" / "config.toml").read_text()
    assert text.count("[mcp_servers.rbg]") == 1
    assert str(tmp_path) in text


def test_uninstall_removes_entry_and_empty_container(tmp_path):
    install(["cursor"], scope="project", directory=tmp_path, roots=[tmp_path])
    install(["cursor"], scope="project", directory=tmp_path, roots=[tmp_path],
            uninstall=True)
    assert "mcpServers" not in _json(tmp_path / ".cursor" / "mcp.json")


def test_dry_run_writes_nothing(tmp_path):
    records = install(["opencode"], scope="project", directory=tmp_path, dry_run=True)
    assert records[0]["status"] == "planned"
    assert not (tmp_path / "opencode.json").exists()


def test_invalid_json_is_not_clobbered(tmp_path):
    path = tmp_path / "opencode.json"
    path.write_text("{not json")
    with pytest.raises(InstallError):
        install(["opencode"], scope="project", directory=tmp_path)
    assert path.read_text() == "{not json"


def test_user_scope_honours_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    path = config_path("opencode", "user", tmp_path)
    assert path == tmp_path / "xdg" / "opencode" / "opencode.json"
    install(["opencode"], scope="user", directory=tmp_path, roots=[tmp_path])
    assert path.exists()


def test_user_scope_requires_an_explicit_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(InstallError, match="explicit --root"):
        install(["opencode"], scope="user", directory=tmp_path)


def test_project_scope_defaults_to_the_project_dir(tmp_path):
    records = install(["opencode"], scope="project", directory=tmp_path)
    entry = _json(tmp_path / "opencode.json")["mcp"]["rbg"]
    assert entry["environment"]["RBG_ALLOWED_ROOTS"] == str(tmp_path.resolve())
    assert records[0]["roots"] == [str(tmp_path.resolve())]


def test_missing_root_and_filesystem_root_are_rejected(tmp_path):
    with pytest.raises(InstallError, match="not a directory"):
        install(["opencode"], scope="project", directory=tmp_path,
                roots=[tmp_path / "nope"])
    with pytest.raises(InstallError, match="filesystem root"):
        install(["opencode"], scope="project", directory=tmp_path, roots=["/"])


def test_roots_are_deduped(tmp_path):
    records = install(["opencode"], scope="project", directory=tmp_path,
                      roots=[tmp_path, tmp_path])
    assert records[0]["roots"] == [str(tmp_path.resolve())]


def test_file_mode_is_preserved(tmp_path):
    path = tmp_path / ".mcp.json"
    path.write_text("{}")
    path.chmod(0o600)
    install(["claude"], scope="project", directory=tmp_path, roots=[tmp_path])
    assert path.stat().st_mode & 0o777 == 0o600


def test_symlinked_config_stays_a_symlink(tmp_path):
    real = tmp_path / "dotfiles" / "mcp.json"
    real.parent.mkdir()
    real.write_text("{}")
    link = tmp_path / ".mcp.json"
    link.symlink_to(real)
    install(["claude"], scope="project", directory=tmp_path, roots=[tmp_path])
    assert link.is_symlink()
    assert "mcpServers" in json.loads(real.read_text())


def test_second_install_reports_unchanged(tmp_path):
    install(["opencode"], scope="project", directory=tmp_path, roots=[tmp_path])
    records = install(["opencode"], scope="project", directory=tmp_path, roots=[tmp_path])
    assert records[0]["status"] == "unchanged"


def test_codex_dotted_key_conflict_is_not_written(tmp_path):
    path = tmp_path / ".codex" / "config.toml"
    path.parent.mkdir()
    original = '[mcp_servers]\nrbg.command = "something-else"\n'
    path.write_text(original)
    with pytest.raises(InstallError, match="conflicts"):
        install(["codex"], scope="project", directory=tmp_path, roots=[tmp_path])
    assert path.read_text() == original


def test_server_command_comes_from_the_running_environment():
    cmd = server_command()
    assert Path(cmd[0]).parent == Path(sys.executable).parent
    assert cmd[-2:] == ["-m", "rbg.mcp_server"] or cmd[0].endswith("rbg-mcp")
