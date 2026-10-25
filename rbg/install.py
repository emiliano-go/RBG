"""Register the RBG MCP server with agent harnesses.

``rbg install`` writes the smallest config entry each client needs, merging
into existing JSON/TOML instead of overwriting it.  Supported targets:

    opencode, claude, cursor, windsurf, gemini, codex, vscode

User scope writes the client's global config; project scope writes a config
inside the working tree.  The MCP server denies every repository by default,
so the entry always carries ``RBG_ALLOWED_ROOTS`` naming the roots the
operator chose.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import tempfile
import tomllib
from pathlib import Path

TARGETS = ("opencode", "claude", "cursor", "windsurf", "gemini", "codex", "vscode")

_BINARIES = {
    "opencode": ("opencode",),
    "claude": ("claude",),
    "cursor": ("cursor", "cursor-agent"),
    "windsurf": ("windsurf",),
    "gemini": ("gemini",),
    "codex": ("codex",),
    "vscode": ("code",),
}

_TOML_BEGIN = "# >>> rbg >>>"
_TOML_END = "# <<< rbg <<<"


class InstallError(RuntimeError):
    """A config file exists but cannot be safely edited."""


def _home() -> Path:
    return Path(os.path.expanduser("~"))


def _config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or (_home() / ".config"))


def config_path(target: str, scope: str, directory: str | Path = ".") -> Path | None:
    """Where ``target`` keeps its MCP config, or ``None`` if unsupported."""
    d = Path(directory).resolve()
    home = _home()
    if target == "opencode":
        return (_config_home() / "opencode" / "opencode.json"
                if scope == "user" else d / "opencode.json")
    if target == "claude":
        return home / ".claude.json" if scope == "user" else d / ".mcp.json"
    if target == "cursor":
        return (home / ".cursor" / "mcp.json" if scope == "user"
                else d / ".cursor" / "mcp.json")
    if target == "windsurf":
        return home / ".codeium" / "windsurf" / "mcp_config.json" if scope == "user" else None
    if target == "gemini":
        return (home / ".gemini" / "settings.json" if scope == "user"
                else d / ".gemini" / "settings.json")
    if target == "codex":
        return (home / ".codex" / "config.toml" if scope == "user"
                else d / ".codex" / "config.toml")
    if target == "vscode":
        return None if scope == "user" else d / ".vscode" / "mcp.json"
    raise InstallError(f"unknown target: {target}")


def detect_targets(directory: str | Path = ".", scope: str = "user") -> list[str]:
    """Targets that look installed: a client binary, its home dir, or a config."""
    found = []
    home = _home()
    markers = {
        "opencode": (_config_home() / "opencode",),
        "claude": (home / ".claude.json", home / ".claude"),
        "cursor": (home / ".cursor",),
        "windsurf": (home / ".codeium" / "windsurf",),
        "gemini": (home / ".gemini",),
        "codex": (home / ".codex",),
        "vscode": (),
    }
    for target in TARGETS:
        path = config_path(target, scope, directory)
        if path is None:
            continue
        if (any(shutil.which(b) for b in _BINARIES[target])
                or any(m.exists() for m in markers[target])
                or (path is not None and path.exists())):
            found.append(target)
    return found


def server_command() -> list[str]:
    """Absolute argv for the stdio server.

    Only a console script installed next to the running interpreter is trusted;
    an arbitrary earlier ``rbg-mcp`` on ``PATH`` may be a different install.
    """
    for name in ("rbg-mcp", "rbg-mcp.exe"):
        sibling = Path(sys.executable).with_name(name)
        if sibling.is_file() and os.access(sibling, os.X_OK):
            return [str(sibling)]
    return [sys.executable, "-m", "rbg.mcp_server"]


def _entry(target: str, roots: list[str]) -> dict:
    cmd = server_command()
    env = {"RBG_ALLOWED_ROOTS": os.pathsep.join(roots)} if roots else {}
    if target == "opencode":
        return {"type": "local", "command": cmd, "enabled": True, "environment": env}
    if target == "vscode":
        return {"type": "stdio", "command": cmd[0], "args": cmd[1:], "env": env}
    return {"command": cmd[0], "args": cmd[1:], "env": env}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstallError(f"{path}: cannot read JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise InstallError(f"{path}: top-level JSON must be an object")
    return data


def _write_text(path: Path, text: str) -> None:
    """Atomically replace ``path``, following symlinks and keeping its mode."""
    target = Path(os.path.realpath(path))
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else None
        fd, tmp_name = tempfile.mkstemp(
            prefix=target.name + ".", suffix=".rbg-tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            if mode is not None:
                os.chmod(tmp_name, mode)
            os.replace(tmp_name, target)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    except OSError as exc:
        raise InstallError(f"{path}: cannot write ({exc})") from exc


def _apply_json(path: Path, key: str, entry: dict | None) -> bool:
    data = _read_json(path)
    before = json.dumps(data, sort_keys=True)
    container = data.setdefault(key, {})
    if not isinstance(container, dict):
        raise InstallError(f"{path}: {key!r} must be an object")
    if entry is None:
        container.pop("rbg", None)
        if not container:
            data.pop(key, None)
    else:
        container["rbg"] = entry
    if json.dumps(data, sort_keys=True) == before:
        return False
    text = json.dumps(data, indent=2) + "\n"
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise InstallError(f"{path}: refusing to write invalid JSON ({exc})") from exc
    _write_text(path, text)
    return True


def _drop_toml_table(lines: list[str], header: str) -> list[str]:
    out: list[str] = []
    dropping = False
    for line in lines:
        stripped = line.strip()
        if stripped == f"[{header}]":
            dropping = True
            continue
        if dropping and stripped.startswith("["):
            dropping = False
        if not dropping:
            out.append(line)
    return out


def _apply_toml(path: Path, entry: dict | None) -> bool:
    try:
        original = path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError as exc:
        raise InstallError(f"{path}: cannot read ({exc})") from exc
    lines = original.splitlines()
    kept: list[str] = []
    skip = False
    for line in lines:
        if line.strip() == _TOML_BEGIN:
            skip = True
            continue
        if line.strip() == _TOML_END:
            skip = False
            continue
        if not skip:
            kept.append(line)
    if entry is not None:
        kept = _drop_toml_table(kept, "mcp_servers.rbg")
    text = "\n".join(kept).rstrip()
    if entry is not None:
        block = [_TOML_BEGIN, "[mcp_servers.rbg]",
                 f"command = {json.dumps(entry['command'])}"]
        block.append(f"args = {json.dumps(entry['args'])}" if entry["args"] else "args = []")
        if entry["env"]:
            inner = ", ".join(f"{k} = {json.dumps(v)}" for k, v in entry["env"].items())
            block.append(f"env = {{ {inner} }}")
        block.append(_TOML_END)
        text = (text + "\n\n" if text else "") + "\n".join(block)
    text = text + "\n" if text else ""
    if text == original:
        return False
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise InstallError(
            f"{path}: existing config conflicts with the rbg entry ({exc}); "
            "remove the unmanaged mcp_servers.rbg table and retry") from exc
    _write_text(path, text)
    return True


def _resolve_roots(roots: list[str] | None, scope: str,
                   directory: Path) -> list[str]:
    if roots is None:
        if scope == "user":
            raise InstallError(
                "user-scope install needs an explicit --root (the MCP server "
                "denies every other path); pass e.g. --root ~/code, or use "
                "--project to grant only this repository")
        roots = [str(directory)]
    resolved: list[str] = []
    for raw in roots:
        value = str(raw).strip()
        if not value:
            continue
        path = os.path.realpath(os.path.expanduser(value))
        if path == os.sep:
            raise InstallError("refusing to grant the filesystem root; pass a repository directory")
        if not os.path.isdir(path):
            raise InstallError(f"root is not a directory: {raw}")
        if path not in resolved:
            resolved.append(path)
    if not resolved:
        raise InstallError("at least one repository root is required")
    return resolved


def broad_roots(roots: list[str]) -> list[str]:
    """Roots that grant more than the operator may expect (HOME or an ancestor)."""
    home = os.path.realpath(os.path.expanduser("~"))
    return [r for r in roots if home == r or home.startswith(r.rstrip(os.sep) + os.sep)]


def install(
    targets: list[str] | None = None,
    *,
    scope: str = "user",
    directory: str | Path = ".",
    roots: list[str] | None = None,
    dry_run: bool = False,
    uninstall: bool = False,
) -> list[dict]:
    """Write (or remove) the RBG server entry for every target.

    Returns one record per target: ``status`` is ``created``/``updated``/
    ``removed``/``unchanged``, ``planned`` when ``dry_run`` is set, or
    ``unsupported``.  Missing targets are auto-detected.
    """
    if scope not in ("user", "project"):
        raise InstallError("scope must be 'user' or 'project'")
    directory = Path(directory).resolve()
    if not dry_run:
        directory.mkdir(parents=True, exist_ok=True)

    if targets is None:
        targets = detect_targets(directory, scope)
        if not targets:
            raise InstallError(
                "no supported harness detected; pass --target "
                f"({'|'.join(TARGETS)})")
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        raise InstallError(f"unknown target(s): {', '.join(unknown)}")

    roots = _resolve_roots(roots, scope, directory)

    records = []
    for target in targets:
        path = config_path(target, scope, directory)
        if path is None:
            records.append({"target": target, "status": "unsupported",
                            "reason": f"{target} has no {scope}-scope MCP config"})
            continue
        entry = None if uninstall else _entry(target, roots)
        exists = path.exists()
        changed = True
        if not dry_run:
            if target == "codex":
                changed = _apply_toml(path, entry)
            else:
                key = {"opencode": "mcp", "vscode": "servers"}.get(target, "mcpServers")
                changed = _apply_json(path, key, entry)
        if dry_run:
            status = "planned"
        elif not changed:
            status = "unchanged"
        elif uninstall:
            status = "removed"
        elif exists:
            status = "updated"
        else:
            status = "created"
        records.append({"target": target, "status": status, "path": str(path),
                        "entry": entry, "roots": roots})
    return records
