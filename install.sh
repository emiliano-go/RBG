#!/usr/bin/env bash
# Install RBG and register its MCP server with every detected client.
#
#   ./install.sh                       # auto-detect opencode, claude, cursor, ...
#   ./install.sh --target claude       # one client
#   ./install.sh --project             # project-local config in the cwd
#   RBG_SOURCE=git+ssh://... ./install.sh
set -euo pipefail

src="${RBG_SOURCE:-}"
if [ -z "$src" ]; then
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [ -f "$here/pyproject.toml" ]; then
    src="$here"
  else
    src="git+https://github.com/emiliano-go/RBG.git"
  fi
fi

if command -v pipx >/dev/null 2>&1; then
  pipx install --force "$src"
elif command -v uv >/dev/null 2>&1; then
  uv tool install --force "$src"
else
  python3 -m pip install --user --upgrade "$src"
fi

case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) export PATH="$HOME/.local/bin:$PATH" ;;
esac

# User scope needs an explicit root.  Default the one-liner to the directory it
# was run from, never HOME.
want_root=1
for arg in "$@"; do
  case "$arg" in
    --root|--root=*|--project|--uninstall) want_root=0 ;;
  esac
done
if [ "$want_root" -eq 1 ]; then
  set -- "$@" --root "$PWD"
fi

exec rbg install "$@"
