# Configuration

One command installs the package and registers its MCP server with every agent
client it finds:

```bash
./install.sh --root ~/code     # or: pipx install . && rbg install --root ~/code
```

`rbg install` merges an `rbg` entry into each client's existing config (nothing
else in the file is touched) and is idempotent. Re-run it after moving the
install, or use `--uninstall` to remove the entry. `install.sh` defaults
`--root` to the directory it was run from, never `$HOME`.

```bash
rbg install --root ~/code          # auto-detect clients; root is required
rbg install --target claude        # one client (repeatable)
rbg install --project              # project-local config; grants the cwd
rbg install --dir ~/code/proj      # project config elsewhere
rbg install --dry-run              # show files, write nothing
rbg install --uninstall            # remove the entry
```

| Target | User config | Project config |
|---|---|---|
| `opencode` | `~/.config/opencode/opencode.json` | `./opencode.json` |
| `claude` | `~/.claude.json` | `./.mcp.json` |
| `cursor` | `~/.cursor/mcp.json` | `./.cursor/mcp.json` |
| `windsurf` | `~/.codeium/windsurf/mcp_config.json` | - |
| `gemini` | `~/.gemini/settings.json` | `./.gemini/settings.json` |
| `codex` | `~/.codex/config.toml` | `./.codex/config.toml` |
| `vscode` | - | `./.vscode/mcp.json` |

The server **denies every repository by default**. The generated entry sets
`RBG_ALLOWED_ROOTS` to exactly the directories passed with `--root` (or the
project directory for `--project`). There is deliberately no `$HOME` default:
every path under an allowed root is readable through the retrieval tools, so
grant repository directories, not your home. Passing `/` is refused; granting
`$HOME` or a parent prints a warning. Restart the client after installing so it
reloads the MCP server list.

## Environment

| Variable | Purpose |
|---|---|
| `RBG_ALLOWED_ROOTS` | Colon-separated roots the MCP server may serve. Empty denies everything. |
| `RBG_DB_DIR`, `RBG_DB_URL`, `RBG_DB_IN_REPO` | Relocate the libSQL store, or use `<repo>/.rbg/rbg.db`. |
| `RBG_DB_BUSY_TIMEOUT_MS` | Write-lock wait before failing (default `30000`). |
| `RBG_DB_SYNCHRONOUS` | `FULL` (default) or `NORMAL` durability. |
| `RBG_ALLOW_UNISOLATED` | Development-only escape hatch; results can never be `verified`. |
| `RBG_ENV_ALLOW` | Extra environment variables passed to sandboxed commands. |
| `RBG_ALLOW_SENSITIVE` | Index credential-shaped files that are skipped by default. |
| `RBG_TOOL_TIMEOUT_MS` | Default MCP tool wall-clock budget (default `60000`). |
| `RBG_TOOL_TIMEOUT_MAX_MS` | Non-negotiable per-call cap (default `300000`). |
| `RBG_READ_BUDGET_MS` | Soft retrieval budget before a result is marked degraded. |
| `RBG_FAULT_LOCALIZATION` | Set `1` to run Ochiai line-coverage localization and feed suspects into a bounded repair attempt (opt-in: reruns the suite). |
| `RBG_STATIC_MODEL`, `RBG_EMBEDDING_MODEL`, `RBG_RERANKER` | Optional encoders/reranker. |
| `RBG_LLM_BASE_URL`, `RBG_LLM_API_KEY`, `RBG_LLM_MODEL`, `RBG_LLM_TIMEOUT` | Live-model provider for `rbg replay`/benchmarks. |

## Model artifacts

Every model is optional and selected by environment; the deterministic path is
unchanged when none is configured.

| Preset | Env | Notes |
|---|---|---|
| `potion-code` | `RBG_STATIC_MODEL=<path>` | 16M static code encoder, CPU, no torch. |
| `code-rank` | `RBG_EMBEDDING_MODEL=<path>` | CodeRankEmbed 137M, needs `[semantic]`. |
| `gte-reranker` | `RBG_RERANKER=gte-reranker` | ModernBERT code reranker, needs `[semantic]`. |

Checkpoints are pinned to immutable upstream commits. `rbg models verify`
rejects a manifest whose repo/revision does not match the pinned spec, a
modified or missing file, and any unexpected extra file.

Pull and verify artifacts explicitly:

```bash
rbg models list
rbg models pull potion-code
rbg models verify potion-code
```

Semantic/torch presets still install with `pip install 'rbg[semantic]'`.
Hosted embeddings (Voyage-code-4, Cohere, Gemini) are deliberately not wired
in: they would send repository source to a provider.
