# MCP tools

Served by `rbg/mcp_server.py` (stdio); the tools live in `rbg/mcp_tools.py`, plan tools in `rbg/mcp_plan.py`, and the service layer in `rbg/mcp_service.py`. `root` is always the absolute path
to a repository; the server indexes it lazily and caches the index per revision.
The persisted index is shared across processes, so parallel agents on one
project parse each revision once.

| Tool | Arguments | Returns |
|---|---|---|
| `search_code` | `root`, `query`, `limit=10`, `include`, `exclude` | Ranked code hits (path, symbol, lines, snippet) from the hybrid retriever, with `include`/`exclude` glob filters applied before fusion. |
| `retrieve_context` | `root`, `query`, `limit=8`, `expand=true`, `include`, `exclude` | A bounded, source-grounded context bundle with provenance (path, lines, content hash, commit). `expand` pulls neighbouring/caller context. |
| `read_chunk` | `root`, `chunk_id` | The full text of one indexed chunk by id, for when a capped discovery snippet is not enough. |
| `find_symbol` | `root`, `name`, `exact=false`, `path`, `kind` | Definition location(s) plus every referencing file. `path` filters definition paths by glob; `kind` filters `function`/`class`/`constant`. |
| `get_change_impact` | `root`, `symbols` | Predicted **and** graph-derived impact scope: referenced files, likely tests, fan-out. Advisory. |
| `rank_tests` | `root`, `changed_symbols`, `limit=20` | Candidate tests ranked by regression-detection value, each with evidence (`covers_changed_symbol`, …), predicted relevance/flakiness and duration. Mandatory suites are flagged, never skipped. Advisory. |
| `run_checks` | `root`, `profile=compile`, `timeout_s=120` | Runs an approved check profile in the sandbox: `compile`, `collect_tests`, or `tests`. Returns pass/fail and captured output. |
| `get_diagnostics` | `root`, `task_id` | The recorded compiler/test evidence (diagnostics) for a task handle. |
| `get_task_status` | `root`, `task_id` | Status, verified gates, and recorded checks for a task handle. |
| `propose_patch` | `root`, `path`, `content` | Validates a patch proposal (syntax/shape) and returns the verdict. **Does not apply it.** |
| `train_models` | `root`, `validate=false` | Trains risk, selector, reranker and impact models from the ledger; returns metrics and calibration state. With `validate=true`, gates the risk head on a held-out Brier/coverage comparison and records the verdict. |
| `plan_audit` | `root`, `plan_text`, `task=""`, `changed_files`, `task_id` | With `task_id`, runs the stored-plan **drift audit** (declared files, allowed paths, acceptance, unplanned writes and deletions) and returns the plan, rollup, and `drift` verdict. Otherwise parses free text into steps and computes **scope drift** and a **completeness audit**. The deterministic verdict is a gate; the learned head is advisory. |
| `plan_create` | `root`, `task_id`, `title`, `sections` | Creates the server-owned JSON plan for `(root, task_id)`; versions start at 1. |
| `plan_append` / `plan_attach` | `root`, `task_id`, `section`, `parent_id` | Appends a top-level section or nests one under an existing section (depth cap 5). |
| `plan_complete` | `root`, `task_id`, `section_id`, `status`, `evidence`, `validation` | Marks a section complete only after the automatic drift audit passes. A caller-supplied validation cannot override a deterministic blocker. Completion rolls up to parents. |
| `plan_modify` / `plan_note` / `plan_check` | `root`, `task_id`, `section_id`, … | Edit fields, add a note, or record a named check on a section. |
| `plan_get` / `plan_tree` | `root`, `task_id`, `section_id` | Read one section or the whole rollup tree. |
| `get_calibration` | `root` | Conformal calibration state and training metrics (coverage, ECE, thresholds). Advisory. |
| `check_drift` | `root` | Time-ordered drift / OOD check over the execution ledger: PSI per feature, calibration drift, OOD rate. Advisory. |
| `db_stats` | `root` | libSQL store statistics: persisted chunks, vectors, ledger rows and model artefacts. |
| `vector_search` | `root`, `query`, `limit=10` | On-disk ANN search through the libSQL vector index (cosine), joined back to chunk metadata. |

All tools return MCP `structuredContent`; every response carries the repo
`revision` so a result is attributable to a tree state.

---
