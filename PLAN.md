# RBG-Harness Hardening & Evaluation — Full Implementation Plan

Status: **approved**. Reviewed ground-up via the installed harness MCP
(`retrieve_context`, `search_code`, `find_symbol`, `plan_audit`).

## Locked decisions

| # | Decision | Consequence |
|---|---|---|
| D-Iso | Isolation required; `bwrap` → `nsjail` → **deny** | Unisolated runs never yield `verified` |
| D-Root | `HARNESS_ALLOWED_ROOTS` allowlist; empty ⇒ every tool errors | Server config must list roots |
| D-Scope | A–F in one programme | Real provider + benchmark in scope |
| D-Disc | Discovery tools return capped snippets **and** a `read_chunk` tool | Agents stop falling back to raw file reads |
| D-Learned | Learned models shadow-only until the benchmark proves gain | Deterministic policy authoritative |

## Progress

| Task | Status | Verified |
|---|---|---|
| P0-1 containment helper + 3 prefix-check replacements | **done** | `tests/test_sandbox.py` (7) + full suite 65 pass |
| P0-2 `Sandbox.run(argv, shell=False)` + all call sites | **done** | full suite 65 pass |
| P0-3 environment allowlist (replaces secret denylist) | **done** | 4 new tests; full suite 69 pass |
| P0-4 `harness/isolation.py` (bwrap/nsjail, probe-gated) wired into `Sandbox.run`; results carry `isolation`/`isolated`/`network_disabled` | **done** | 3 new tests; full suite 72 pass |
| P0-5 `harness/workspace.py` disposable git worktree for the edit/verify loop (fallback in place) | **done** | worktree test; full suite 73 pass |
| P0-6 MCP root authorization via `HARNESS_ALLOWED_ROOTS`, deny by default | **done** | 4 tests; full suite 77 pass |
| P0-7 `harness/status.py` evidence-derived `TerminalState`/`derive_status` | **done** | 8 tests; full suite 85 pass |
| P0-8 `harness/checks.py` language-aware compile/test commands | **done** | 7 tests; full suite 92 pass |
| P0-9 discovery tools return capped snippets + `read_chunk` tool | **done** | new test; full suite 93 pass |
| P1-1 union diff across repair iterations | **done** | new test; full suite 94 pass |
| P1-2 content-based revision + cache invalidation (C2/C3, folds in G3) | **done** | 2 tests; full suite 96 pass |
| P1-3 unique task id per run by default (C4) | **done** | new test; full suite 97 pass |
| P1-4 pyproject + console scripts + CI (installs bubblewrap) | **done** | editable install + `rbg models` verified |
| P1-5 OpenAI-compatible `MainLLM` provider (stdlib urllib) | **done** | 2 tests; full suite 99 pass |
| P1-6 `docs/threat-model.md` + MIT `LICENSE` | **done** | — |
| P2-1 temporal train/cal/test split + repo holdout (D1) | **done** | 2 tests; full suite 101 pass |
| P2-2 conformal on held-out slice; version v2 | **done** | folds into P2-1/push |
| P2-3 n + Wilson CI on metrics (D3) | **done** | test; full suite 102 pass |
| P2-4 selector/impact label = newly-failing regression (D2/D4) | **done** | test; full suite 103 pass |
| P2-5 learned risk shadow-gated on `validated` (D5) | **done** | 2 tests; full suite 104 pass |
| P2-6 deterministic benchmark + `rbg eval`/`compare` + smoke suite | **done** | benchmark test; full suite 105 pass; CI runs smoke |
| P2-7 curated 50-task real-bugfix suite (`eval/suites/curated`, `build.py --check`) | **done** | 50/50 resolved, 0 regressions, coverage 1.0 |
| P2-8 SWE-bench Verified adapter | deferred | needs infra + model key |

## Goals / non-goals

**Goals.** Trustworthy execution boundary; verification status that matches
evidence; correct repair-loop and cache state; statistically honest models; a
reproducible benchmark that decides which components earn their place; an
installable, auditable package.

**Non-goals (this pass).** New learned models; multi-agent; incremental indexing;
language-adapter interface; benchmark-number optimisation. Deferred to Phase 3,
gated on Phase 2 evidence.

---

## Cross-cutting building blocks (build first, reuse everywhere)

| Block | New file | Surface |
|---|---|---|
| Containment | `harness/paths.py` | `resolve_within(root, path) -> str` (realpath + `commonpath`, symlink-escape reject); `ensure_within` for new paths |
| Execution | `harness/isolation.py` | `detect_backend()`, `wrap(argv, root, network) -> argv`, `IsolationResult(backend, isolated, network_disabled)` |
| Checks | `harness/checks.py` | `checks_for(paths) -> list[Check]`; per-ext validators (`.py→py_compile`, `.json→json.load`, `.toml→tomllib`, `.yaml→optional`, `.md/.txt→skip`) |
| Status | `harness/status.py` | `TerminalState` + `derive_status(checks, tests, isolation, policy)` |
| Fingerprint | `harness/index.py` | `tree_fingerprint(root)` content hash; git tree hash when repo |

`TerminalState = verified | passed_with_unverified_checks | no_tests_collected |
failed | needs_human_review`.

---

## Verified findings (evidence)

| # | Finding | Evidence |
|---|---|---|
| A1 | Prefix containment bug (sibling `/repo-escape` passes) | `verify.py:86`, `orchestrator.py:115`, `mcp_server.py:336` |
| A2 | `shell=True` with string-assembled commands | `verify.py:91-92`; sites `verify.py:139-157`, `orchestrator.py:173`, `mcp_server.py:288-296` |
| A3 | Network "off" is only env markers | `verify.py:79-81` |
| A4 | Secret handling is a denylist | `verify.py:71-77` |
| A5 | MCP `root` only checked `isdir`; no allowlist | `mcp_server.py:48-51` |
| B1 | `all()` over empty checks → `True` | `orchestrator.py:353-354` |
| B2 | Unverified typecheck still counted in success | `orchestrator.py:294-299,358-361` |
| B3 | pytest exit 5 accepted as success | `orchestrator.py:356,362` |
| B4 | `py_compile` run on every changed path | `orchestrator.py:256` → `verify.py:139-141` |
| C1 | `diffs`/`files_changed` overwritten each iteration | `orchestrator.py:242,250-252` |
| C2 | Revision = `size:int(mtime)` | `index.py:227` |
| C3 | Index cache never invalidated | `mcp_server.py:54-62,510-516` |
| C4 | Default `task_id="task-1"` collapses training groups | `orchestrator.py:208` |
| D1 | Calibrator fit on the rows it scores (in-sample) | `models.py:488-500`; `training.py:~264` |
| D2 | Selector label = historical fail, not regression detection | `training.py:123,160` |
| D3 | Metrics reported without n/CI | `training.py` metrics dict |
| F1 | `search_code`/`find_symbol` return no source text | `retrieval.py:113-120`; `mcp_server.py` |

---

## Phase 0 (P0) — Security boundary, honest status, readable tools

### A. Sandbox & path containment
- **A1** Replace prefix checks in `verify.py:86`, `orchestrator.py:115`,
  `mcp_server.py:336` with `paths.resolve_within`.
- **A2** `Sandbox.run(argv: list[str], …, shell=False)`; rewrite call sites
  `verify.py:139-157`, `orchestrator.py:173`, `mcp_server.py` `_PROFILES`/`run_checks`.
  Delete the string path.
- **A3** Env allowlist (`PATH, HOME, LANG, LC_*, TMPDIR, VIRTUAL_ENV, PYTHON*`,
  harness vars) replaces `SECRET_HINTS` (`verify.py:71-77`).
- **A4** Route `Sandbox.run` through `isolation.wrap`: read-only bind of `root`,
  writable tmpfs scratch, `unshare -n` when `network=False`; record
  `isolated`/`network_disabled` on `CommandResult` and the ledger. No backend ⇒
  raise `IsolationUnavailable` (deny); `HARNESS_ALLOW_UNISOLATED=1` runs but
  forces non-`verified`.
- **A5** Apply edits in a disposable `git worktree` (fallback temp copy); report
  carries `worktree`; live tree untouched without explicit approval.
- **A6** `HARNESS_ALLOWED_ROOTS` (path-list) enforced in `_index`/every tool;
  deny-by-default.

### B. Verification semantics
- **B1** `Report.verification: TerminalState`, separate from `outcome`.
- **B2** Kill empty-`all()` truth (`orchestrator.py:353-354`); a check that did
  not run ⇒ `passed_with_unverified_checks`.
- **B3** Exit 5 ⇒ `no_tests_collected` (`orchestrator.py:356,362`).
- **B4** `checks.checks_for(paths)` replaces `compile_check(changed)`
  (`orchestrator.py:256`); unknown types skipped explicitly.
- **B5** `derive_status` combines checks + tests + `isolation.isolated` + policy;
  isolation required and absent ⇒ `needs_human_review`.

### F. MCP discovery returns source
- **F1** `search_code` merges capped `content` per hit (`MAX_OUTPUT_BYTES`);
  `find_symbol` resolves definition `content`.
- **F2** New `read_chunk(root, path, offset, limit)` — bounded, containment-checked.

**Phase 0 DoD.** Escape/symlink tests fail hard; no shell interpolation;
unisolated runs cannot report `verified`; zero-test and missing-checker runs
report their true state; discovery tools return readable code under caps.

---

## Phase 1 (P1) — Repair-loop & cache correctness, packaging, provider

### C. State correctness
- **C1** Snapshot tree pre-loop; `files_changed` = union of touched; `diffs` =
  original→final (`orchestrator.py:242-254`); downstream impact/completeness use
  the union.
- **C2** `tree_fingerprint` = content hash (`index.py:211-230`), symlink-aware;
  git tree hash when available.
- **C3** `_index` compares fingerprint to cached revision and evicts
  `_INDEX_CACHE`/`_RETRIEVER_CACHE`/`_LEARNED_CACHE` together on change
  (`mcp_server.py:54-62,510-516`); `_index_fresh` becomes a mutation trigger.
- **C4** `run(task, task_id=None)` generates unique ids (`orchestrator.py:208`).

### E. Packaging / CI / provider
- **E1** `pyproject.toml`: deps `numpy`, `libsql` (pinned); extras `mcp`, `dev`;
  console script `rbg`.
- **E2** `.github/workflows/ci.yml`: pytest + `eval/effectiveness.py` +
  `rbg eval --suite smoke`.
- **E3** `harness/providers.py`: `OpenAICompatibleLLM(BaseURL, model, key-env)`
  implementing `MainLLM`; `ScriptedLLM` stays demo-only.
- **E4** `docs/threat-model.md` (assets, trust boundaries, isolation, network,
  roots, residual risk) + `LICENSE`.

**Phase 1 DoD.** Two-iteration subset edit yields full change set; same-size/
same-mtime content edit changes revision and refreshes MCP; default task ids
unique; `pip install -e .[dev,mcp]` gives working `rbg`; CI green; a real model
completes the demo task end-to-end.

---

## Phase 2 (P1/P2) — Statistical validity + the benchmark that judges it

### D. Learned-model validity
- **D1** `training.py`: temporal train/calibration/test split + repo-level
  holdout; no fit-and-score overlap.
- **D2** `ConformalCalibrator.fit` (`models.py:488-500`) quantiles the
  **nonconformity score on the held-out calibration slice only**; `predict_set`
  standard rule; drop in-sample `qhat`. Bump version `conformal-platt-v1` → `v2`.
- **D3** Every metric carries `n` + CI (Wilson/Clopper-Pearson coverage; ECE with n).
- **D4** Relabel `selector_examples`/`impact_examples` (`training.py:108-162`)
  to change→test regression evidence; until that signal exists, models `shadow`.
- **D5** `use_learned` gated on a benchmark-validated flag; deterministic policy
  authoritative.

### E3' Benchmark & ablations
- `eval/benchmark.py`: `rbg eval --suite <dir> --config <name> --out j.json`;
  `rbg compare a.json b.json`.
- Configs: `baseline`, `+retrieval`, `+plan_audit`, `+test_ranking`, `full`.
- Metrics: task-resolution rate, regression rate, verification coverage,
  repair-loop success, wall time, tokens, cost; Recall@k/MRR; test-selection
  regression-detection recall; held-out ECE/coverage with CI.
- Suites: `eval/suites/smoke/` (~5, CI) → curated 50–100 real bugfix → SWE-bench
  Verified adapter.
- Reproducibility: pinned model config, prompts, tool settings, results per run.

**Phase 2 DoD.** Held-out coverage ≈ target (in-sample construction no longer
passes); benchmark reproducible; ablation table published; any `+component` that
does not move an outcome stays `shadow` and is documented as such.

---

## Phase 3 (deferred, gated on Phase 2)

Incremental indexing; language-adapter interface; typed action I/O with
preconditions/limits; release polish. Not started until the benchmark says which
components earn further investment.

---

## Workstream G — Retrieval latency (perf)

**Observed.** `retrieve_context` against a large snapshot
(`…/TrustSight/trustsight`) with a long multi-entity query is too slow:
`"G003 coordinated adoption novelty dormancy weighting, G008 first-seen
adoption burst, adoption_events, cycle_events last_modified"`. Separately,
`plan_audit` on the same repo **times out** at the MCP client after several
seconds — it is a text-only tool that should be milliseconds.

**Root causes (from code).**

| # | Cause | Evidence |
|---|---|---|
| G-a | Every discovery call runs `_learned()` → `train_from_ledger`: GBDT + `Reranker.lstsq` + impact + drift, incl. an O(n²·d) pairwise distance matrix, on the read path | `mcp_server.py` `_learned`; `training.py:260` |
| G-b | `_index_fresh` walks the whole tree with `os.stat` per file every call; planned content hashing (C2) would worsen it | `index.py:211-230`, `mcp_server.py:510-516` |
| G-c | `BM25.rank` is a pure-Python flat scan: O(query_terms × N_chunks); the existing libSQL FTS5 table is unused by retrieval | `retrieval.py:150-161`; `db.py:98` (`chunks_fts`) |
| G-d | `expand` is O(hits × callers × N_chunks) — an inner scan of all chunks per caller | `retrieval.py:514-529` |
| G-e | No query-level result cache; identical queries refetch everything | `retrieval.py:464` |
| G-f | LSI projector refits on any revision change (fit is cached only per revision) | `retrieval.py:435-438`; `mcp_server.py` `_retriever` |
| G-g | `plan_audit` (text-only) builds the **full index** (`_index` → whole-tree `_tree_hash`) and fits the LSI retriever just to score plan scope; cold cache ⇒ multi-second ⇒ client timeout. Same for any read tool that touches `_index`/`_learned`/`_retriever`. | `mcp_server.py:189,197,204`; `index.py:211-230` |

**Fixes (ordered by expected win).**

- **G1 Instrument.** Add `timings` (index/learned/retrieve/expand ms) and
  `cached` flags to `search_code`/`retrieve_context`; extend `eval/perf.py` with
  a p50/p95 latency-budget assertion including this long query.
- **G2 Collapse the ledger scan.** Keep a `ledger_n` counter in `meta`; cache
  `LearnedModels` by `(revision, ledger_n)`; never train inside a read tool —
  training is an explicit `train_models` call / background refresh.
- **G3 Cheap freshness.** Check a metadata fingerprint (path, size, mtime ns)
  every call; compute the content fingerprint only when metadata changed, then
  cache it. Reconciles C2 with hot-path cost.
- **G4 Lexical via the platform.** Use the existing FTS5 `chunks_fts` (SQL,
  native index) for BM25 candidates instead of the Python flat scan; keep the
  Python ranker as an offline scorer/fallback. Candidate *selection* can be
  approximate (inverted-index/topical prune) while scores stay exact.
- **G5 Precompute maps.** Build `symbol → chunks` and `path → chunks` on the
  index; `expand` and `callers_of` become O(fanout), not O(N).
- **G6 Result cache.** Bounded LRU keyed by
  `(revision, query, limit, include, exclude, expand, diversify)`.
- **G7 Adaptive pool + query hygiene.** Scale the candidate pool by repo size
  with early termination; cap query terms (drop stopwords/dupes, weight rare
  terms) so a long query does not multiply the scan.
- **G8 Incremental projector.** Keep the persisted LSI projector
  (`save_encoder`/`load_encoder`); update it incrementally on revision change
  rather than a full refit.
- **G9 Dense-first routing.** For non-identifier queries prefer the dense/ANN
  leg and the fused shortlist over a full lexical scan; keep RRF, measure.
- **G10 Read-path budget.** `plan_audit`, `propose_patch`, `judge_verdict`,
  and `validate_diff` must **never** build the index, retriever, or models;
  scope matching for `plan_audit` is lexical/deterministic only. Give every
  MCP tool a wall-clock budget and, on overrun, return partial results with
  `degraded: true` instead of hanging the client.
- **G11 Lazy index/retriever.** Build the index and fit the projector only for
  the discovery tools that need them, on an explicit `index_build` / background
  refresh, with a per-call `timeout_ms` and a structured error. `plan_audit`
  must answer on a cold cache in milliseconds.

**Research basis.** Parameterized dense–sparse fusion (arXiv 2609.22770);
Hierarchical BM25, sub-linear top-k (arXiv 2608.00229); hybrid-RAG latency
telemetry — RRF reaches ~95% of learned fusion, cross-encoder rerank adds
~80–130 ms (tim-ponomarev/hybrid-rag, ridwanspace/hybrid-rag-reference);
LSA partial index / incremental SVD (PMC5433746).

**DoD.** The cited query against the TrustSight snapshot is under budget warm
(target p95 < 500 ms) and per-stage timings show where the time goes; no read
tool triggers model training.

## Workstream H — Plan document tool (create / append / done / modify / notes / check)

The plan is a **JSON document**, not prose: a tree of sections, each with its
own body, status, notes, and checks. `plan.py` today only parses/audits text;
this makes the plan itself a first-class, editable artifact.

**The MCP server owns the plan.** The plan is not a file the assistant rewrites;
it is server-side state addressed by `(root, task_id)` and manipulated through
MCP tools, so any assistant model can read and change it in bounded increments
without ever regenerating the whole document. Every mutation goes through the
tool surface — there is no "edit the plan file" path — which is what lets the
same constraints as the rest of the harness apply uniformly:

- **Root authorization** — `root` passes `_authorize_root` (deny by default via
  `HARNESS_ALLOWED_ROOTS`), exactly like every other tool.
- **Containment** — the document is keyed by `(root, task_id)`; it never writes
  outside the store, and never touches the caller's checkout.
- **Evidence-gated status** — `done` is never a free-text edit; it requires the
  model-validation gate below, same spirit as `verified` in `status.py`.
- **Bounded operations** — append/attach are depth-capped; `modify`/`note`/
  `check` mutate one section at a time; the whole tree is only ever *read* back
  via `plan_get`/`plan_tree`. An LLM therefore makes a change by calling a tool,
  not by rewriting the plan file.
- **Versioned** — each mutation bumps `version` and appends to `notes`, so the
  ledger always has the record of who changed what.

**Document shape.**
```json
{
  "id": "plan-…", "root": "…", "task_id": "…", "version": 3, "title": "…",
  "sections": [
    { "id": "…", "title": "…", "body": "…",
      "status": "open|done|blocked|dropped",
      "notes":  [{"ts": "…", "text": "…"}],
      "checks": [{"name": "…", "status": "pass|fail", "evidence": "…"}],
      "validation": {"by": "model", "model": "…", "verdict": "pass|fail", "evidence": "…"},
      "children": [ /* same shape, recursive */ ] }
  ]
}
```

**Depth cap.** A section nests to depth **5** (root = 1); a deeper `attach`
is rejected with a structured error, keeping projection and validation bounded.

**Operations** (`PlanEditor` in `plan.py`, exposed as MCP tools):
- `plan_create(title, sections)` — new document.
- `plan_append` / `plan_attach` — add a section or subsection under a parent
  (depth-capped).
- `plan_complete(section_id)` — flips to `done` **only** when a model
  validation passes; otherwise stays `open`/`blocked`.
- `plan_modify(section_id, fields)` — edit title/body/status(open|blocked|dropped).
- `plan_note(section_id, text)` — append a timestamped note.
- `plan_check(section_id, name, status, evidence)` — record a check result.
- `plan_get(section_id?)` / `plan_tree()` — read the document or a subtree.
- `plan_audit` — rollup: a parent is `done` only when all children are `done`.

**Model-validation gate (the core rule).** A section cannot become `done`
until one of the configured models validates it against the section's own
criteria (its body / definition of done). Reuse the `MainLLM` provider
abstraction so validation is a real model call, and store the verdict (model
id + evidence) on the section. No verdict ⇒ no `done`. Enforced in
`PlanEditor.complete` as a `constraint`, never merely advisory.

- **H1** JSON `PlanDocument`/`PlanSection` (recursive, depth cap 5) +
  `PlanEditor`: create/append/attach/complete/modify/note/check/get/audit.
- **H2** Model-validation gate via `MainLLM`; `complete` stores
  `validation{by,model,verdict,evidence}` and refuses without a pass.
- **H3** MCP tools `plan_create`, `plan_append`, `plan_attach`,
  `plan_complete`, `plan_modify`, `plan_note`, `plan_check`, `plan_get`;
  persist the document per `(root, task_id)` in the store.
- **H4** `plan_audit` consumes the JSON document: per-section status rollup;
  drift ignores `done`/`dropped`; text-plan input still parsed → document for
  back-compat.
- **H5** Tests: create/append/attach; depth cap rejects depth 6; `complete`
  without validation stays open and with a passing verdict → `done`;
  modify/note/check round-trip; parent rollup.
- **H6** MCP owns the plan end to end: the store is the source of truth (there
  is no plan-file edit path), the toolset is feature-complete for incremental
  editing (`create`/`append`/`attach`/`complete`/`modify`/`note`/`check`/`get`/
  `tree`/`audit`), and every mutation is root-authorized, containment-checked,
  version-bumped and recorded in the ledger — so an assistant can change the
  plan without regenerating it.

This pulls Stage 1's "structured plans" forward and makes the plan the tool's
own source of truth.

## Task list (ordered, by dependency)

| ID | Task | Files | Depends |
|---|---|---|---|
| P0-1 | `paths.py` + replace 3 prefix checks | paths.py, verify.py, orchestrator.py, mcp_server.py | — |
| P0-2 | argv + `shell=False` + call sites | verify.py, orchestrator.py, mcp_server.py | P0-1 |
| P0-3 | env allowlist | verify.py | — |
| P0-4 | `isolation.py`, wire into run, ledger field | isolation.py, verify.py, orchestrator.py | P0-2 |
| P0-5 | disposable worktree | orchestrator.py, db.py | P0-1 |
| P0-6 | allowed roots, deny-by-default | mcp_server.py | P0-1 |
| P0-7 | `status.py` + `derive_status` | status.py, orchestrator.py | P0-4 |
| P0-8 | language-aware `checks.py` | checks.py, orchestrator.py | P0-2 |
| P0-9 | discovery snippets + `read_chunk` | mcp_server.py, retrieval.py | P0-6 |
| P1-1 | loop snapshot + union diff | orchestrator.py | P0-7 |
| P1-2 | content fingerprint + cache eviction | index.py, mcp_server.py | — |
| P1-3 | unique task ids | orchestrator.py | — |
| P1-4 | pyproject + CI + `rbg` | pyproject.toml, ci.yml, __main__.py | — |
| P1-5 | real LLM provider | providers.py | P1-4 |
| P1-6 | threat model + license | docs/, LICENSE | — |
| P2-1 | temporal/repo splits | training.py, db.py | P1-1 |
| P2-2 | conformal fix | models.py | P2-1 |
| P2-3 | metrics + CI reporting | training.py, drift.py | P2-2 |
| P2-4 | selector relabel | training.py | P2-3 |
| P2-5 | shadow gate | orchestrator.py, mcp_server.py | P2-3 |
| P2-6 | benchmark + compare + suites | eval/benchmark.py, eval/suites/ | P1-5, P2-4 |
| P2-7 | curated 50–100 real-bugfix suite | eval/suites/curated/ | **done** (50 tasks, verified) |
| P2-8 | SWE-bench Verified adapter | eval/suites/swebench/ | P2-6 |
| G1 | retrieval latency instrumentation + perf budget | mcp_server.py, eval/perf.py | **done** (latency_ms/budget_ms/degraded on search_code, retrieve_context) |
| G2 | ledger_n counter + learned cache, no train on read path | db.py, training.py, mcp_server.py | **done** (Store.ledger_count, read path never trains) |
| G3 | cheap metadata freshness + cached content fingerprint | index.py, mcp_server.py | P1-2 |
| G4 | BM25 candidates via libSQL FTS5 | retrieval.py, db.py | **done** (FTSBackend over chunks_fts, BM25 fallback) |
| G5 | precompute symbol→chunks / path→chunks maps | index.py, retrieval.py | **done** (expand builds one symbol→chunks pass) |
| G6 | bounded LRU result cache | retrieval.py, mcp_server.py | **done** (payload-only LRU, evicted on revision change) |
| G7 | adaptive candidate pool + query term cap | retrieval.py | **done** (query dedupe + term cap) |
| G8 | incremental LSI projector update | retrieval.py, db.py | **done** (warm-start: reuse proj, re-embed only) |
| G9 | dense-first routing for non-identifier queries | retrieval.py | **superseded by G4** (FTS5 removed the full lexical scan G9 routed around; RRF already fuses dense) |
| G10 | read-path budget; plan_audit never builds index; degraded results | mcp_server.py | **done** (plan_audit drops retriever; degraded flag) |
| G11 | lazy index/retriever + per-call timeout_ms | mcp_server.py, index.py | **done** (index/retriever already lazy; timeout_ms per call) |
| H1 | JSON PlanDocument/PlanSection (depth cap 5) + PlanEditor | plan.py | — | — **done**: PlanDocument/PlanSection + PlanEditor (depth cap 5)
| H2 | model-validation gate on complete via MainLLM | plan.py, providers | H1 | — **done**: model-validation gate on complete (verdict required)
| H3 | plan_create/append/attach/complete/modify/note/check/get tools | mcp_server.py, db.py | H1, H2 | — **done**: plan_create/append/attach/complete/modify/note/check/get/tree/audit tools, persisted per (root, task_id)
| H4 | plan_audit consumes JSON doc + rollup + back-compat text | plan.py, mcp_server.py | H1 | — **done**: plan_audit consumes stored JSON doc; text path retained
| H5 | plan editor tests (depth cap, validation gate, rollup) | tests/test_plan_editor.py | H1, H2 | — **done**: tests in test_harness.py (depth cap, gate, rollup, persist)
| H6 | MCP-owned plan: store is source of truth, complete incremental toolset, no file-edit path | mcp_server.py, db.py, plan.py | H3 | — **done**: MCP owns the plan; mutations versioned and ledgered

## Test plan (new)

`tests/test_sandbox.py` (escape, symlink, argv-injection, env,
isolation-required), `tests/test_status.py` (all five states incl. no-collector,
missing-checker, unisolated), `tests/test_loop_cache.py` (union diff,
content-revision, MCP freshness, unique ids), `tests/test_training_splits.py`
(no leakage, held-out coverage, shadow gate), `tests/test_mcp_tools.py` (roots
deny, snippet + `read_chunk` caps). Existing `tests/test_harness.py` extended,
not replaced.

## Rollout / compatibility

- `status` values change → five states. One-release `status_legacy` shim in
  `Report.to_dict`; ledger `schema_version` bump to 2 with a migration note.
- `Sandbox.run` signature change is breaking for `run_checks` profile strings;
  update in the same commit.
- Deny-by-default roots require `HARNESS_ALLOWED_ROOTS` in the opencode MCP
  config for the installed `rbg-harness` entry, or all tools error — document in
  README quickstart.

## Risks

| Risk | Mitigation |
|---|---|
| `bwrap`/`nsjail` absent ⇒ harness unusable | Clear error + documented dev escape hatch; CI installs `bubblewrap` |
| Worktree breaks non-git snapshots | Temp-copy fallback + test |
| Content-hash revision slower on big trees | git tree hash fast path; cached fingerprint |
| No model key in CI | ScriptedLLM for `baseline`+`full` in CI; `ReplayLLM` (recorded session-model transcript) covers provider end-to-end; live-model runs gated/manual |
| Calibration fix shifts metrics | Version bump; old artifacts ignored by version |

## Definition of done (programme)

1. Untrusted commands run isolated or not at all; paths cannot escape; roots are
   allowlisted.
2. No report claims `verified` for a check that did not run or tests that did not
   collect.
3. Repair loop records the true final change set; caches track a content revision.
4. Models train/calibrate on held-out data, report n+CI, stay shadow until the
   benchmark clears them.
5. `rbg eval` reproduces a published ablation across revisions; the provider
   completes a task end-to-end. — **done** via `rbg replay` (recorded
   session-model transcript, same `parse_edits` contract as a live provider);
   `examples/replay` fails before, passes after, under bwrap. Blind ablation
   on the 50-task suite: `blind-baseline` 0/50 (recall 0) vs `blind-full`
   50/50 (recall/MRR 1.0), coverage 0 → 1.0.

## Open item

- **Suite origin:** curated 50-task bugfix suite shipped in P2-7 (verified
  task-by-task by `build.py --check`). SWE-bench Verified adapter remains P2-8,
  gated on infra + a model key.

---

## Follow-up: caveat closure (task `caveat-closure`)

Closing the gaps the evaluation found after Phase 2.

| Task | Status | Verified |
|---|---|---|
| C0 land four manual-test fixes (compileall, PYTHONPYCACHEPREFIX, CLI dispatch, read-only plan steps) + regression tests + CI CLI run | **done** | commit `96d7180`; 128 tests |
| C1 make model validation real: `rule_risk`, out-of-sample Platt, `validation:learned` in `meta`, `rbg validate-models`, `ledger_farm.py`, `get_calibration.validated` | **done** | temporal 60/20/20, learned Brier 0.0004 vs rule 0.3665 → promoted; 130 tests |
| C2 hard retrieval suite (multi-file + distractors, Recall@1, ablation) | **done** | 54 tasks verified; hard hybrid recall@1 0.259 > lexical 0.241 > dense 0.241; 131 tests |
| C3 prove `plan_audit` causally (`PartialLLM` A/B) | **done** | partial-noaudit 0/8 vs partial-audit 8/8 on `suites/multifile` |
| C4 `eval/perf.py --budget-ms` semantics | **done** | unset budget reports `ok:null`; set budget grades and exits 2 |
| C5 SWE-bench Lite retrieval-only adapter | **done** | `eval/swebench_retrieval.py`: base_commit checkout, problem-statement query, file Recall@k/1 + MRR vs gold patch files, local-fixture test |
