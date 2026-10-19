# RBG Harness — Random Bullshit, GO!

A single-LLM control plane for coding agents: **one** generative model,
surrounded by deterministic retrieval, conventional verification, and small
learned models. The idea (RBG: *Random Bullshit, GO!*) is to let the LLM do
the one thing it is good at — propose a change — and make everything around
it boring, measurable and replayable.

> One generative LLM. Everything else is retrieval, conventional checks, or a
> small discriminative model. No swarm of agents.

The whole surface is exposed over **MCP** (see [MCP tools](#mcp-tools)), so any
agent or operator can drive it: index a repo, retrieve context, audit a plan,
rank tests, run checks, train the models, and read store/drift stats.

---

## Contents

- [Architecture](#architecture)
- [MCP tools](#mcp-tools) — all 25, with arguments and returns
- [Models](#models) — every learned model and statistical component
- [Retrieval](#retrieval)
- [The control loop](#the-control-loop)
- [Plan + step-level drift](#plan--step-level-drift)
- [Layout](#layout)
- [Quickstart](#quickstart)
- [Status](#status)

---

## Architecture

Three tiers, in the spec's terms:

1. **One generative LLM** (`MainLLM`) — writes the plan and the edits. Nothing
   else in the system generates text.
2. **Deterministic control plane** — indexing, retrieval, the repair loop,
   sandboxed verification, the ledger, plan/drift checks. Conventional code,
   fully replayable.
3. **Small learned models** — risk, calibration, test selection, change impact,
   reranking, flakiness. Trained from the run ledger, never generative.

Storage is **libSQL / Turso** (the only supported store). It holds repo chunks,
dense vectors (ANN index), the execution ledger, and trained model artefacts.

---

## MCP tools

Served by `harness/mcp_server.py` (stdio). `root` is always the absolute path
to a repository; the server indexes it lazily and caches the index per revision.

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
| `plan_audit` | `root`, `plan_text`, `task=""`, `changed_files`, `task_id` | With `task_id`, returns the stored JSON plan and its status rollup. Otherwise parses free text into steps and computes **scope drift** and a **completeness audit**. Advisory. |
| `plan_create` | `root`, `task_id`, `title`, `sections` | Creates the server-owned JSON plan for `(root, task_id)`; versions start at 1. |
| `plan_append` / `plan_attach` | `root`, `task_id`, `section`, `parent_id` | Appends a top-level section or nests one under an existing section (depth cap 5). |
| `plan_complete` | `root`, `task_id`, `section_id`, `status`, `evidence`, `validation` | Marks a section complete; the `validation` verdict is the model gate. Completion rolls up to parents. |
| `plan_modify` / `plan_note` / `plan_check` | `root`, `task_id`, `section_id`, … | Edit fields, add a note, or record a named check on a section. |
| `plan_get` / `plan_tree` | `root`, `task_id`, `section_id` | Read one section or the whole rollup tree. |
| `get_calibration` | `root` | Conformal calibration state and training metrics (coverage, ECE, thresholds). Advisory. |
| `check_drift` | `root` | Time-ordered drift / OOD check over the execution ledger: PSI per feature, calibration drift, OOD rate. Advisory. |
| `db_stats` | `root` | libSQL store statistics: persisted chunks, vectors, ledger rows and model artefacts. |
| `vector_search` | `root`, `query`, `limit=10` | On-disk ANN search through the libSQL vector index (cosine), joined back to chunk metadata. |

All tools return MCP `structuredContent`; every response carries the repo
`revision` so a result is attributable to a tree state.

---

## Models

Everything below is small, trained from the ledger, and deterministic given its
inputs. None of them generate text — the LLM is the only generator.

### Learned models (`harness/models.py`)

| Model | Kind | Purpose | Interface |
|---|---|---|---|
| **GBDT** | Gradient-boosted decision trees (numpy) | Risk score for a proposed action/change from `TaskFeatures`. | `fit(X, y)`, `predict(X)` |
| **PlattScaler** | Platt scaling | Maps raw risk scores to calibrated probabilities (logistic fit). | `fit(scores, labels)` → calibrated score |
| **ConformalCalibrator** | Split conformal prediction | Turns raw scores into prediction sets with a coverage guarantee; drives OOD detection. | `fit(raw, labels, matrix=None)`, `predict_set(raw)` |
| **Policy** | Decision policy | Chooses the next step from the bounded action menu; holds the GBDT risk model and the uncertainty/OOD gates. | `select_action(...)` + `risk_model` |
| **TestSelector** | Ranker | Ranks tests by relevance to changed symbols (token overlap + learned weights). | `rank(tests, changed_symbols)` → `[TestRank]` |
| **FlakyModel** | Flakiness predictor | Estimates P(flaky) from a test's pass/fail history and duration. | `predict(history, duration)` |
| **ChangeImpactModel** | Symbol × test impact | Predicts which tests a symbol change affects, from symbol and test tokens. | `predict(symbol, test_path, test_tokens, ...)` |
| **PatchCorrectnessModel** | Static patch risk | P(regression) from patch shape (test edits, trivial changes, context overlap, size) — the "passing tests ≠ correct patch" signal (ODS/Leopard-style). | `predict(features)`, `train(X, y)` |
| **Reranker** | Linear feature reranker | Reorders retrieval hits using query/chunk features; trained from ledger relevance. | `score(query, chunk)`, `rerank(...)`, `train(examples)` |
| **CrossEncoderReranker** | Optional code cross-encoder | Heavier reranker behind the `semantic` extra; falls back to `Reranker`. | `rerank(query, hits, chunks)` |

Experience memory: every verified run is written to the ledger as an
(issue → fix) experience and the single most similar prior fix is recalled
into context (SWE-Exp; N=1 retrieval). Encoder is `experience_encoder()` in
`harness/retrieval.py`, storage `Store.save_experience`/`search_experiences`.

Supporting types: `TaskFeatures` (the feature vector fed to GBDT/impact),
`Decision` (a chosen action + rationale), `TestRank` (a ranked test),
`select_action(*, has_failures, full_suite_required, full_suite_ran, uncertainty,
ood)` — the deterministic fallback the learned policy refines.

#### The action menu

`select_action` picks from a fixed, bounded menu (`ACTIONS`), so the agent can
never invent an action:

```
retrieve_more, expand_call_graph,
run_targeted_tests, run_related_suite, run_full_suite,
run_static_analyzer, run_security_checks, rerun_flaky_failure,
ask_main_llm_to_revise, stop_with_verified_result, escalate_for_human_review
```

### Training (`harness/training.py`)

`train_from_ledger(root, index=None, store=None)` reads the execution ledger,
builds examples per component, and fits the models; it returns `LearnedModels`
(a bundle of `policy`, `selector`, `reranker`, `impact`, `conformal`, plus
`metrics`, `drift`, `ood_threshold`, `reference`). Example builders:

- `risk_examples` → `(X, y)` for the GBDT risk head (Brier-scored).
- `selector_examples`, `reranker_examples`, `impact_examples` → per-component
  supervision from recorded runs.
- `test_histories` → pass/fail histories for the flaky model.

Call it via the `train_models` MCP tool or from Python.

#### Shadow-only until evidence promotes it

A learned risk head never drives a decision on its own say-so. `validated`
starts `False`, and only `validate_and_promote(root)` (the `rbg validate-models`
command, or `train_models(validate=True)`) can flip it. Promotion requires a
**temporal** test slice of at least 20 rows with both classes, a Brier score
lower than the deterministic `rule_risk` it would replace, and conformal
coverage within tolerance of the target. The verdict is stored keyed by
`(revision, ledger_n)`, so it applies to exactly the artifact it was earned on
and is discarded when the tree or ledger moves on.

`eval/ledger_farm.py` produces the ledger that gate needs: it runs each suite
task twice into one root — once with retrieval off (fails, conditioning label)
and once with the oracle fix (resolves) — interleaved so the temporal split has
both classes, then trains and validates.

```bash
python3 -m eval.ledger_farm --suite eval/suites/curated --tasks 50
```

### Drift detection (`harness/drift.py`)

| Function | What it measures |
|---|---|
| `psi(reference, current, bins=10)` | Population Stability Index between two distributions. |
| `feature_drift(reference, current, names)` | Per-feature PSI + the max-drift feature + value. |
| `ood_rate(conformal, rows, threshold)` | Fraction of rows outside the conformal set (out-of-distribution). |
| `calibration_drift(ref_scores, ref_labels, cur_scores, cur_labels)` | Change in calibration (ECE-style) between windows. |
| `drift_report(learned, entries, *, feature_names, ...)` | Assembles the above into a `DriftReport` over a time-ordered ledger slice. |

Exposed as the `check_drift` tool and folded into `get_calibration`.

---

## Retrieval

`harness/retrieval.py`.

| Component | Role |
|---|---|
| **FTS5 / BM25** | Lexical ranking from libSQL's `chunks_fts` index (BM25), with an in-memory BM25 fallback. |
| **DenseEncoder** | Deterministic hashing-based dense embedding (`fit`/`encode`), so runs are reproducible across processes. |
| **LSIEncoder** | Truncated-SVD latent semantic indexing over the chunk corpus; `refresh` re-embeds new chunks against the existing projector (warm start, no SVD). |
| **HybridRetriever** | Fuses the ranked lists with reciprocal-rank fusion, applies **path-aware weighting** (source `1.0` > docs `0.6` > tests `0.4`), a symbol-match boost, and optional MMR diversification. Over-fetches ×2 before filtering by `include`/`exclude` globs. |
| **Reranker** | Optional learned final-stage reordering (see Models). |

`Hit` carries the chunk plus its `commit_hash`, so retrieval results are
provenance-linked to a revision.

Measured with `eval/effectiveness.py` (Recall@k, MRR, and test-selection
precision/recall).

---

## The control loop

`harness/orchestrator.py`. The loop is:

```
retrieve context
  parse plan
  [plan coverage check]
  loop:
    ask the LLM to edit (given the plan + diagnostics)
    apply edits -> re-index -> compile
    [scope drift: missing / extra]   <- fed back into the next iteration
    break when compile ok AND drift ok
  typecheck
  rank tests
  run tests (with flaky-rerun recovery)
  [completeness audit]
  report
```

`missing` planned files are fed back as guidance so the repair loop closes the
gap. Unmet steps after the last iteration are recorded as a **scope reduction**,
never a hard failure — the compile/test gates stay authoritative for success.

The `Report` carries `plan`, `drift` (per stage + per iteration) and
`completeness`.

---

## Plan + step-level drift

`harness/plan.py`. There are two paths:

- The orchestrator still accepts the main LLM's **free-text** plan, parses it
  into concrete steps, and checks it after every step.
- Over MCP, the **plan is a server-owned JSON document** keyed by
  `(root, task_id)`: `PlanDocument`/`PlanSection` with an explicit depth cap,
  status rollup, and a completion gate that requires evidence. Every mutation
  is root-authorized, containment-checked, version-bumped, and written to the
  ledger. There is no plan-file edit path, so an assistant can change the plan
  without regenerating it.

The checks below are deterministic — no new model, no extra LLM call.

- **`parse_plan(text, known_files)`** — each line is a step (leading `-`, `*`,
  `1.`, `Step 1:` stripped). File tokens are recorded **only if they exist in
  the repo**. A step with a run/test/verify verb is `verify`, otherwise `edit`.
- **`scope_drift(plan, changed_files)`** — planned-but-unchanged (`missing`) and
  changed-but-unplanned (`extra`).
- **`audit_completion(index, changed_symbols, changed_files, context_files)`** —
  callers of changed symbols neither edited nor retrieved. Underscore-private,
  dunder and generic method names (`run`, `to_dict`, `main`, …) are skipped: a
  bare method name collides across the tree, so it carries no task signal.
- **`format_guidance(plan, drift)`** — repair-loop guidance for missing steps.

Public API: `Plan`, `PlanStep`, `Drift`, `parse_plan`, `scope_drift`,
`audit_completion`, plus `PlanDocument`/`PlanSection`/`PlanEditor` for the
stored document. Exposed as `plan_audit` and the `plan_*` toolset.

---

## Layout

```
harness/
  index.py        repo indexing: symbols, chunks, call graph
  db.py           libSQL/Turso store (chunks, vectors, plans, ledger, models)
  retrieval.py    FTS5 + dense + LSI hybrid retrieval, reranker
  models.py       learned models + the action menu
  training.py     train every model from the ledger
  verify.py       sandbox, pytest parsing, the ledger
  isolation.py    bwrap/nsjail probe and argv wrapping
  paths.py        containment checks (resolve-within)
  workspace.py    disposable git worktree for the edit/verify loop
  status.py       evidence-derived terminal state
  checks.py       language-aware compile/test commands
  providers.py    OpenAI-compatible provider + recorded-session replay
  drift.py        PSI / feature-shift / OOD drift reports
  plan.py         JSON plans, plan editing, step-level drift checks
  orchestrator.py the control loop (the "harness")
  mcp_server.py   MCP surface (25 tools)
  __main__.py     CLI: demo, replay, index, retrieve, eval
eval/
  benchmark.py      configs, metrics, and the deterministic benchmark
  effectiveness.py  retrieval + test-selection metrics (Recall@k, MRR)
  perf.py           tool latency benchmark (p50/p95 + budget gate)
  retrieval_ablation.py  lexical/dense/hybrid recall@1/@k/MRR ablation
  swebench_retrieval.py  retrieval-only SWE-bench Lite file Recall@k/MRR
  suites/smoke/     3-task suite run in CI
  suites/curated/   54 verified bugfix tasks + build.py --check
  suites/hard/      curated with the answer's names stripped + distractors
  suites/multifile/ the 8 multi-file tasks, for the plan-audit ablation
examples/replay/   buggy module + test + recorded transcript (no key)
tests/
  test_harness.py   end-to-end suite
  test_sandbox.py   containment + isolation
  test_status.py    evidence-derived status
mcp_main.py       stdio launcher for the MCP server
```

---

## Quickstart

Requires Python 3.12+ and [libSQL](https://github.com/tursodatabase/libsql)
(`pip install libsql`); only `numpy` is needed beyond that.

```bash
# install the package (adds the `rbg` and `rbg-mcp` console scripts)
pip install -e '.[dev]'

# run the end-to-end demo on a throwaway repo
rbg demo

# run a recorded session model (no API key) end-to-end on a real failing test
rbg replay examples/replay examples/replay/transcript.json \
  "fix median so even-length lists average the two middle values"

# index a repo and retrieve context
rbg index /path/to/repo
rbg retrieve /path/to/repo "where are edits applied"

# measure the retriever
python3 -m eval.effectiveness /path/to/repo --sample 200

# run the deterministic benchmark and compare two configs
rbg eval --suite eval/suites/smoke --config baseline --out baseline.json
rbg eval --suite eval/suites/smoke --config full --out full.json
rbg compare baseline.json full.json

# the curated bugfix suite (regenerate + verify task-by-task; also writes hard/)
python3 -m eval.suites.curated.build --check
rbg eval --suite eval/suites/curated --config baseline --out curated.json

# retrieval recall ablation on the hard suite: the query has the answer's
# identifiers removed and every repo carries distractor modules
python3 -m eval.retrieval_ablation eval/suites/hard --k 5

# retrieval-only SWE-bench Lite: check out each base_commit and score whether
# the retriever surfaces the files the gold patch touches
python3 -m eval.swebench_retrieval --limit 120 --out swebench-lite.json

# causal retrieval ablation: the model only sees retrieved context
rbg eval --suite eval/suites/curated --config blind-baseline --out blind-base.json
rbg eval --suite eval/suites/curated --config blind-full --out blind-full.json
rbg compare blind-base.json blind-full.json

# causal plan-audit ablation: one oracle that edits one planned file per
# iteration, with and without the drift feedback that names the rest
rbg eval --suite eval/suites/multifile --config partial-noaudit --out noaudit.json
rbg eval --suite eval/suites/multifile --config partial-audit --out audit.json
rbg compare noaudit.json audit.json

# farm a real ledger, train, and gate the risk head on held-out evidence
python3 -m eval.ledger_farm --suite eval/suites/curated --tasks 50
rbg validate-models /path/to/repo --min-test 20

# run the suite
python3 -m pytest -q
```

On the 50-task suite, `blind-baseline` resolves **0/50** (recall 0) and
`blind-full` resolves **50/50** (recall/MRR 1.0), so the ablation measures
retrieval causally rather than assuming it.

Those queries name the function they are about, so retrieval is easy on the
curated suite. The hard suite strips those names from the query and adds
distractor modules that share the vocabulary: recall@1 drops to ~0.26 for the
hybrid retriever, ahead of lexical-only and dense-only, and `pytest` fails if
any mode ever returns to triviality.

The plan-audit ablation is causal too: `partial-noaudit` uses a model that
edits one planned file per iteration and stops there, and `partial-audit`
gives it the drift feedback that names the remaining files. On the 8
multi-file tasks the former resolves 0/8 and the latter 8/8.

---

## Status

Experimental. The deterministic baseline (index, retrieval, gates, ledger)
runs without any model service; the learned components train from recorded
runs and stay advisory — the compile/test gates remain authoritative. The
learned risk head stays in shadow (reason code `learned_risk_in_shadow`) until
`validate_and_promote` has beaten the deterministic rule on a held-out slice
with recorded evidence.

`bwrap` (or `nsjail`) is required: when no isolation backend probes successfully
the harness refuses to run unless `HARNESS_ALLOW_UNISOLATED=1` is set, and an
unisolated run is never reported as verified. CI installs bubblewrap and lifts
Ubuntu 24.04's AppArmor userns restriction before running the 125-test suite.
