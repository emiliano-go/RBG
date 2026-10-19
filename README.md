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
- [MCP tools](#mcp-tools) — all 15, with arguments and returns
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
| `find_symbol` | `root`, `name`, `exact=false`, `path`, `kind` | Definition location(s) plus every referencing file. `path` filters definition paths by glob; `kind` filters `function`/`class`/`constant`. |
| `get_change_impact` | `root`, `symbols` | Predicted **and** graph-derived impact scope: referenced files, likely tests, fan-out. Advisory. |
| `rank_tests` | `root`, `changed_symbols`, `limit=20` | Candidate tests ranked by regression-detection value, each with evidence (`covers_changed_symbol`, …), predicted relevance/flakiness and duration. Mandatory suites are flagged, never skipped. Advisory. |
| `run_checks` | `root`, `profile=compile`, `timeout_s=120` | Runs an approved check profile in the sandbox: `compile`, `collect_tests`, or `tests`. Returns pass/fail and captured output. |
| `get_diagnostics` | `root`, `task_id` | The recorded compiler/test evidence (diagnostics) for a task handle. |
| `get_task_status` | `root`, `task_id` | Status, verified gates, and recorded checks for a task handle. |
| `propose_patch` | `root`, `path`, `content` | Validates a patch proposal (syntax/shape) and returns the verdict. **Does not apply it.** |
| `train_models` | `root` | Trains risk, selector, reranker and impact models from the ledger; returns metrics and calibration state. |
| `plan_audit` | `root`, `plan_text`, `task=""`, `changed_files` | Parses a free-text plan into steps, computes **scope drift** (planned-but-unchanged / changed-but-unplanned) and a **completeness audit** (callers of changed symbols neither edited nor retrieved). Advisory. |
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
| **Reranker** | Linear feature reranker | Reorders retrieval hits using query/chunk features; trained from ledger relevance. | `score(query, chunk)`, `rerank(...)`, `train(examples)` |

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
| **BM25** | Lexical ranking over tokenised chunks. |
| **DenseEncoder** | Deterministic hashing-based dense embedding (`fit`/`encode`), so runs are reproducible across processes. |
| **LSIEncoder** | Truncated-SVD latent semantic indexing over the chunk corpus. |
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

`harness/plan.py`. The main LLM still returns a free-text plan; the harness
parses it into concrete steps and checks it after every step. Deterministic —
no new model, no extra LLM call.

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
`audit_completion`. Exposed as the `plan_audit` tool.

---

## Layout

```
harness/
  index.py        repo indexing: symbols, chunks, call graph
  db.py           libSQL/Turso store (chunks, vectors, ledger, models)
  retrieval.py    BM25 + dense + LSI hybrid retrieval, reranker
  models.py       learned models + the action menu
  training.py     train every model from the ledger
  verify.py       sandbox, pytest parsing, the ledger
  drift.py        PSI / feature-shift / OOD drift reports
  plan.py         structured plans + step-level drift checks
  orchestrator.py the control loop (the "harness")
  mcp_server.py   MCP surface (15 tools)
  __main__.py     demo CLI
eval/
  effectiveness.py  retrieval + test-selection metrics (Recall@k, MRR)
  perf.py           tool latency benchmark
tests/
  test_harness.py   end-to-end suite
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

# the 50-task curated bugfix suite (regenerate + verify task-by-task)
python3 -m eval.suites.curated.build --check
rbg eval --suite eval/suites/curated --config baseline --out curated.json

# causal retrieval ablation: the model only sees retrieved context
rbg eval --suite eval/suites/curated --config blind-baseline --out blind-base.json
rbg eval --suite eval/suites/curated --config blind-full --out blind-full.json
rbg compare blind-base.json blind-full.json

# run the suite
python3 -m pytest -q
```

On the 50-task suite, `blind-baseline` resolves **0/50** (recall 0) and
`blind-full` resolves **50/50** (recall/MRR 1.0), so the ablation measures
retrieval causally rather than assuming it.

---

## Status

Experimental. The deterministic baseline (index, retrieval, gates, ledger)
runs without any model service; the learned components train from recorded
runs and stay advisory — the compile/test gates remain authoritative.
