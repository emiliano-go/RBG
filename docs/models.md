# Models

Every model below is small and deterministic given its inputs — learned heads
are fitted from the ledger, encoders are either fixed hashing/SVD or pinned
offline checkpoints. None of them generate text — the LLM is the only
generator. Each model is advisory unless it is the deterministic gate named in
its row.

### Learned models (`rbg/models.py`)

| Model | Kind | Purpose | Interface |
|---|---|---|---|
| **GBDT** | Gradient-boosted decision trees (numpy) | Risk score for a proposed action/change from `TaskFeatures`. | `fit(X, y)`, `predict(X)` |
| **PlattScaler** | Platt scaling | Maps raw risk scores to calibrated probabilities (logistic fit). | `fit(scores, labels)` → calibrated score |
| **ConformalCalibrator** | Split conformal prediction | Turns raw scores into prediction sets with a coverage guarantee; drives OOD detection. `fit()` holds out half the calibration slice for the quantile when it has at least 40 rows, and reports `mode="shared"` rather than implying a guarantee when it does not. | `fit(raw, labels, matrix=None)`, `predict_set(raw)` |
| **Policy** | Decision policy | Chooses the next step from the bounded action menu; holds the GBDT risk model and the uncertainty/OOD gates. | `select_action(...)` + `risk_model` |
| **TestSelector** | Ranker | Ranks tests by relevance to changed symbols (token overlap + learned weights). | `rank(tests, changed_symbols)` → `[TestRank]` |
| **FlakyModel** | Flakiness predictor | Estimates P(flaky) from a test's pass/fail history and duration. | `predict(history, duration)` |
| **ChangeImpactModel** | Symbol × test impact | Predicts which tests a symbol change affects, from symbol and test tokens. | `predict(symbol, test_path, test_tokens, ...)` |
| **PatchCorrectnessModel** | Static patch risk | P(regression) from patch shape (test edits, trivial changes, context overlap, size) — the "passing tests ≠ correct patch" signal (ODS/Leopard-style). | `predict(features)`, `train(X, y)` |
| **Reranker** | Linear feature reranker | Reorders retrieval hits using query/chunk features; trained from ledger relevance. | `score(query, chunk)`, `rerank(...)`, `train(examples)` |
| **CrossEncoderReranker** | Optional code cross-encoder | Heavier reranker behind the `semantic` extra; falls back to `Reranker`. | `rerank(query, hits, chunks)` |
| **DriftModel** | Shadow GBDT head over audit features | Scores plan-audit features (`drift_features`) to predict scope drift; reports `max_score` only and never overrides the deterministic audit verdict. | `fit(X, y)`, `predict(X)` |

### Retrieval and embedding models

These are used to *find* context, not to decide or generate. All are optional;
the deterministic path falls back cleanly when none is configured.

| Model | Kind | Purpose | Selected by |
|---|---|---|---|
| **DenseEncoder** | Deterministic hashing encoder | Reproducible dense embedding for chunks and experience memory; no training, identical across processes. | Default; `experience_encoder()` |
| **LSIEncoder** | Truncated-SVD latent semantic index | Corpus-fit dense vectors that capture term co-occurrence; warm-start `refresh` embeds new chunks without re-running SVD. | Default dense path |
| **StaticCodeEncoder** (`potion-code`) | 16M static code encoder | CPU-only embeddings for semantic retrieval; distilled from CodeRankEmbed, no torch, falls back to LSI/hash when absent. | `RBG_STATIC_MODEL=<path>` |
| **SentenceTransformerEncoder** (`code-rank`) | 137M code bi-encoder (CodeRankEmbed) | Higher-quality code embeddings with 8192-token context; requires `[semantic]`, falls back to the dependency-free encoder when unavailable. | `RBG_EMBEDDING_MODEL=<path>` |
| **gte-reranker** | 149M ModernBERT cross-encoder | Final-stage reranking of retrieval hits; requires `[semantic]`. | `RBG_RERANKER=gte-reranker` |

Experience memory: every verified run is written to the ledger as an
(issue → fix) experience and the single most similar prior fix is recalled
into context (SWE-Exp; N=1 retrieval). Encoder is `experience_encoder()` in
`rbg/retrieval.py`, storage `Store.save_experience`/`search_experiences`.

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

### Training (`rbg/training.py`)

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

### Drift detection (`rbg/drift.py`)

| Function | What it measures |
|---|---|
| `psi(reference, current, bins=10)` | Population Stability Index between two distributions. |
| `feature_drift(reference, current, names)` | Per-feature PSI + the max-drift feature + value. |
| `ood_rate(conformal, rows, threshold)` | Fraction of rows outside the conformal set (out-of-distribution). |
| `calibration_drift(ref_scores, ref_labels, cur_scores, cur_labels)` | Change in calibration (ECE-style) between windows. |
| `drift_report(learned, entries, *, feature_names, ...)` | Assembles the above into a `DriftReport` over a time-ordered ledger slice. |

Exposed as the `check_drift` tool and folded into `get_calibration`.

---
