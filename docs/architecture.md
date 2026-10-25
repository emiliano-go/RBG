# Architecture

Three tiers, in the spec's terms:

1. **One generative LLM** (`MainLLM`); writes the plan and the edits. Nothing
   else in the system generates text.
2. **Deterministic control plane**: indexing, retrieval, the repair loop,
   sandboxed verification, the ledger, plan/drift checks. Conventional code,
   fully replayable.
3. **Small learned models**: risk, calibration, test selection, change impact,
   reranking, flakiness. Trained from the run ledger, never generative.

Storage is **libSQL / Turso** (the only supported store). It holds repo chunks,
dense vectors (ANN index), the execution ledger, and trained model artefacts.

Local databases run in **WAL mode** with a long busy timeout, so several agents
(say, one MCP server process each) can read the same project in parallel while
writes queue instead of failing with `database is locked`. Reads are fully
concurrent; libSQL still serialises writes, so the store keeps write
transactions short and takes the write lock up front (`BEGIN IMMEDIATE`). The
first process to see a revision builds and persists the index under a
cross-process lock; the rest wait briefly and reuse it. Tune with
`RBG_DB_BUSY_TIMEOUT_MS` (default `30000`) and
`RBG_DB_SYNCHRONOUS` (`FULL` by default; `NORMAL` trades power-loss
durability for fewer fsyncs).

---

## Retrieval

`rbg/retrieval.py`.

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

`rbg/orchestrator.py`. The loop is:

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
never a hard failure; the compile/test gates stay authoritative for success.

The `Report` carries `plan`, `drift` (per stage + per iteration) and
`completeness`.

---

## Plan + step-level drift

`rbg/plan.py`. There are two paths:

- The orchestrator still accepts the main LLM's **free-text** plan, parses it
  into concrete steps, and checks it after every step.
- Over MCP, the **plan is a server-owned JSON document** keyed by
  `(root, task_id)`: `PlanDocument`/`PlanSection` with an explicit depth cap,
  status rollup, and a completion gate that requires evidence. Every mutation
  is root-authorized, containment-checked, version-bumped, and written to the
  ledger. There is no plan-file edit path, so an assistant can change the plan
  without regenerating it.

The checks below are deterministic; no new model, no extra LLM call.

- **`parse_plan(text, known_files)`**: each line is a step (leading `-`, `*`,
  `1.`, `Step 1:` stripped). File tokens are recorded **only if they exist in
  the repo**. A step with a run/test/verify verb is `verify`, otherwise `edit`.
- **`scope_drift(plan, changed_files)`**: planned-but-unchanged (`missing`) and
  changed-but-unplanned (`extra`).
- **`git_changed(root)`**: collects modified, added, deleted, renamed, and
  untracked paths directly from git, so the audit never trusts the model's own
  report of what it changed. When the root is a workspace rather than a
  checkout, nested checkouts one level down are aggregated with their directory
  prefix (`repo_1/src/a.py`), so cross-repo drift needs no caller-supplied
  `changed_files` list.
- **`audit_document(doc, changed, root)`**: the hard gate. A declared file
  that is unchanged, an unplanned write or deletion, a missing or failed
  acceptance item, or a section marked done without a passing validation
  blocks. A section that declares no scope at all is reported as unverifiable,
  not as clean.
- **`section_validation(...)`**: the validator behind `plan_complete`. It
  fails on any document-level blocker, so an unplanned write cannot be excused
  by completing a different section.
- **`DriftModel`**: an optional GBDT head over the audit features. It runs in
  shadow, reports `max_score`, and never overrides the deterministic verdict.
- **`audit_completion(index, changed_symbols, changed_files, context_files)`**:
  callers of changed symbols neither edited nor retrieved. Underscore-private,
  dunder and generic method names (`run`, `to_dict`, `main`, …) are skipped: a
  bare method name collides across the tree, so it carries no task signal.
- **`format_guidance(plan, drift)`**: repair-loop guidance for missing steps.

Plan sections may declare `files`, `allowed_paths`, `symbols`, `artifacts`,
`acceptance`, `intent`, and `non_goals`. Old plans without these fields still
load; their drift is reported as unverifiable rather than assumed clean.

Hardening rules:

- **Baseline at creation.** Files already dirty when the plan is created are
  recorded and subtracted from "unplanned", so a dirty checkout does not fail
  every task. They still count toward a planned file's observed change.
- **Acceptance cannot self-certify.** A `status` field on an acceptance item is
  ignored. The verdict comes from a recorded check the audit can attribute.
- **Artifacts must resolve inside the root.** Absolute paths, `..`, and
  symlinks that leave the root are rejected.
- **Bounded.** A plan over 5,000 sections or a working tree over 20,000 changed
  files blocks instead of truncating silently. Ledger rows keep a bounded
  summary, not the full change set.
- **Scope edits are visible.** Changing `files`, `allowed_paths`, `symbols`, or
  `acceptance` increments `scope_version`; the next audit warns
  `plan_scope_changed`.
- **No stale completion.** A completed section records the changed set it was
  completed against; later changes to those files warn
  `changed_after_completion`.

Public API: `Plan`, `PlanStep`, `Drift`, `parse_plan`, `scope_drift`,
`audit_completion`, `ChangedSet`, `git_changed`, `audit_document`,
`section_validation`, plus `PlanDocument`/`PlanSection`/`PlanEditor` for the
stored document. Exposed as `plan_audit` and the `plan_*` toolset.

---


## Patch lifecycle

The orchestrator never edits the caller's checkout. A git checkout gets a
temporary worktree at `HEAD` with staged, unstaged, and untracked start state
applied; anything else is copied to a temporary directory. `cleanup()` exports
the complete unified diff before removing the workspace, and the returned
`Report.patch` remains readable afterwards. Failure to create an isolated
workspace is an error, not a silent in-place fallback.

## Fault localization

`rbg/fault_localization.py` collects per-test Python line coverage with a small
pytest tracer plugin and computes deterministic Ochiai suspiciousness over
failing vs passing spectra. It ranks lines and maps them back to symbols for
retrieval/test selection. It is opt-in (`RBG_FAULT_LOCALIZATION=1`), advisory,
and never changes a verification verdict.

## Verification status

`verified` requires recorded evidence: a passing compile gate, at least one test
that actually ran and passed, and an isolated sandbox. A zero exit code alone is
not enough. An empty suite, an all-skipped suite, an unparsed test runner, or a
missing isolation backend leaves the run `partial`/`failed`, never `verified`.
