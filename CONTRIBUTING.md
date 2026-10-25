# Contributing

Development setup, tests, and layout.

## Layout

```
rbg/
  index.py        repo indexing: symbols, chunks, call graph
  db.py           libSQL/Turso store (chunks, vectors, plans, ledger, models)
  retrieval.py    FTS5 + dense + LSI hybrid retrieval, reranker
  models.py       learned models + the action menu
  training.py     train every model from the ledger
  verify.py       sandbox, pytest parsing, the ledger
  isolation.py    bwrap/nsjail probe and argv wrapping
  paths.py        containment checks (resolve-within)
  workspace.py    isolated worktree/copy + durable patch export
  status.py       evidence-derived terminal state
  checks.py       language-aware compile/test commands
  providers.py    OpenAI-compatible provider + recorded-session replay
  drift.py        PSI / feature-shift / OOD drift reports
  plan.py         JSON plans, plan editing, step-level drift checks
  orchestrator.py the control loop (the "harness")
  mcp_server.py   thin JSON-RPC transport (timeouts, rate limits)
  mcp_service.py  authorization, caches, indexing, retrieval, learned models
  mcp_tools.py    retrieval/diagnostic/check tools
  mcp_plan.py      server-owned plan tools
  mcp_schemas.py   tool/resource registration
  __main__.py     CLI: demo, replay, index, retrieve, eval
eval/
  benchmark.py      configs, metrics, and the deterministic benchmark
  live_benchmark.py real-model runs: resolution, latency, tokens, cost
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
  test_rbg.py   end-to-end suite
  test_sandbox.py   containment + isolation
  test_status.py    evidence-derived status
  test_properties.py hypothesis paths/plans/artifact invariants
  test_workspace.py  isolated start state + durable patch export
mcp_main.py       stdio launcher for the MCP server
```

---

## Tests and lint

```bash
pip install -e '.[dev]'
python -m pytest -q
ruff check rbg eval tests
```

CI runs the suite on Python 3.12 and 3.13 with bubblewrap installed, plus
`pip-audit` on the dependency set. Keep the sandbox and retrieval-ablation
checks as regression gates.
