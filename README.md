# RBG Harness — Random Bullshit, GO!

A single-LLM control plane for coding agents: one generative model, surrounded
by deterministic retrieval, verification and small learned models. The idea
(RBG: *Random Bullshit, GO!*) is to let the LLM do the one thing it is good at
— propose a change — and make everything around it boring, measurable and
replayable.

> One generative LLM. Everything else is retrieval, conventional checks, or a
> small discriminative model. No swarm of agents.

## What it does

- **Index** a repo into symbols and chunks, stored in a libSQL (Turso) database.
- **Retrieve** context with a hybrid of BM25 + a deterministic dense encoder +
  LSI, fused with reciprocal-rank fusion and path-aware weighting.
- **Orchestrate**: retrieve → plan → edit → apply → re-index → compile →
  typecheck → rank tests → run tests, with a repair loop fed by diagnostics.
- **Verify**: sandboxed command execution, pytest parsing, and a Ledger of
  outcomes.
- **Plan**: parse the LLM's free-text plan into concrete steps and check, after
  every step, that the agent did what it said it would (scope drift + a
  completeness audit).
- **Learn**: small models (GBDT risk, Platt + conformal calibration, test
  selector, change-impact, reranker) trained from the run ledger.
- **Serve** all of the above over MCP.

## Layout

```
harness/
  index.py        repo indexing: symbols, chunks, call graph
  db.py           libSQL/Turso store (chunks, vectors, ledger)
  retrieval.py    BM25 + dense + LSI hybrid retrieval
  models.py       learned models + action selection
  training.py     train models from the ledger
  verify.py       sandbox, pytest parsing, ledger
  drift.py        feature-shift / PSI drift reports
  plan.py         structured plans + step-level drift checks
  orchestrator.py the control loop (the "harness")
  mcp_server.py   MCP surface (tools + resources)
  __main__.py     demo CLI
eval/
  effectiveness.py  retrieval + test-selection metrics (Recall@k, MRR)
  perf.py           tool latency benchmark
tests/
  test_harness.py   end-to-end suite
mcp_main.py       stdio launcher for the MCP server
```

## Quickstart

Requires Python 3.12+ and [libSQL](https://github.com/tursodatabase/libsql)
(`pip install libsql`); only `numpy` is needed beyond that.

```bash
# run the end-to-end demo on a throwaway repo
python3 -m harness demo

# index a repo and retrieve context
python3 -m harness index /path/to/repo
python3 -m harness retrieve /path/to/repo "where are edits applied"

# measure the retriever
python3 -m eval.effectiveness /path/to/repo --sample 200
```

## Status

Experimental. Deterministic baseline (index, retrieval, gates, ledger) runs
without any model service; the learned components train from recorded runs.
