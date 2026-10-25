# Evaluation

The suites below are **oracle-controlled mechanism tests**, not live-model
benchmarks. `OracleLLM` already knows the correct edit, and `BlindLLM` restricts
that known edit to files retrieval surfaced. Results therefore measure
retrieval, plan-drift feedback, and verification; not whether a real LLM can
repair real tasks. Use the live-model benchmark for that claim; it records
resolution, regressions, verification coverage, latency, tokens, and optional
cost for real model runs.

```bash
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

## Live-model runs

Real-model runs are reported separately from oracle ablations:

```bash
RBG_LLM_API_KEY=... RBG_LLM_MODEL=... \
  python3 -m eval.live_benchmark --suite eval/suites/smoke --repeats 3 \
  --tasks discount,off_by_one --price-in 0.15 --price-out 0.60 --out live.json
```

Use a held-out repository or task subset, keep the configuration with the
result, and compare repeats rather than a single lucky run.

## Fault localization

The deterministic localizer is evaluated against reference bug locations:

```bash
python3 -m eval.fault_localization_eval --suite eval/suites/smoke --out fl.json
python3 -m eval.fault_localization_eval --suite eval/suites/fl-broad --out fl-broad.json
```

The broad suite covers insertion-only, deletion-only, multi-line, multi-file,
interacting-failure, and indirect-constant fixes. It reports every task,
skipped ids, per-category Hit@1/5/10, and an overall baseline currently around
0.83; the smoke suite remains the fast 1.0 CI guard.

`metrics.hit_at_1/5/10` reports how often a line changed by the fix appears in
the top suspects. CI enforces the smoke-suite baseline (`hit_at_1 = 1.0`) and
requires every selected task to be evaluated; diagnostic runs can pass
`--allow-incomplete`. To measure its effect on repair, A/B the live benchmark:

```bash
python3 -m eval.live_benchmark --suite <held-out-suite> --repeats 3 --out without.json
python3 -m eval.live_benchmark --suite <held-out-suite> --repeats 3 \
  --fault-localization --out with.json --compare-with without.json
```

`--compare-with` adds candidate-minus-baseline deltas for resolution,
regressions, coverage, latency, tokens, and cost.

No training rows are needed for localization itself.

Before changing a default encoder or reranker, measure it:

```bash
python -m eval.core_bench --repo /path/to/repo --data core_bench.jsonl --out cand.json
python -m eval.swe_rebench --repo /path/to/repo --data swe_rebench.jsonl --out cand.json
python -m eval.repobench --repo /path/to/repo --data mined_prs.jsonl --out cand.json
python -m eval.model_gate --baseline base.json --candidate cand.json
```
