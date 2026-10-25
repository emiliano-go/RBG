# RBG: Verified Control Plane for Coding Agents

RBG is an experimental, local-first control plane for AI-assisted code changes.
It combines repository-aware retrieval, one coding model, sandboxed
verification, and an auditable execution ledger.

**The model proposes changes. Deterministic checks decide what has actually been
verified.**

## Why RBG?

Coding agents can produce plausible patches without changing the right files,
running the right tests, or preserving evidence of what happened. RBG makes
those steps explicit and measurable.

- **Grounded context:** hybrid code retrieval and symbol-aware indexing.
- **Controlled execution:** repository access, filesystem paths, and checks are
  mediated by deterministic policy.
- **Evidence-backed verification:** compile and test results determine status,
  not the model's claim that a task is complete.
- **Durable patches:** every run exports the full unified diff before its
  disposable workspace is removed.
- **Auditable runs:** the execution ledger records decisions, checks, outcomes,
  and provenance.
- **Measured improvements:** learned components stay advisory until evaluated
  against deterministic baselines.

One generative model, not a swarm. Retrieval, verification, policy enforcement,
and small learned rankers surround it.

## Quickstart

Requires Python 3.12+ and `bwrap` (or `nsjail`) for isolated execution. The base
install is CPU-only and offline-capable.

```bash
pip install -e '.[dev]'
./install.sh --root ~/code          # register the MCP server with your clients

rbg demo                            # offline end-to-end repair on a throwaway repo
rbg index /path/to/repo
rbg retrieve /path/to/repo "where are edits applied"
```

A run returns a `Report` with `status`, per-gate evidence, the parsed plan, drift
findings, and `patch`; the complete diff produced in the isolated workspace.
The caller's checkout is never edited.

## How it works

```
index + retrieve context
  └─ one model proposes a plan and edits
       └─ edits applied in an isolated worktree/copy
            └─ compile → typecheck → targeted/full tests
                 └─ plan + scope drift audited each step
                      └─ status derived from recorded evidence
                           └─ patch + ledger exported
```

See [Architecture](docs/architecture.md) for the component map, data flow, and
plan-audit rules.

## Security and limitations

- Commands run inside a probed OS sandbox (`bwrap`, then `nsjail`). The
  candidate repository is mounted at a fixed path; host home directories are
  hidden; networking is disabled by default. With no working backend RBG
  refuses to run unless `RBG_ALLOW_UNISOLATED=1`, and an unisolated run can
  never be `verified`.
- The MCP server denies every repository by default. The installer grants only
  the roots you name with `--root`; there is no `$HOME` default.
- `verified` requires an executed and passing compile gate, tests that actually
  ran and passed (an all-skipped or unparsed suite stays `partial`), and an
  isolated sandbox.
- Learned models are advisory. The risk head stays in shadow until
  `validate_and_promote` beats the deterministic rule on a held-out slice.
- Synthetic benchmarks use an oracle that already knows the fix, or an oracle
  limited to retrieved files. They are causal component tests, **not** evidence
  of real LLM coding performance.
- Optional hosted embedding providers are not wired in: they would send
  repository source to a third party.

See the [threat model](docs/threat-model.md) for trust boundaries and residual
risk.

## Evaluation

The deterministic benchmark measures mechanisms, not model capability:

| Config | What it isolates | Result |
|---|---|---|
| `blind-baseline` | oracle with no retrieval | 0/50 resolved |
| `blind-full` | oracle limited to retrieved files | 50/50 resolved |
| `partial-noaudit` | one planned file per iteration, no drift feedback | 0/8 resolved |
| `partial-audit` | same model plus drift feedback | 8/8 resolved |

These are oracle-controlled ablations, not live-model results. For real model
runs, `eval.live_benchmark` records resolution, regressions, verification
coverage, latency, tokens, and optional cost on a held-out task subset. See
[Evaluation](docs/evaluation.md) for the suites, metrics, retrieval floor, and
reproducibility commands.

## Documentation

- [Architecture](docs/architecture.md): retrieval, control loop, plan drift
- [MCP tools](docs/mcp-tools.md): every tool, argument, and return
- [Models](docs/models.md): learned, statistical, and embedding models
- [Configuration](docs/configuration.md): install, roots, sandbox, model presets
- [Evaluation](docs/evaluation.md): benchmarks, ablations, model gates
- [Threat model](docs/threat-model.md): assets, boundaries, mitigations
- [Contributing](CONTRIBUTING.md): development setup, tests, PR expectations

## Project status

Experimental. The deterministic baseline runs without any model service; learned
components train from recorded runs and stay advisory. The compile/test gates
remain authoritative.
