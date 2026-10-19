# Threat model

What the harness defends against, what it assumes, and where the boundary
actually is. The guiding rule: an LLM reply and repository contents are
**untrusted input**; nothing they contain may become an action without passing
a deterministic gate.

## Assets

- The operator's checkout and machine (files, credentials on disk, network).
- The integrity of the verification verdict: `verified` must mean evidence
  supports it, never that a model claimed it.
- The execution ledger (the evidence the learned models train on).

## Trust boundaries

| Boundary | Control | Where |
|---|---|---|
| Repo path → filesystem | Containment: `resolve_within` resolves symlinks and rejects anything outside the root (`..`, sibling-prefix, symlink escape). | `harness/paths.py` |
| MCP client → repository | Deny by default: no path is served unless it is inside `HARNESS_ALLOWED_ROOTS`. | `harness/mcp_server.py` |
| Model output → edits | Every edit path goes through `resolve_within`; writes land in a disposable git worktree, and the caller's checkout is never mutated. | `harness/workspace.py`, `harness/orchestrator.py` |
| Command → OS | A required isolation backend (`bwrap`, then `nsjail`), probed once with a trivial command; commands run argv-only (`shell=False`). | `harness/isolation.py`, `harness/verify.py` |
| Parent env → child | Allowlist only (`PATH`, `HOME`, `LANG`, temp vars, `LC_*`, `PYTHON*`), never a denylist; extend deliberately with `HARNESS_ENV_ALLOW`. | `harness/verify.py` |
| Evidence → status | `verified` is derived from recorded command results, never from `all([])`. An unisolated run can never be `verified`. | `harness/status.py` |

## Adversaries considered

1. **A malicious or compromised repository.** A checkout may ship a hostile
   `Makefile`, `conftest.py`, or `package.json` that runs arbitrary code. The
   harness will execute the project's own checks — that is the point — so they
   are executed **inside the sandbox**, with the network off and the environment
   stripped to the allowlist. Reading is safe; running is confined.
2. **A hostile LLM reply.** Plans and edits are parsed, path-checked, and only
   ever written inside the worktree. A reply that asks to touch `/etc`, escape
   via `..`, or shell out cannot: edits are data, and commands are argv lists.
3. **An unauthorised MCP caller.** With no allowlist configured, every root is
   refused. Roots are matched on realpath, so a symlink or a sibling directory
   prefixed with an allowed name does not slip through.
4. **A lying verifier.** Status is not self-reported. Gates are derived from
   command results; a run with no isolation backend is marked unisolated and
   cannot be promoted to `verified`.

## Explicit non-goals / accepted risks

- **The isolation backend is trusted.** `bwrap`/`nsjail` are assumed sound; the
  harness only ensures one of them *actually starts* before relying on it. A
  kernel escape is out of scope.
- **The escape hatch is real.** `HARNESS_ALLOW_UNISOLATED=1` runs commands
  bare. It exists for development and for CI hosts without unprivileged user
  namespaces; results are flagged unisolated and can never be `verified`.
- **Learned models are advisory.** Risk, calibration, ranking, impact and
  flakiness models inform decisions and never gate a pass. They are trained
  from the ledger and are treated as untrusted when they are wrong.
- **Prompt-injection is not solved.** A repository can embed text that steers
  the model. The harness bounds the *consequences* (containment, isolation,
  argv-only, evidence-derived status); it does not claim to make the model
  immune to injection.
- **Secrets on disk are not exfiltrated-proof** while the network is up and a
  legitimate command reads them; the mitigation is the sandbox's network
  isolation, not per-file access control.

## Reporting

Security issues should be reported privately to the maintainers rather than
filed as public issues.
