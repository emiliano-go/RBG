"""The orchestrator: the task state machine from spec section 6.

It owns tool permissions, budgets, and routing.  The single main coding LLM
(the only generative component) implements :class:`MainLLM`; it can request
edits and tools but never touches the filesystem or shell directly.  All of
that is mediated here through the sandbox, and success is declared only from
observed command results (spec 3.2, 6.11).
"""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Protocol

from . import db
from .index import RepoIndex
from .models import Policy, TestSelector, TaskFeatures, extract_features, select_action
from .paths import resolve_within
from .plan import Plan, audit_completion, format_guidance, parse_plan, scope_drift
from .retrieval import HybridRetriever, Hit, Reranker
from .status import Gates, TerminalState, derive_status, gate_state
from .training import LearnedModels, test_histories
from .verify import Ledger, Sandbox, TestOutcome, Verifier
from .workspace import Workspace


@dataclass
class EditOp:
    path: str
    content: str


class MainLLM(Protocol):
    """The one generative model.  Implements planning and code editing given a
    bounded, source-grounded context bundle."""

    def plan(self, task: str, context: list[Hit]) -> str: ...

    def edit(self, task: str, plan: str, context: list[Hit],
             diagnostics: str = "") -> list[EditOp]: ...


@dataclass
class Report:
    task_id: str
    success: bool
    status: str
    files_changed: list[str] = field(default_factory=list)
    checks_run: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    scope_reductions: list[str] = field(default_factory=list)
    decision: dict = field(default_factory=dict)
    context_used: int = 0
    iterations: int = 0
    actions: list[str] = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    drift: list[dict] = field(default_factory=list)
    completeness: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "success": self.success, "status": self.status,
            "files_changed": self.files_changed, "checks_run": self.checks_run,
            "failures": self.failures, "unverified": self.unverified,
            "scope_reductions": self.scope_reductions, "decision": self.decision,
            "context_used": self.context_used, "iterations": self.iterations,
            "actions": self.actions, "plan": self.plan, "drift": self.drift,
            "completeness": self.completeness,
        }


class Orchestrator:
    def __init__(self, root: str, llm: MainLLM, *,
                 policy: Policy | None = None, selector: TestSelector | None = None,
                 learned: LearnedModels | None = None, use_reranker: bool = False,
                 max_iterations: int = 3, context_limit: int = 15) -> None:
        self.root = os.path.abspath(root)
        self.llm = llm
        self.learned = learned
        self.policy = (learned.policy if learned and learned.risk_trained
                       else policy or Policy())
        self.selector = (learned.selector if learned and learned.selector_trained
                         else selector or TestSelector())
        self.use_reranker = use_reranker or bool(learned and learned.reranker)
        self.max_iterations = max_iterations
        self.context_limit = context_limit
        self.sandbox = Sandbox(self.root, timeout_s=90)
        self.verifier = Verifier(self.sandbox)
        self.index: RepoIndex | None = None
        self.retriever: HybridRetriever | None = None
        self.repo_id = hashlib.sha256(self.root.encode()).hexdigest()[:12]
        self.ledger = Ledger(db.Store(self.root))
        self._history = test_histories(self.ledger.store.read_ledger())

    def _retriever(self) -> HybridRetriever:
        reranker = self.learned.reranker if self.learned else None
        return HybridRetriever(self.index, reranker=reranker if self.use_reranker else None)

    # -- helpers ----------------------------------------------------------- #
    def _log(self, component: str, result: dict, *, model_name: str, task_id: str,
             reasons: list[str] | None = None, **kw) -> dict:
        return self.ledger.record(
            component=component, task_id=task_id, repo_id=self.repo_id,
            revision=self.index.revision if self.index else "", model_name=model_name,
            result=result, reason_codes=reasons, **kw)

    @staticmethod
    def _apply_edits(edits: list[EditOp], root: str) -> dict[str, tuple[str, str]]:
        """Write edits, returning path -> (before, after) for the diff."""
        diffs: dict[str, tuple[str, str]] = {}
        for e in edits:
            full = resolve_within(root, e.path)
            before = ""
            if os.path.exists(full):
                with open(full, encoding="utf-8", errors="replace") as fh:
                    before = fh.read()
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(e.content)
            diffs[e.path] = (before, e.content)
        return diffs

    @staticmethod
    def _changed_symbols(index: RepoIndex, paths: list[str]) -> list[str]:
        syms: list[str] = []
        for c in index.chunks:
            if c.path in paths and c.symbol:
                syms.append(c.symbol)
        return sorted(set(syms))

    @staticmethod
    def _diff_lines(diffs: dict[str, tuple[str, str]]) -> int:
        """Added + removed lines across the applied edits (spec 5.3 changed-line
        feature: a real signal, not the 0 placeholder intake alone can supply)."""
        changed = 0
        for before, after in diffs.values():
            for line in difflib.unified_diff(before.splitlines(), after.splitlines(),
                                             lineterm=""):
                if line[:1] in "+-" and not line.startswith(("+++", "---")):
                    changed += 1
        return changed

    def _assess(self, feats: TaskFeatures, report: Report, task_id: str):
        """Score risk/uncertainty for a feature vector, log it, and return
        ``(decision, uncertainty, ood)``.  Called at intake and again once the
        real diff is known, so the risk head sees changed-file/line/fanout."""
        uncertainty, ood = 0.0, False
        if self.learned and self.learned.risk_trained and self.learned.conformal.fitted:
            raw = float(self.learned.policy.risk_model.predict(feats.vector())[0])
            uncertainty = 1.0 if len(self.learned.conformal.predict_set(raw)) > 1 else 0.0
            if self.learned.ood_threshold is not None:
                ood = self.learned.conformal.ood_score(feats.vector()) > self.learned.ood_threshold
                if ood:
                    uncertainty = 1.0  # spec 3.2/5.3: OOD input -> conservative fallback
        decision = self.policy.decide(feats, uncertainty=uncertainty,
                                      use_learned=bool(self.learned and self.learned.risk_trained))
        if ood:
            decision.reason_codes.append("ood_input")
            decision.fallback = "deterministic_high_coverage"
        report.decision = decision.to_dict()
        self._log("policy", decision.to_dict(), model_name=self.policy.version,
                  task_id=task_id, reasons=decision.reason_codes)
        self._log("task_features", {"values": feats.values, "task_class": feats.task_class,
                                    "feature_vector": feats.vector().tolist()},
                  model_name="extract_features-v1", task_id=task_id)
        return decision, uncertainty, ood

    def _discover_tests(self) -> list[dict]:
        if not self.verifier.toolchain().is_python:
            return []
        res = self.sandbox.run([self.sandbox.python, "-m", "pytest", "--collect-only", "-q"])
        tests: list[dict] = []
        for line in res.stdout.splitlines():
            line = line.strip()
            if "::" in line and not line.startswith("<"):
                tests.append({"test_id": line, "duration": 1.0, "history": []})
        if not tests:
            for path in self.index.files:
                if re.search(r"test_.*\.py$|_test\.py$", path):
                    tests.append({"test_id": path, "duration": 1.0, "history": []})
        return tests

    # -- main entry -------------------------------------------------------- #
    def _rerun_flaky_failures(self, failed: list[TestOutcome], outcomes: list[TestOutcome],
                              report: Report, task_id: str) -> list[TestOutcome]:
        """Execute the ``rerun_flaky_failure`` action (spec 5.7): a failure with
        a prior flip history is retried once; if it passes, it is a flake, not
        a regression, and is dropped from the failure set with evidence."""
        remaining: list[TestOutcome] = []
        for o in failed:
            flake = self.selector.flaky.predict(self._history.get(o.test_id, []))
            if flake > 0.0:
                res, outs = self.verifier.run_tests([o.test_id])
                passed = any(x.test_id == o.test_id and x.status == "passed" for x in outs)
                self._log("verifier", {"name": "flaky_rerun", "test_id": o.test_id,
                                       "flake_score": round(flake, 4), "recovered": passed,
                                       **res.to_dict()},
                          model_name="flaky-rerun-v1", task_id=task_id,
                          reasons=["flaky_rerun_executed"])
                if passed:
                    report.scope_reductions.append(f"flaky_rerun_recovered:{o.test_id}")
                    continue
            remaining.append(o)
        return remaining

    def run(self, task: str, task_id: str | None = None) -> Report:
        """Run the loop against a disposable copy of the repo when possible, so
        a failed attempt never mutates the caller's checkout.  Without an
        explicit task_id every run gets its own, so ledger entries never collide.
        """
        task_id = task_id or f"task-{uuid.uuid4().hex[:12]}"
        workspace = Workspace(self.root)
        work = workspace.setup()
        try:
            return self._run(task, task_id, work)
        finally:
            workspace.cleanup()

    def _run(self, task: str, task_id: str, work: str) -> Report:
        if work != self.root:
            self.sandbox = Sandbox(work, timeout_s=90)
            self.verifier = Verifier(self.sandbox)
        # 2. repository snapshot
        self.index = RepoIndex.build(work)
        self.retriever = self._retriever()
        report = Report(task_id=task_id, success=False, status="running")

        # 3. initial assessment (task text only: no diff exists yet)
        feats: TaskFeatures = extract_features(task)
        decision, uncertainty, ood = self._assess(feats, report, task_id)
        if not decision.verification["require_full_suite"]:
            report.scope_reductions.append("full_suite_not_required_by_policy")

        # 4. context assembly
        hits = self.retriever.retrieve(task, limit=decision.retrieval["candidate_limit"])
        if decision.retrieval.get("expand"):
            hits = self.retriever.expand(hits)
        hits = hits[:self.context_limit]
        if self.use_reranker:
            hits = Reranker().rerank(task, hits, self.index.chunks)[:self.context_limit]
        report.context_used = len(hits)
        self._log("retriever", {"query": task, "candidates": [h.to_dict() for h in hits]},
                  model_name="hybrid-rrf-v1", task_id=task_id)

        # 5-8. plan, edit, cheap verify, repair loop.  Every step is checked
        # against the parsed plan, so a missed step is fed back to the next
        # iteration instead of being shipped silently.
        plan_text = self.llm.plan(task, hits)
        typed: Plan = parse_plan(plan_text, self.index.files)
        report.plan = typed.to_dict()
        self._log("plan", typed.to_dict(), model_name="plan-parse-v1", task_id=task_id)
        covered = set(typed.all_files())
        uncovered = sorted({h.path for h in hits if h.path not in covered})
        report.drift.append({"stage": "plan", "retrieval_uncovered": uncovered})
        diagnostics = ""
        diffs: dict[str, tuple[str, str]] = {}   # cumulative, original -> latest
        drift = None
        for iteration in range(1, self.max_iterations + 1):
            report.iterations = iteration
            edits = self.llm.edit(task, plan_text, hits, diagnostics)
            if not edits:
                report.failures.append("llm_produced_no_edits")
                break
            for path, (before, after) in self._apply_edits(edits, work).items():
                # keep the pre-loop original so the diff is the whole change, not
                # just the last iteration's delta for repeatedly-edited files
                diffs[path] = (diffs[path][0] if path in diffs else before, after)
            changed = list(diffs)
            report.files_changed = changed
            self.index = RepoIndex.build(work)
            self.retriever = self._retriever()

            check = self.verifier.compile_check(changed)
            if check is not None:
                report.checks_run.append({"name": "compile", **check.to_dict()})
                self._log("verifier", {"name": "compile", **check.to_dict()},
                          model_name="py_compile", task_id=task_id,
                          latency_ms=check.duration_ms)
            drift = scope_drift(typed, changed)
            report.drift.append({"stage": "step", "iteration": iteration, **drift.to_dict()})
            self._log("plan_drift", {"iteration": iteration, **drift.to_dict()},
                      model_name="scope-drift-v1", task_id=task_id,
                      reasons=["missing_steps"] if drift.missing else [])
            if (check is None or check.ok) and drift.ok():
                break
            parts = []
            if check is not None and not check.ok:
                parts.append(check.stderr or check.stdout)
                report.failures = [f"compile: iteration {iteration}"]
            parts.append(format_guidance(typed, drift))
            diagnostics = "\n".join(p for p in parts if p)
        if drift is not None and drift.missing:
            report.scope_reductions.append("unmet_plan_steps:" + ",".join(drift.missing))

        if not diffs:
            report.status = "failed"
            self._log("orchestrator", report.to_dict(), model_name="orchestrator-v1", task_id=task_id)
            return report

        # 3b. re-assess with the real diff: changed-file/line/fan-out features.
        # The intake decision could only see the request text; verification
        # breadth has to reflect what actually changed (spec 5.3, 6.9).
        changed_symbols = self._changed_symbols(self.index, report.files_changed)
        feats = extract_features(
            task, changed_files=report.files_changed,
            changed_lines=self._diff_lines(diffs),
            dep_fanout=sum(len(self.index.callers_of(s)) for s in changed_symbols))
        decision, uncertainty, ood = self._assess(feats, report, task_id)

        # cheap type gate (spec 4 P0 / 5.3): run only when a checker exists, so
        # "verified" is never reported for a check that could not run.
        if self.verifier.typecheck_available():
            py = [p for p in report.files_changed if p.endswith(".py")]
            tc = self.verifier.typecheck(py)
            report.checks_run.append({"name": "typecheck", **tc.to_dict()})
            self._log("verifier", {"name": "typecheck", **tc.to_dict()},
                      model_name="mypy", task_id=task_id, latency_ms=tc.duration_ms)

        # 9. risk-based, test-selection-ordered verification
        tests = self._discover_tests()
        ranked = self.selector.rank(tests, changed_symbols)
        if ranked:
            self._log("test_selector", {"ranking": [r.to_dict() for r in ranked],
                                        "changed_symbols": changed_symbols},
                      model_name=self.selector.version, task_id=task_id)
            report.scope_reductions.append("tests_reordered_not_skipped")
        selected = [r.test_id for r in ranked] if ranked else None

        test_res, outcomes = self.verifier.run_tests(selected)
        report.checks_run.append({"name": "tests", **test_res.to_dict(),
                                  "outcomes": [o.__dict__ for o in outcomes]})
        self._log("verifier", {"name": "tests", **test_res.to_dict(),
                               "outcomes": [o.__dict__ for o in outcomes]},
                  model_name="pytest", task_id=task_id, latency_ms=test_res.duration_ms)

        failed = [o for o in outcomes if o.status in ("failed", "error")]
        final_res = test_res
        if failed and any(self.selector.flaky.predict(self._history.get(o.test_id, [])) > 0.0
                          for o in failed):
            report.actions.append("rerun_flaky_failure")
        failed = self._rerun_flaky_failures(failed, outcomes, report, task_id)

        # 9b. choose the next step from the bounded action menu (spec 5.3)
        action, action_reasons = select_action(
            has_failures=bool(failed) or not final_res.ok, full_suite_ran=False,
            full_suite_required=decision.verification["require_full_suite"],
            uncertainty=uncertainty > 0.5, ood=ood)
        if action == "run_full_suite":
            full_res, full_out = self.verifier.run_tests(None)
            report.checks_run.append({"name": "full_suite", **full_res.to_dict()})
            final_res = full_res
            full_failed = [o for o in full_out if o.status in ("failed", "error")]
            if full_failed:
                failed = full_failed
            action, action_reasons = select_action(
                has_failures=bool(failed) or not full_res.ok, full_suite_ran=True,
                full_suite_required=decision.verification["require_full_suite"])
        report.actions.append(action)
        self._log("action", {"action": action, "reason_codes": action_reasons},
                  model_name="action-rules-v1", task_id=task_id, reasons=action_reasons)

        # 9c. completeness audit: callers of the changed symbols that were
        # neither edited nor retrieved (advisory; spec 5.5 / anti-miss).
        report.completeness = audit_completion(
            self.index, changed_symbols, report.files_changed, [h.path for h in hits])
        self._log("completeness", report.completeness, model_name="completeness-v1",
                  task_id=task_id,
                  reasons=["affected_not_edited"] if report.completeness["affected_not_edited"] else [])

        # 10-11. final status, derived only from observed evidence (spec 3.2,
        # 6.11): a gate that never ran is unverified, not a pass; tests that
        # collected nothing do not count as passing; and an unisolated sandbox
        # can never yield "verified".
        recovered = any(s.startswith("flaky_rerun_recovered") for s in report.scope_reductions)
        report.failures = [f"{o.test_id}: {o.status}" for o in failed]
        compile_gate = gate_state(report.checks_run, "compile")
        typecheck_gate = gate_state(report.checks_run, "typecheck")
        if final_res.exit_code == 5:
            tests_gate: bool | None = None
            report.scope_reductions.append("no_tests_collected")
            report.unverified.append("tests")
        else:
            tests_gate = bool(not failed and (final_res.ok or recovered))
        if typecheck_gate is None:
            report.unverified.append("typecheck")
        state = derive_status(Gates(compile=compile_gate, typecheck=typecheck_gate,
                                    tests=tests_gate,
                                    isolated=self.sandbox.isolation().isolated))
        report.status = state.value
        report.success = state is TerminalState.VERIFIED

        self._log("orchestrator", report.to_dict(), model_name="orchestrator-v1",
                  task_id=task_id, reasons=decision.reason_codes)
        return report
