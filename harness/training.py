"""Learn the harness's models from its own ledger (spec 5.3-5.8).

The ledger is the only supervision source: every task the orchestrator runs
records its decision, retrieval, test outcomes and final status.  This module
joins those entries into training sets, fits the models that were previously
present but untrained (risk GBDT + Platt calibration, test selector, reranker,
change-impact model), and adds a split-conformal calibrator so confidence is
reported with finite-sample coverage rather than a bare point score.

No model is trusted silently: :class:`LearnedModels` exposes ``metrics`` and
the orchestrator keeps the risk head in shadow until it is calibrated.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from .index import RepoIndex
from .models import (FEATURE_NAMES, ChangeImpactModel, ConformalCalibrator,
                     PatchCorrectnessModel, PlattScaler, Policy, TestSelector)
from .retrieval import Reranker, tokenize

# Enough local rows to trust a repo-specific fit; below this the pooled tier
# (other repos' examples, attached via libSQL) supplies the data instead.
MIN_LOCAL = 24
MIN_FIT = 8
# Fewer than this and there is not even a signal to calibrate against, so the
# risk head stays heuristic unless a pooled prior can carry it.
MIN_ROWS = 4


@dataclass
class LearnedModels:
    policy: Policy = field(default_factory=Policy)
    selector: TestSelector = field(default_factory=TestSelector)
    reranker: Reranker | None = None
    impact: ChangeImpactModel = field(default_factory=ChangeImpactModel)
    patch: PatchCorrectnessModel = field(default_factory=PatchCorrectnessModel)
    conformal: ConformalCalibrator = field(default_factory=ConformalCalibrator)
    metrics: dict = field(default_factory=dict)
    drift: dict = field(default_factory=dict)
    ood_threshold: float | None = None
    reference: dict | None = None
    # A learned head is shadow-only until a benchmark proves it beats the
    # deterministic policy; nothing in training sets this, only the benchmark.
    validated: bool = False

    @property
    def risk_trained(self) -> bool:
        return self.policy.risk_model is not None and self.policy.risk_model.trained

    @property
    def selector_trained(self) -> bool:
        return self.selector.model is not None and self.selector.model.trained

    @property
    def impact_trained(self) -> bool:
        return self.impact.model is not None and self.impact.model.trained

    @property
    def patch_trained(self) -> bool:
        return self.patch.model is not None and self.patch.model.trained


def read_ledger(root: str) -> list[dict]:
    from . import db
    store = db.Store(os.path.abspath(root))
    try:
        return store.read_ledger()
    finally:
        store.close()


def _by_task(entries: list[dict], component: str) -> dict[str, dict]:
    """Latest entry per task for a component (append-only, last wins)."""
    out: dict[str, dict] = {}
    for e in entries:
        if e.get("component") == component:
            out[e.get("task_id", "")] = e
    return out


def risk_examples(entries: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    feats = _by_task(entries, "task_features")
    orch = _by_task(entries, "orchestrator")
    X, y = [], []
    for tid, o in orch.items():
        f = feats.get(tid)
        if not f:
            continue
        vec = f.get("result", {}).get("feature_vector")
        if not vec:
            continue
        X.append(vec)
        y.append(0.0 if o.get("result", {}).get("success") else 1.0)
    if not X:
        return np.zeros((0, 0)), np.zeros(0)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def test_histories(entries: list[dict]) -> dict[str, list[bool]]:
    hist: dict[str, list[bool]] = {}
    for e in entries:
        if e.get("component") != "verifier":
            continue
        for o in e.get("result", {}).get("outcomes", []):
            tid = o.get("test_id")
            if tid:
                hist.setdefault(tid, []).append(o.get("status") == "passed")
    return hist


def _regression_outcomes(entries: list[dict]):
    """Per task, label each test outcome as regression evidence.

    A test is a positive example (worth selecting) when it *fails now but had
    not failed earlier in the ledger* -- i.e. the change is what broke it.  A
    test that was already red, or that passes, is labelled 0; a historical
    failure is not evidence that a change needs this test.  Only the latest
    verifier entry per task is used, matching the per-task join elsewhere.

    Returns ``(by_task, hist)`` where ``by_task[task_id]`` is a list of
    ``(test_id, label, duration_s)`` and ``hist`` is the ordered pass/fail
    history per test.
    """
    ranks = _by_task(entries, "test_selector")
    last: dict[str, int] = {}
    for i, e in enumerate(entries):
        if e.get("component") == "verifier":
            last[e.get("task_id", "")] = i
    failed: dict[str, bool] = {}
    hist: dict[str, list[bool]] = {}
    by_task: dict[str, list] = {}
    for i, e in enumerate(entries):
        tid = e.get("task_id", "")
        if e.get("component") != "verifier" or last.get(tid) != i or tid not in ranks:
            continue
        items = []
        for o in e.get("result", {}).get("outcomes", []):
            path = o.get("test_id", "")
            passed = o.get("status") == "passed"
            label = 1 if (not passed and not failed.get(path, False)) else 0
            items.append((path, label, passed, o.get("duration_s", 1.0) or 1.0))
        by_task[tid] = items
        for path, _label, passed, _dur in items:
            hist.setdefault(path, []).append(passed)
            if not passed:
                failed[path] = True
    return by_task, hist


def selector_examples(entries: list[dict]):
    ranks = _by_task(entries, "test_selector")
    outcomes, hist = _regression_outcomes(entries)
    out = []
    for tid, r in ranks.items():
        symbols = set(s.lower() for s in r.get("result", {}).get("changed_symbols", []))
        if not symbols or tid not in outcomes:
            continue
        for path, label, _passed, duration in outcomes[tid]:
            out.append((symbols, set(tokenize(path)), path, duration, hist.get(path, []), label))
    return out


def reranker_examples(entries: list[dict], index: RepoIndex):
    retr = _by_task(entries, "retriever")
    orch = _by_task(entries, "orchestrator")
    by_key = {f"{c.path}:{c.start_line}": c for c in index.chunks}
    out = []
    for tid, r in retr.items():
        changed = set(orch.get(tid, {}).get("result", {}).get("files_changed", []))
        if not changed:
            continue
        query = r.get("result", {}).get("query", "")
        for h in r.get("result", {}).get("candidates", []):
            chunk = by_key.get(f"{h.get('path')}:{h.get('start_line')}")
            if chunk is None:
                continue
            out.append((query, chunk, 1.0 if h.get("path") in changed else 0.0))
    return out


def impact_examples(entries: list[dict], index: RepoIndex):
    ranks = _by_task(entries, "test_selector")
    outcomes, _ = _regression_outcomes(entries)
    out = []
    for tid, r in ranks.items():
        symbols = r.get("result", {}).get("changed_symbols", [])
        if not symbols or tid not in outcomes:
            continue
        for sym in symbols:
            fanout = len(index.callers_of(sym))
            for path, label, _passed, _dur in outcomes[tid]:
                out.append((sym, path, set(tokenize(path)), fanout, float(label)))
    return out


def patch_examples(entries: list[dict]):
    """(X, y) for the patch-correctness head.

    Label 1 = the task's patch has regression evidence: at least one test that
    now fails but had not failed earlier in the ledger (``_regression_outcomes``).
    Label 0 = no such evidence.  This is the honest proxy the oracle ("tests
    passed") cannot give: a patch that passes its tests but breaks others is
    exactly the overfitting case.  Kept distinct from the risk head, which
    predicts task failure from *task* features rather than a bad patch from
    *patch* features.
    """
    feats = _by_task(entries, "patch_features")
    outcomes, _ = _regression_outcomes(entries)
    X, y = [], []
    for tid, f in feats.items():
        vec = f.get("result", {}).get("feature_vector")
        if not vec or tid not in outcomes:
            continue
        X.append(vec)
        y.append(1.0 if any(lbl for _p, lbl, _passed, _dur in outcomes[tid]) else 0.0)
    if not X:
        return np.zeros((0, 0)), np.zeros(0)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def _brier(model, X, y) -> float:
    p = np.asarray(model.predict(X), dtype=float).ravel()
    return float(np.mean((p - np.asarray(y, dtype=float)) ** 2))


def rule_risk(X) -> np.ndarray:
    """The deterministic policy's risk score, from a feature matrix.

    Mirrors :meth:`Policy.decide` exactly, so a learned head can be compared
    against the rule it is meant to replace on the same held-out rows.
    """
    X = np.asarray(X, dtype=float)
    col = {name: i for i, name in enumerate(FEATURE_NAMES)}
    risk = (0.2
            + 0.3 * X[:, col["security_sensitive"]]
            + 0.3 * X[:, col["public_api"]]
            + 0.1 * np.minimum(X[:, col["changed_files"]], 3)
            + 0.05 * X[:, col["dep_fanout"]])
    return np.clip(risk, 0.0, 0.95)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson score interval for a proportion; a metric without an interval is
    a number without error bars (spec: report n and CI, not a bare point)."""
    if n <= 0:
        return [0.0, 1.0]
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _fit_risk(X, y, store, repo_id: str):
    """Fit the risk head with partial pooling and a champion/challenger gate.

    A repo-specific GBDT is only fit when it has enough local rows; when the
    ledger is thin a model fit on the pooled cross-repo examples is used
    instead.  When both exist, the one with the lower out-of-sample Brier score
    on a time-ordered holdout wins.  The repo's own rows are written to the
    shared tier either way, so the pool grows with every task."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    info: dict = {"n_local": int(n), "source": "none", "gate": "n/a"}
    if store is not None and n:
        store.save_examples(repo_id, "risk", X, y)

    local_ok = n >= MIN_FIT and len(set(y.tolist())) > 1
    prior_X, prior_y = (store.load_examples("risk", exclude_repo=repo_id)
                        if store is not None else (np.zeros((0, 0)), np.zeros(0)))
    prior_ok = len(prior_y) >= MIN_FIT and len(set(prior_y.tolist())) > 1
    info["n_prior"] = int(len(prior_y))

    candidates: list[tuple[str, object]] = []
    if local_ok:
        candidates.append(("local", Policy().train_risk(X.tolist(), y.tolist()).risk_model))
    if local_ok and prior_ok:
        augX = np.vstack([X, prior_X])
        augY = np.concatenate([y, prior_y])
        candidates.append(("pooled", Policy().train_risk(augX.tolist(), augY.tolist()).risk_model))
    elif not local_ok and prior_ok:
        candidates.append(("pooled", Policy().train_risk(prior_X.tolist(), prior_y.tolist()).risk_model))
    if not candidates:
        return None, info

    chosen = candidates[0]
    if len(candidates) > 1 and n >= MIN_LOCAL:
        cut = max(MIN_FIT, int(n * 0.8))
        hX, hy = X[cut:], y[cut:]
        if len(hy) >= 3 and len(set(hy.tolist())) > 1:
            scored = sorted(candidates, key=lambda c: _brier(c[1], hX, hy))
            chosen = scored[0]
            info["gate"] = "holdout_brier"
            info["brier"] = {name: round(_brier(m, hX, hy), 4) for name, m in candidates}
        else:
            chosen = next((c for c in candidates if c[0] == "local"), candidates[0])
    info["source"] = chosen[0]
    return chosen[1], info


def _flaky(hist: list[bool]) -> bool:
    """A test that has both passed and failed across the ledger is a flaky-test
    candidate.  Kept as a signal for future label cleaning; it is deliberately
    not applied to the examples, where a mixed outcome can be the very signal a
    changed symbol would predict."""
    return len(set(hist)) > 1


@dataclass
class RiskSplit:
    """Time-ordered train / calibration / test slices of the risk examples.

    ``mode`` is ``"temporal"`` when there was enough history to hold the
    calibration slice out of training, and ``"in_sample"`` when the ledger is
    too thin to split -- in which case metrics are honestly marked as such
    rather than presented as held-out.
    """
    X_train: np.ndarray
    y_train: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    mode: str


def temporal_split(X, y, train_frac: float = 0.6, cal_frac: float = 0.2) -> RiskSplit:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 20:  # too few to hold anything out meaningfully
        return RiskSplit(X, y, X, y, X[:0], y[:0], "in_sample")
    train = max(MIN_FIT, int(n * train_frac))
    cal = max(train + 1, int(n * (train_frac + cal_frac)))
    return RiskSplit(X[:train], y[:train], X[train:cal], y[train:cal],
                     X[cal:], y[cal:], "temporal")


def _repo_holdout(model, store, repo_id: str) -> dict:
    """Cross-repo holdout: score the fitted head on other repos' examples only,
    so generalisation beyond the repo it was trained on is visible."""
    try:
        prior_X, prior_y = store.load_examples("risk", exclude_repo=repo_id)
    except Exception:
        return {"n": 0}
    n = int(len(prior_y))
    if n < MIN_FIT:
        return {"n": n}
    raw = np.asarray(model.predict(prior_X), dtype=float).ravel()
    return {"n": n, "brier": round(float(np.mean((raw - prior_y) ** 2)), 4)}


def train_from_ledger(root: str, index: RepoIndex | None = None, store=None) -> LearnedModels:
    root = os.path.abspath(root)
    entries = read_ledger(root)
    models = LearnedModels()
    if store is None:
        from . import db
        store = db.Store(root)
    repo_id = os.path.basename(root.rstrip("/")) or "repo"
    if store is not None:
        try:  # keep the per-repo id stable irrespective of training path
            repo_id = store.get_meta("repo_id") or repo_id
            store.set_meta("repo_id", repo_id)
        except Exception:
            pass

    X, y = risk_examples(entries)
    if len(y) >= MIN_ROWS:
        split = temporal_split(X, y)
        model, info = _fit_risk(split.X_train, split.y_train, store, repo_id)
        if model is not None:
            policy = Policy()
            policy.risk_model = model
            raw_cal = np.asarray(model.predict(split.X_cal), dtype=float).ravel()
            # Calibrate on the held-out slice, never on the rows the GBDT fit
            # on: an in-sample Platt curve reports confidence it has not earned.
            if len(split.y_cal) >= 2 and len(set(split.y_cal.tolist())) > 1:
                policy.calibrator = PlattScaler().fit(raw_cal, split.y_cal)
            matrix = split.X_train if len(split.X_train) >= MIN_FIT else None
            conformal = ConformalCalibrator().fit(raw_cal, split.y_cal, matrix=matrix)
            models.policy = policy
            models.conformal = conformal
            hits = sum(1 for r, t in zip(raw_cal, split.y_cal)
                       if int(t > 0.5) in conformal.predict_set(r))
            risk = {
                "n": int(len(y)), "positive_rate": float(y.mean()),
                "positive_rate_ci": wilson_ci(int(y.sum()), len(y)),
                "split": split.mode,
                "n_train": int(len(split.y_train)),
                "n_cal": int(len(split.y_cal)),
                "n_test": int(len(split.y_test)),
                "ece": round(conformal.base.ece([conformal.base(float(r)) for r in raw_cal],
                                                split.y_cal), 4),
                "conformal_coverage": round(conformal.coverage(raw_cal, split.y_cal), 4),
                "conformal_coverage_ci": wilson_ci(hits, len(split.y_cal)),
                "alpha": conformal.alpha, "pooling": info,
            }
            if len(split.y_test):
                raw_test = np.asarray(model.predict(split.X_test), dtype=float).ravel()
                scored = (np.asarray([policy.calibrator(float(r)) for r in raw_test],
                                     dtype=float)
                          if policy.calibrator is not None else raw_test)
                risk["test_brier"] = round(float(np.mean((scored - split.y_test) ** 2)), 4)
                risk["rule_brier"] = round(float(np.mean(
                    (rule_risk(split.X_test) - split.y_test) ** 2)), 4)
                risk["test_coverage"] = round(conformal.coverage(raw_test, split.y_test), 4)
                risk["test_positive_rate"] = round(float(split.y_test.mean()), 4)
            if store is not None:
                risk["repo_holdout"] = _repo_holdout(model, store, repo_id)
            models.metrics["risk"] = risk
            models.reference = {"n": int(len(y)), "positive_rate": float(y.mean())}
            if matrix is not None:
                d = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2)
                models.ood_threshold = float(np.percentile(d, 95))
    sel = selector_examples(entries)
    if len(sel) >= MIN_FIT and len({e[-1] for e in sel}) > 1:
        models.selector = TestSelector().train(sel)
        models.metrics["selector"] = {"n": len(sel),
                                      "positive_rate": round(float(np.mean([e[-1] for e in sel])), 4)}
    if index is None:
        index = RepoIndex.build(root)
    rr = reranker_examples(entries, index)
    if len(rr) >= MIN_FIT and len({e[2] for e in rr}) > 1:
        models.reranker = Reranker.train(rr)
        models.metrics["reranker"] = {"n": len(rr),
                                      "positive_rate": round(float(np.mean([e[2] for e in rr])), 4)}
    im = impact_examples(entries, index)
    if len(im) >= MIN_FIT and len({e[-1] for e in im}) > 1:
        models.impact = ChangeImpactModel().train(im)
        models.metrics["impact"] = {"n": len(im),
                                    "positive_rate": round(float(np.mean([e[-1] for e in im])), 4)}
    px, py = patch_examples(entries)
    if len(py) >= MIN_FIT and len(set(py.tolist())) > 1:
        models.patch = PatchCorrectnessModel().train(px, py)
        models.metrics["patch"] = {"n": len(py),
                                   "positive_rate": round(float(py.mean()), 4)}
    # Drift/OOD monitoring over a time-ordered split of the ledger (spec 5.8, 8.3).
    from .drift import drift_report
    models.drift = drift_report(models, entries, feature_names=FEATURE_NAMES).to_dict()
    models.metrics["ledger_n"] = len(entries)
    if store is not None:
        try:
            store.save_model("learned", "learned-v1", models, models.metrics,
                             revision=index.revision)
        except Exception:
            pass
    return models


PROMOTION_KEY = "validation:learned"


def validate_and_promote(root: str, min_test: int = 20,
                         tolerance: float = 0.1) -> dict:
    """Gate the learned risk head on evidence, and record the verdict.

    Promotion requires a temporal test slice with enough rows and both classes,
    a lower Brier score than the deterministic rule it shadows, and conformal
    coverage within ``tolerance`` of the target.  The verdict is stored in the
    ``meta`` table keyed by (revision, ledger_n), so it applies to exactly the
    artifact that was validated and is lost when the ledger or tree moves on.
    """
    from . import db
    root = os.path.abspath(root)
    index = RepoIndex.build(root)
    store = db.Store(root)
    try:
        models = train_from_ledger(root, index=index, store=store)
        risk = models.metrics.get("risk", {})
        n_test = int(risk.get("n_test", 0))
        positive = risk.get("test_positive_rate")
        learned = risk.get("test_brier")
        rule = risk.get("rule_brier")
        coverage = risk.get("test_coverage")
        alpha = float(risk.get("alpha", 0.1))
        reasons: list[str] = []
        if risk.get("split") != "temporal":
            reasons.append("insufficient_history")
        if n_test < min_test:
            reasons.append(f"test_n_below_{min_test}")
        if positive in (None, 0.0, 1.0):
            reasons.append("test_needs_both_classes")
        if learned is None or rule is None or learned >= rule:
            reasons.append("learned_brier_not_better_than_rule")
        if coverage is None or coverage < 1.0 - alpha - tolerance:
            reasons.append("coverage_below_target")
        promoted = not reasons
        evidence = {
            "revision": index.revision,
            "ledger_n": int(models.metrics.get("ledger_n", 0)),
            "n_test": n_test, "test_positive_rate": positive,
            "test_brier": learned, "rule_brier": rule,
            "test_coverage": coverage, "alpha": alpha,
            "min_test": min_test, "tolerance": tolerance,
            "promoted": promoted, "reasons": reasons,
        }
        store.set_meta(PROMOTION_KEY, evidence)
        if promoted:
            models.validated = True
            store.save_model("learned", "learned-v1", models, models.metrics,
                             revision=index.revision)
        return {"root": root, **evidence, "metrics": models.metrics}
    finally:
        store.close()
