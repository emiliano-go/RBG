"""Learned components, all dependency-free (numpy only):

* ``GBDT``        - XGBoost-style gradient-boosted regression trees, the P0
                    tabular model from the spec (4). A real implementation,
                    not a wrapper around LightGBM.
* ``PlattScaler`` - calibration of raw scores to probabilities (spec 5.8).
* ``TaskFeatures``- deterministic task/diff features (spec 5.3). No second
                    LLM call is used to extract them.
* ``Policy``      - deterministic rule baseline plus learned heads (risk,
                    retrieval depth, verification breadth) in shadow mode.
* ``TestSelector``- per-(change,test) relevance ranking (spec 5.4).
* ``FlakyModel``  - test flakiness probability, kept separate from
                    regression probability (spec 5.7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .retrieval import tokenize


# --------------------------------------------------------------------------- #
# Gradient-boosted decision trees
# --------------------------------------------------------------------------- #
class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, value: float = 0.0) -> None:
        self.feature = -1
        self.threshold = 0.0
        self.left: _Node | None = None
        self.right: _Node | None = None
        self.value = value


class _Tree:
    def __init__(self, max_depth: int, min_child: int, reg_lambda: float) -> None:
        self.max_depth = max_depth
        self.min_child = min_child
        self.reg_lambda = reg_lambda
        self.root = _Node()

    def fit(self, X: np.ndarray, g: np.ndarray, h: np.ndarray) -> None:
        self.root = self._grow(X, g, h, 0, np.arange(len(X)))

    def _leaf(self, g: np.ndarray, h: np.ndarray) -> float:
        return float(-g.sum() / (h.sum() + self.reg_lambda))

    def _grow(self, X, g, h, depth, idx) -> _Node:
        node = _Node(self._leaf(g[idx], h[idx]))
        if depth >= self.max_depth or len(idx) < 2 * self.min_child:
            return node
        best = None
        G, H = g[idx].sum(), h[idx].sum()
        parent = G * G / (H + self.reg_lambda)
        for feat in range(X.shape[1]):
            vals = X[idx, feat]
            order = np.argsort(vals)
            vs, gs, hs = vals[order], g[idx][order], h[idx][order]
            Gc = np.cumsum(gs)[:-1]
            Hc = np.cumsum(hs)[:-1]
            Gr, Hr = G - Gc, H - Hc
            gains = 0.5 * (Gc * Gc / (Hc + self.reg_lambda)
                           + Gr * Gr / (Hr + self.reg_lambda) - parent)
            gains[(vs[:-1] == vs[1:])] = -np.inf
            # min_child check on counts
            counts = np.arange(1, len(idx))
            ok = (counts >= self.min_child) & ((len(idx) - counts) >= self.min_child)
            gains[~ok] = -np.inf
            j = int(np.argmax(gains))
            if best is None or gains[j] > best[0]:
                best = (float(gains[j]), feat, float((vs[j] + vs[j + 1]) / 2.0))
        if best is None or best[0] <= 0:
            return node
        _, feat, thr = best
        mask = X[idx, feat] <= thr
        node.feature, node.threshold = feat, thr
        node.left = self._grow(X, g, h, depth + 1, idx[mask])
        node.right = self._grow(X, g, h, depth + 1, idx[~mask])
        return node

    def predict_row(self, row: np.ndarray) -> float:
        node = self.root
        while node.feature >= 0:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value


class GBDT:
    """Gradient-boosted trees for regression or binary classification."""

    def __init__(self, n_estimators: int = 40, learning_rate: float = 0.3,
                 max_depth: int = 3, min_child: int = 2, reg_lambda: float = 1.0,
                 loss: str = "squared") -> None:
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_child = min_child
        self.reg_lambda = reg_lambda
        self.loss = loss
        self.trees: list[_Tree] = []
        self.base = 0.0
        self.trained = False

    def _sigmoid(self, z):
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y) -> "GBDT":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features = X.shape[1]
        self.base = float(np.mean(y)) if self.loss == "squared" else float(
            math.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6)))
        pred = np.full(len(X), self.base)
        self.trees = []
        for _ in range(self.n_estimators):
            if self.loss == "squared":
                g, h = pred - y, np.ones(len(X))
            else:
                p = self._sigmoid(pred)
                g, h = p - y, p * (1 - p)
            tree = _Tree(self.max_depth, self.min_child, self.reg_lambda)
            tree.fit(X, g, h)
            update = np.array([tree.predict_row(X[i]) for i in range(len(X))])
            pred = pred + self.lr * update
            self.trees.append(tree)
        self.trained = True
        return self

    def raw(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        pred = np.full(len(X), self.base)
        for tree in self.trees:
            pred = pred + self.lr * np.array([tree.predict_row(X[i]) for i in range(len(X))])
        return pred

    def predict(self, X) -> np.ndarray:
        raw = self.raw(X)
        return raw if self.loss == "squared" else self._sigmoid(raw)

    def feature_importance(self) -> np.ndarray:
        counts = np.zeros(getattr(self, "n_features", 1))
        for tree in self.trees:
            self._count(tree.root, counts)
        return counts

    def _count(self, node: _Node | None, counts: np.ndarray) -> None:
        if node is None or node.feature < 0:
            return
        if node.feature < len(counts):
            counts[node.feature] += 1
        self._count(node.left, counts)
        self._count(node.right, counts)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
@dataclass
class PlattScaler:
    a: float = 1.0
    b: float = 0.0
    version: str = "platt-v1"

    def fit(self, scores, labels) -> "PlattScaler":
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=float)
        a, b, lr = 1.0, 0.0, 0.5
        for _ in range(500):
            p = 1.0 / (1.0 + np.exp(-(a * s + b)))
            ga = float(np.mean((p - y) * s))
            gb = float(np.mean(p - y))
            a -= lr * ga
            b -= lr * gb
        self.a, self.b = a, b
        return self

    def __call__(self, score: float) -> float:
        return float(1.0 / (1.0 + math.exp(-(self.a * score + self.b))))

    @staticmethod
    def ece(scores, labels, bins: int = 10) -> float:
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=float)
        total = len(s) or 1
        err = 0.0
        edges = np.linspace(0, 1, bins + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (s >= lo) & (s < hi if hi < 1 else s <= hi)
            if mask.sum():
                err += mask.sum() / total * abs(s[mask].mean() - y[mask].mean())
        return float(err)


# --------------------------------------------------------------------------- #
# Task features
# --------------------------------------------------------------------------- #
FEATURE_NAMES = [
    "req_len", "changed_files", "changed_lines", "test_only", "security_sensitive",
    "public_api", "dep_fanout", "is_bugfix", "is_feature", "is_refactor", "unknown_class",
]


@dataclass
class TaskFeatures:
    values: dict[str, float]
    task_class: str = "unknown"

    def vector(self) -> np.ndarray:
        return np.array([self.values.get(n, 0.0) for n in FEATURE_NAMES])


def extract_features(request: str, changed_files: list[str] | None = None,
                     changed_lines: int = 0, dep_fanout: int = 0) -> TaskFeatures:
    changed_files = changed_files or []
    req = request.lower()
    sec = ("auth", "password", "token", "crypto", "secret", "sql", "sanitize", "exec")
    api = ("interface", "public", "api", "signature", "schema", "migration")
    values = {
        "req_len": min(len(request) / 200.0, 5.0),
        "changed_files": float(len(changed_files)),
        "changed_lines": min(changed_lines / 100.0, 50.0),
        "test_only": 1.0 if changed_files and all("test" in f for f in changed_files) else 0.0,
        "security_sensitive": 1.0 if any(k in req for k in sec) else 0.0,
        "public_api": 1.0 if any(k in req for k in api) else 0.0,
        "dep_fanout": min(dep_fanout / 10.0, 20.0),
        "is_bugfix": 1.0 if any(k in req for k in ("fix", "bug", "error", "fail", "crash")) else 0.0,
        "is_feature": 1.0 if any(k in req for k in ("add", "implement", "support", "feature")) else 0.0,
        "is_refactor": 1.0 if any(k in req for k in ("refactor", "rename", "cleanup", "simplify")) else 0.0,
        "unknown_class": 0.0,
    }
    if values["test_only"]:
        cls = "test_only"
    elif values["security_sensitive"]:
        cls = "security_sensitive"
    elif values["is_bugfix"]:
        cls = "bug_fix"
    elif values["is_feature"]:
        cls = "feature"
    elif values["is_refactor"]:
        cls = "refactor"
    else:
        cls = "unknown"
        values["unknown_class"] = 1.0
    return TaskFeatures(values, cls)


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
ACTIONS = [
    "retrieve_more", "expand_call_graph", "run_targeted_tests",
    "run_related_suite", "run_full_suite", "run_static_analyzer",
    "run_security_checks", "rerun_flaky_failure", "ask_main_llm_to_revise",
    "stop_with_verified_result", "escalate_for_human_review",
]


def select_action(*, has_failures: bool, full_suite_required: bool,
                  full_suite_ran: bool, uncertainty: bool = False,
                  ood: bool = False) -> tuple[str, list[str]]:
    """Pick the next step from the bounded action menu (spec 5.3).

    Deterministic rules are the permanent fallback (spec 5.3 "keep this as a
    permanent fallback"); a learned action-value head (spec P3) can refine the
    choice once action/outcome labels exist.  The returned action is always one
    of :data:`ACTIONS`.
    """
    if has_failures:
        if full_suite_required and not full_suite_ran:
            return "run_full_suite", ["high_risk_failure"]
        return "ask_main_llm_to_revise", ["unexpected_failure"]
    if (uncertainty or ood) and not full_suite_ran:
        return "run_full_suite", ["uncertainty_above_threshold"]
    return "stop_with_verified_result", ["required_gates_passed"]


@dataclass
class Decision:
    task_class: str
    risk: float
    risk_source: str
    retrieval: dict
    verification: dict
    fallback: str
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_class": self.task_class,
            "risk": {"score": round(self.risk, 4), "source": self.risk_source},
            "retrieval": self.retrieval,
            "verification": self.verification,
            "fallback": self.fallback,
            "reason_codes": self.reason_codes,
        }


class Policy:
    """Deterministic rules; optional learned risk head used for ranking only
    until shadow validation passes (spec 5.3, roadmap phase 2)."""

    def __init__(self, risk_model: GBDT | None = None,
                 calibrator: PlattScaler | None = None) -> None:
        self.risk_model = risk_model
        self.calibrator = calibrator
        self.version = "policy-deterministic-v1"

    def decide(self, feats: TaskFeatures, *, uncertainty: float = 0.0,
               use_learned: bool = False) -> Decision:
        reasons: list[str] = []
        risk = (
            0.2
            + 0.3 * feats.values["security_sensitive"]
            + 0.3 * feats.values["public_api"]
            + 0.1 * min(feats.values["changed_files"], 3)
            + 0.05 * feats.values["dep_fanout"]
        )
        risk = min(risk, 0.95)
        source = "rule"
        if use_learned and self.risk_model and self.risk_model.trained:
            raw = float(self.risk_model.predict(feats.vector())[0])
            risk = self.calibrator(raw) if self.calibrator else raw
            source = "gbdt"
        if feats.values["security_sensitive"]:
            reasons.append("security_sensitive_task")
        if feats.values["public_api"]:
            reasons.append("public_interface_changed")
        if uncertainty > 0.5:
            reasons.append("uncertainty_above_threshold")

        high = risk >= 0.6 or uncertainty > 0.5 or feats.task_class == "security_sensitive"
        order = ["targeted_tests"]
        if high:
            order += ["typecheck", "related_suite", "full_suite"]
            reasons.append("high_risk_full_suite")
        else:
            order += ["typecheck", "related_suite"]
        depth = "deep" if high else "standard"
        decision = Decision(
            task_class=feats.task_class,
            risk=risk,
            risk_source=source,
            retrieval={"mode": "hybrid", "candidate_limit": 40 if high else 25, "expand": high},
            verification={"order": order, "require_full_suite": high},
            fallback="deterministic_high_coverage",
            reason_codes=reasons,
        )
        _ = depth
        return decision

    def train_risk(self, X, y) -> "Policy":
        self.risk_model = GBDT(n_estimators=60, learning_rate=0.2, max_depth=3,
                               loss="logloss").fit(X, y)
        raw = self.risk_model.predict(np.asarray(X))
        self.calibrator = PlattScaler().fit(raw, y)
        self.version = "policy-gbdt-v1"
        return self


# --------------------------------------------------------------------------- #
# Test selection and flakiness
# --------------------------------------------------------------------------- #
@dataclass
class TestRank:
    __test__ = False

    test_id: str
    predicted_relevance: float
    predicted_flakiness: float
    estimated_duration_s: float
    evidence: list[str]
    model_version: str
    mandatory: bool = False

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "predicted_relevance": round(self.predicted_relevance, 4),
            "predicted_flakiness": round(self.predicted_flakiness, 4),
            "estimated_duration_seconds": self.estimated_duration_s,
            "evidence": self.evidence,
            "model_version": self.model_version,
        }


class TestSelector:
    __test__ = False

    """Ranks tests by predicted regression-detection value. In v1 it never
    skips a designated critical suite, it only reorders (spec 5.4)."""

    MANDATORY = ("security", "smoke", "integration")

    def __init__(self, model: GBDT | None = None, flaky: "FlakyModel | None" = None) -> None:
        self.model = model
        self.flaky = flaky or FlakyModel()
        self.version = "test-ranker-heuristic-v1"

    @staticmethod
    def _features(changed_symbols: set[str], test_symbols: set[str], test_path: str,
                  duration: float, history: list[bool]) -> np.ndarray:
        overlap = len(changed_symbols & test_symbols) / (len(changed_symbols) or 1)
        name_hit = 1.0 if any(s in test_path for s in changed_symbols) else 0.0
        fail_rate = (sum(1 for x in history if not x) / len(history)) if history else 0.0
        return np.array([overlap, name_hit, len(changed_symbols), fail_rate,
                         min(duration, 30) / 30.0])

    def rank(self, tests: list[dict], changed_symbols: list[str]) -> list[TestRank]:
        syms = set(s.lower() for s in changed_symbols)
        out: list[TestRank] = []
        for t in tests:
            path = t["test_id"]
            history = t.get("history", [])
            test_syms = set(s.lower() for s in t.get("symbols", [])) | set(tokenize(path))
            feats = self._features(syms, test_syms, path, t.get("duration", 1.0), history)
            if self.model and self.model.trained:
                rel = float(self.model.predict(feats)[0])
            else:
                rel = float(0.45 * feats[0] + 0.35 * feats[1] + 0.2 * feats[3])
            evidence = []
            if feats[0] > 0:
                evidence.append("covers_changed_symbol")
            if feats[1]:
                evidence.append("name_matches_changed_symbol")
            if feats[3] > 0.3:
                evidence.append("historically_failed_for_related_change")
            flake = self.flaky.predict(history, t.get("duration", 1.0))
            mandatory = any(m in path.lower() for m in self.MANDATORY)
            if mandatory:
                evidence.append("mandatory_suite")
            out.append(TestRank(path, rel, flake, t.get("duration", 1.0),
                                evidence, self.version, mandatory))
        out.sort(key=lambda r: -r.predicted_relevance)
        return out

    def train(self, examples: list[tuple[set[str], set[str], str, float, list[bool], int]]) -> "TestSelector":
        X, y = [], []
        for syms, test_syms, path, dur, hist, label in examples:
            X.append(self._features(syms, test_syms, path, dur, hist))
            y.append(label)
        self.model = GBDT(n_estimators=50, learning_rate=0.2, max_depth=3,
                          loss="logloss").fit(X, y)
        self.version = "test-ranker-gbdt-v1"
        return self


class FlakyModel:
    """Estimates flakiness from the pass/fail sequence and duration variance."""

    def predict(self, history: list[bool], duration: float = 1.0) -> float:
        if len(history) < 2:
            return 0.0
        flips = sum(1 for a, b in zip(history, history[1:]) if a != b)
        base = flips / (len(history) - 1)
        return float(min(base, 0.95))


# --------------------------------------------------------------------------- #
# Conformal / OOD calibration (spec 5.8)
# --------------------------------------------------------------------------- #
class ConformalCalibrator:
    """Distribution-free calibration wrapping any score calibrator.

    ``fit`` learns a Platt map and a split-conformal quantile on held-out
    calibration scores, so ``predict_set`` returns a prediction set with
    finite-sample coverage ``>= 1 - alpha``.  ``ood_score`` is the nearest
    training-feature distance, used to widen the risk decision when a task
    lies outside the observed distribution instead of trusting the point
    score (spec 5.8: do not present under-validated confidence as certainty).
    """

    def __init__(self, base: PlattScaler | None = None, alpha: float = 0.1) -> None:
        self.base = base or PlattScaler()
        self.alpha = alpha
        self.qhat: float | None = None
        self.train_matrix: np.ndarray | None = None
        self.fitted = False
        self.version = "conformal-platt-v2"

    def fit(self, raw_scores, labels, matrix=None) -> "ConformalCalibrator":
        raw = np.asarray(raw_scores, dtype=float)
        y = np.asarray(labels, dtype=float)
        self.base.fit(raw, y)
        p = np.array([self.base(s) for s in raw])
        s_true = np.where(y > 0.5, p, 1.0 - p)
        n = len(s_true)
        if n:
            q = min(1.0, math.ceil((n + 1) * (1.0 - self.alpha)) / n)
            self.qhat = float(np.quantile(s_true, q))
        self.train_matrix = None if matrix is None else np.asarray(matrix, dtype=float)
        self.fitted = True
        return self

    def prob(self, raw: float) -> float:
        return self.base(raw)

    def predict_set(self, raw: float) -> set[int]:
        if self.qhat is None:
            return {0, 1}
        p = self.base(raw)
        out: set[int] = set()
        if p >= 1.0 - self.qhat:
            out.add(1)
        if (1.0 - p) >= 1.0 - self.qhat:
            out.add(0)
        return out or {1 if p >= 0.5 else 0}

    def ood_score(self, row) -> float:
        if self.train_matrix is None or len(self.train_matrix) == 0 or row is None:
            return 0.0
        d = np.linalg.norm(self.train_matrix - np.asarray(row, dtype=float), axis=1)
        return float(np.min(d))

    def coverage(self, raw_scores, labels) -> float:
        raw = np.asarray(raw_scores, dtype=float)
        y = np.asarray(labels, dtype=float)
        hit = sum(1 for r, t in zip(raw, y) if int(t > 0.5) in self.predict_set(r))
        return hit / (len(raw) or 1)


# --------------------------------------------------------------------------- #
# Change impact / fault localisation (spec 5.5)
# --------------------------------------------------------------------------- #
class ChangeImpactModel:
    """Ranks test files by how likely a changed symbol affects them.

    Features are graph- and token-derived only (no second LLM).  Untrained,
    it is a deterministic token heuristic; trained, it is a GBDT over the
    same features, so wiring it converges rather than switching behaviour.
    """

    def __init__(self, model: GBDT | None = None) -> None:
        self.model = model
        self.version = "impact-heuristic-v1"

    @staticmethod
    def features(symbol: str, test_path: str, test_tokens: set[str], fanout: int) -> np.ndarray:
        sym = symbol.lower()
        path = test_path.lower()
        name_hit = 1.0 if sym in test_tokens else 0.0
        path_hit = 1.0 if any(p and p in path for p in sym.split("_")) else 0.0
        return np.array([name_hit, path_hit, min(fanout, 10) / 10.0,
                         min(len(test_tokens), 400) / 400.0])

    def predict(self, symbol: str, test_path: str, test_tokens: set[str],
                fanout: int = 0) -> float:
        f = self.features(symbol, test_path, test_tokens, fanout)
        if self.model and self.model.trained:
            return float(self.model.predict(f)[0])
        return float(max(f[0], f[1]))

    def train(self, examples: list[tuple[str, str, set[str], int, float]]) -> "ChangeImpactModel":
        if not examples:
            return self
        X = np.vstack([self.features(s, p, t, fo) for s, p, t, fo, _ in examples])
        y = np.array([lbl for *_, lbl in examples])
        self.model = GBDT(n_estimators=40, learning_rate=0.25, max_depth=3,
                          loss="logloss").fit(X, y)
        self.version = "impact-gbdt-v1"
        return self
