"""Drift and out-of-distribution monitoring (spec 5.8, 8.3, 9.2, 13).

The learned models are only trustworthy while the data they see resembles what
they were trained on.  The spec calls out *calibration drift after
language/framework/repository changes*, OOD detection, and leakage-resistant,
time-separated evaluation.  This module provides the deterministic checks those
require, all numpy-only and side-effect free:

* ``psi``             - Population Stability Index for one feature.
* ``feature_drift``   - per-feature PSI across the whole feature vector.
* ``calibration_drift`` - ECE/Brier change between a reference window and a
                          recent window.
* ``ood_rate``        - fraction of recent tasks lying outside the reference
                          feature distribution.
* ``drift_report``    - splits the ledger by time and returns a verdict with
                          machine-readable reason codes (spec 3.2).

None of this changes control flow by itself; the orchestrator consults it and
falls back conservatively.  Thresholds are deliberately simple and inspectable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .models import PlattScaler

STABLE, WARNING, DRIFTED = "stable", "warning", "drifted"


def _percentiles(ref: np.ndarray, bins: int) -> np.ndarray:
    qs = np.linspace(0, 100, bins + 1)[1:-1]
    edges = np.unique(np.percentile(ref, qs))
    return edges


def _bucket_share(values: np.ndarray, edges: np.ndarray, bins: int) -> np.ndarray:
    idx = np.searchsorted(edges, values, side="right")
    counts = np.bincount(idx, minlength=bins + 1).astype(float)
    total = counts.sum()
    return counts / total if total else counts


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index of one numeric feature.

    <0.1 no significant shift, 0.1-0.25 moderate, >0.25 major (conventional
    thresholds).  Uses quantile edges from the reference so it is invariant to
    the feature's raw scale.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size == 0 or cur.size == 0:
        return 0.0
    if float(np.std(ref)) < 1e-9:
        # Degenerate reference: quantile binning is meaningless, so any shift
        # off the constant is maximal drift, no shift is none.
        return 0.0 if bool(np.allclose(cur, ref[0])) else 1.0
    edges = _percentiles(ref, bins)
    r = _bucket_share(ref, edges, bins)
    c = _bucket_share(cur, edges, bins)
    eps = 1e-6
    r = np.where(r == 0, eps, r)
    c = np.where(c == 0, eps, c)
    return float(np.sum((c - r) * np.log(c / r)))


@dataclass
class DriftReport:
    verdict: str = STABLE
    reason_codes: list[str] = field(default_factory=list)
    feature_psi: dict[str, float] = field(default_factory=dict)
    max_psi_feature: str = ""
    max_psi: float = 0.0
    calibration_delta: float | None = None
    ood_rate: float = 0.0
    n_reference: int = 0
    n_recent: int = 0

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason_codes": self.reason_codes,
            "max_psi": round(self.max_psi, 4),
            "max_psi_feature": self.max_psi_feature,
            "calibration_delta": (None if self.calibration_delta is None
                                  else round(self.calibration_delta, 4)),
            "ood_rate": round(self.ood_rate, 4),
            "n_reference": self.n_reference,
            "n_recent": self.n_recent,
        }


def feature_drift(reference, current, names: list[str]) -> tuple[dict[str, float], str, float]:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    scores: dict[str, float] = {}
    for j, name in enumerate(names):
        if j >= ref.shape[1]:
            break
        scores[name] = round(psi(ref[:, j], cur[:, j]), 4)
    if not scores:
        return {}, "", 0.0
    worst = max(scores, key=lambda k: scores[k])
    return scores, worst, scores[worst]


def ood_rate(conformal, rows, threshold: float | None = None) -> float:
    """Fraction of ``rows`` whose nearest-training-feature distance exceeds
    ``threshold`` (default: the reference's own max, so in-distribution data is
    ~0)."""
    if conformal is None or conformal.train_matrix is None or len(rows) == 0:
        return 0.0
    if threshold is None:
        d = np.linalg.norm(
            conformal.train_matrix[:, None, :] - conformal.train_matrix[None, :, :], axis=2)
        threshold = float(np.percentile(d, 95)) if d.size else 0.0
    flagged = sum(1 for r in rows if conformal.ood_score(r) > threshold)
    return flagged / len(rows)


def calibration_drift(ref_scores, ref_labels, cur_scores, cur_labels) -> float:
    """Change in Expected Calibration Error (recent - reference)."""
    ece_ref = PlattScaler.ece(ref_scores, ref_labels)
    ece_cur = PlattScaler.ece(cur_scores, cur_labels)
    return float(ece_cur - ece_ref)


def drift_report(learned, entries: list[dict], *, feature_names: list[str],
                 recent_frac: float = 0.3, min_n: int = 8) -> DriftReport:
    """Time-ordered drift check over the ledger's task features and outcomes.

    Reference = the earliest ``1 - recent_frac`` of labelled tasks (the window
    the models were presumably fit on); recent = the rest.  Splitting by time,
    not randomly, follows spec 8.3.
    """
    from .training import risk_examples  # local import avoids a cycle

    X, y = risk_examples(entries)
    rep = DriftReport(n_reference=len(y))
    if len(y) < min_n or X.size == 0:
        rep.reason_codes.append("insufficient_history")
        return rep
    split = max(1, int(len(y) * (1 - recent_frac)))
    if split >= len(y):
        split = len(y) - 1
    ref_X, cur_X = X[:split], X[split:]
    ref_y, cur_y = y[:split], y[split:]
    if len(cur_y) == 0:
        rep.reason_codes.append("insufficient_history")
        return rep
    rep.n_reference, rep.n_recent = len(ref_y), len(cur_y)

    scores, worst, worst_val = feature_drift(ref_X, cur_X, feature_names)
    rep.feature_psi, rep.max_psi_feature, rep.max_psi = scores, worst, worst_val
    if worst_val > 0.25:
        rep.verdict = DRIFTED
        rep.reason_codes.append(f"feature_drift:{worst}")
    elif worst_val > 0.1:
        rep.verdict = WARNING
        rep.reason_codes.append(f"feature_shift:{worst}")

    if learned is not None and learned.risk_trained:
        raw_ref = learned.policy.risk_model.predict(ref_X)
        raw_cur = learned.policy.risk_model.predict(cur_X)
        rep.calibration_delta = calibration_drift(raw_ref, ref_y, raw_cur, cur_y)
        if rep.verdict != DRIFTED and rep.calibration_delta > 0.15:
            rep.verdict = WARNING
            rep.reason_codes.append("calibration_delta")
        rep.ood_rate = ood_rate(learned.conformal, cur_X)
        if rep.ood_rate > 0.3 and rep.verdict != DRIFTED:
            rep.verdict = WARNING
            rep.reason_codes.append("ood_rate")
    if not rep.reason_codes:
        rep.reason_codes.append("no_drift_detected")
    return rep
