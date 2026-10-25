import numpy as np

from rbg.drift import conformal_martingale
from rbg.models import ConformalCalibrator


def test_conformal_stores_calibration_scores():
    cal = ConformalCalibrator().fit([0.1, 0.4, 0.8, 0.9], [0, 0, 1, 1])
    assert cal.cal_scores is not None and len(cal.cal_scores) == 4
    assert 0.0 <= cal.coverage([0.2, 0.7], [0, 1]) <= 1.0


def test_martingale_flags_a_shifted_window():
    rng = np.random.default_rng(0)
    cal = rng.random(200)
    assert conformal_martingale(cal, rng.random(50)) < 1e6
    shifted = np.ones(50)
    assert conformal_martingale(cal, shifted) > 1e6
