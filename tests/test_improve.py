"""Tests for the Improve-phase monitoring rule and the sensor-overlap check."""
import numpy as np
import pandas as pd
import pytest

from src.analyze import correlated_pairs
from src.improve import best_threshold, evaluate_rule


def test_best_threshold_hand():
    # fails are exactly the runs with x >= 4 -> threshold 4, J = 3/3 - 0/3 = 1
    r = best_threshold([1, 2, 3, 4, 5, 6], [0, 0, 0, 1, 1, 1])
    assert (r["threshold"], r["direction"], r["j"]) == (4.0, 1, 1.0)


def test_best_threshold_lower_direction_and_nan():
    # fails read LOW; NaN ignored -> flag x <= 2, J = 1
    r = best_threshold([1, 2, np.nan, 5, 6, 7], [1, 1, 0, 0, 0, 0])
    assert (r["threshold"], r["direction"], r["j"]) == (2.0, -1, 1.0)


def test_evaluate_rule_hand():
    # rule x >= 5 on 8 runs; fails at runs 2, 3, 7
    x = [1, 2, 5, 6, 7, 1, np.nan, 9]
    y = [0, 0, 1, 1, 0, 0, 0, 1]
    e = evaluate_rule(x, y, 5, 1)
    # flagged: runs 2,3,4,7 -> caught 3 fails (2,3,7), 1 false alarm (4); missing run 6 not flagged
    assert (e["flagged"], e["fails_caught"], e["false_alarms"]) == (4, 3, 1)
    assert e["recall"] == pytest.approx(1.0)
    assert e["false_alarm_rate"] == pytest.approx(1 / 5)
    assert e["precision"] == pytest.approx(0.75)
    assert e["lift"] == pytest.approx(0.75 / (3 / 8))     # = 2.0


def test_correlated_pairs():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = pd.DataFrame({"a": a, "b": a * 2 + 1, "c": rng.normal(size=200)})
    p = correlated_pairs(X, 0.7)
    top = p.iloc[0]
    assert {top.sensor_a, top.sensor_b} == {"a", "b"} and top.rho == pytest.approx(1.0)
    assert p["overlap"].sum() == 1
