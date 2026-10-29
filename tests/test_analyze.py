"""Tests for src/analyze.py."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.analyze import (benjamini_hochberg, make_pipeline, missingness_screen, pareto_table,
                         rank_sensors, screen_sensors, selection_frequency, time_split)


def test_bh_hand_example():
    # sorted p: .005 .01 .03 .04 ; p*m/rank = .02 .02 .04 .04 ; running min from the top keeps them
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    assert q == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_bh_running_minimum():
    # p*m/rank = .04, .025 -> the first must drop to .025 (q-values never decrease with p)
    q = benjamini_hochberg([0.02, 0.025])
    assert q == pytest.approx([0.025, 0.025])


def test_bh_matches_statsmodels_and_ignores_nan():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=50) ** 2
    expected = multipletests(p, method="fdr_bh")[1]
    with_nan = np.append(p, np.nan)
    q = benjamini_hochberg(with_nan)
    assert q[:-1] == pytest.approx(expected)
    assert np.isnan(q[-1])


@pytest.fixture
def two_group():
    rng = np.random.default_rng(0)
    n = 400
    fail = pd.Series((np.arange(n) % 10 == 0).astype(int))          # 10% fails
    X = pd.DataFrame({
        "shifted": rng.normal(0, 1, n) + 1.5 * fail,                 # clear difference
        "null": rng.normal(0, 1, n),                                 # no difference
        "sparse": np.where(np.arange(n) < 395, np.nan, 1.0),        # too few measured values
    })
    X.loc[:49, "shifted"] = np.nan                                    # gaps must be excluded, not imputed
    return X, fail


def test_screen_flags_real_difference_only(two_group):
    X, fail = two_group
    s = screen_sensors(X, fail, min_group_n=3, fdr_q=0.05).set_index("sensor")
    assert bool(s.loc["shifted", "significant_both"])
    assert not bool(s.loc["null", "significant_both"])
    assert np.isnan(s.loc["sparse", "welch_p"])                        # gated by min_group_n
    assert s.loc["shifted", "n_pass"] + s.loc["shifted", "n_fail"] == 350  # 50 gaps excluded
    assert s.loc["shifted", "effect_size_d"] > 1


def test_screen_p_values_match_scipy(two_group):
    X, fail = two_group
    s = screen_sensors(X, fail, 3, 0.05).set_index("sensor")
    v = X["null"].to_numpy()
    f = fail.astype(bool).to_numpy()
    assert s.loc["null", "welch_p"] == pytest.approx(stats.ttest_ind(v[f], v[~f], equal_var=False).pvalue)
    assert s.loc["null", "mw_p"] == pytest.approx(stats.mannwhitneyu(v[f], v[~f]).pvalue)


def test_missingness_hand_table():
    # 20 runs; sensor 'a' missing in runs 0-9. fails: runs 0-4 (all while missing).
    fail = pd.Series([1] * 5 + [0] * 15)
    a = np.where(np.arange(20) < 10, np.nan, 1.0)
    X = pd.DataFrame({"a": a, "b": a.copy(), "full": np.ones(20)})
    m = missingness_screen(X, fail, min_count=5, fdr_q=0.05).set_index("sensor")
    assert "full" not in m.index                                     # never missing -> not tested
    assert m.loc["a", "fail_rate_missing"] == pytest.approx(0.5)     # 5 of 10
    assert m.loc["a", "fail_rate_measured"] == pytest.approx(0.0)    # 0 of 10
    assert m.loc["a", "fisher_p"] == pytest.approx(stats.fisher_exact([[5, 5], [0, 10]]).pvalue)
    assert m.loc["a", "pattern_id"] == m.loc["b", "pattern_id"]      # same runs missing -> one pattern


def test_time_split_keeps_order():
    tr, te = time_split(10, 0.3)
    assert tr.tolist() == list(range(7)) and te.tolist() == [7, 8, 9]


def test_imputer_uses_training_medians_only():
    Xtr = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0]})
    ytr = pd.Series([0, 0, 0, 1, 1, 1])
    pipe = make_pipeline(1.0, "balanced", 0).fit(Xtr, ytr)
    assert pipe.named_steps["impute"].statistics_[0] == pytest.approx(3.0)  # median of 1,2,3,5,6


def test_stable_strong_sensor_is_always_selected():
    rng = np.random.default_rng(3)
    n = 300
    y = pd.Series((rng.uniform(size=n) < 0.15).astype(int))
    X = pd.DataFrame({"strong": rng.normal(size=n) + 2.5 * y,
                      **{f"noise{i}": rng.normal(size=n) for i in range(5)}})
    freq = selection_frequency(X, y, C=0.05, cfg_a={"class_weight": "balanced",
                                                    "random_seed": 0, "bootstrap_n": 30})
    assert freq["strong"] == pytest.approx(1.0)


def test_pareto_shares_and_ranking_tiers():
    coef = pd.Series({"a": 0.6, "b": -0.3, "c": 0.1, "d": 0.0})
    freq = pd.Series({"a": 1.0, "b": 0.5, "c": 0.9, "d": 0.0})
    p = pareto_table(coef, freq, top_n=10)
    assert p["sensor"].tolist() == ["a", "b", "c"]                 # zero coefficient left out
    assert p["share"].tolist() == pytest.approx([0.6, 0.3, 0.1])
    assert p["cum_share"].iloc[-1] == pytest.approx(1.0)
    assert p.loc[1, "direction"] == "lower value, more fails"
    screen = pd.DataFrame({"sensor": ["b", "e", "a"], "significant_both": [True, True, False],
                           "mw_q": [0.01, 0.001, 0.5]})
    # tier1: b (sig + selected); tier2: e (sig only); tier3: a then c (selected only, by freq)
    assert rank_sensors(p, screen)[:4] == ["b", "e", "a", "c"]
