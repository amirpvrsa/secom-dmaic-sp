"""Hand-calculated checks for I-MR limits, Western Electric rules and capability indices.

Worked example used throughout:  x = 10, 12, 11, 13, 9   (in time order)
    mean          = 55 / 5 = 11
    moving ranges = |12-10|, |11-12|, |13-11|, |9-13| = 2, 1, 2, 4  ->  MR-bar = 9/4 = 2.25
    sigma_within  = MR-bar / d2 = 2.25 / 1.128 = 1.99468
    sigma_overall = sqrt(((-1)^2 + 1^2 + 0^2 + 2^2 + (-2)^2) / (5-1)) = sqrt(10/4) = 1.58114
"""
import numpy as np
import pandas as pd
import pytest

from src.capability import capability_indices, derive_spec_limits, run_capability
from src.capability import anderson_darling_p, percentile_indices
from src.spc import (chart_sensor, find_stable_window, imr_limits, moving_ranges,
                     stability_ratio, western_electric)

X = [10, 12, 11, 13, 9]


def test_moving_ranges():
    assert moving_ranges(X).tolist() == [2, 1, 2, 4]


def test_imr_limits_hand():
    lim = imr_limits(np.array(X), d2=1.128, D4=3.267, k=3)
    assert lim.center == pytest.approx(11)
    assert lim.mr_bar == pytest.approx(2.25)
    assert lim.sigma == pytest.approx(1.994681, rel=1e-5)
    assert lim.ucl == pytest.approx(11 + 3 * 1.994681, rel=1e-5)   # 16.98404
    assert lim.lcl == pytest.approx(11 - 3 * 1.994681, rel=1e-5)   # 5.01596
    assert lim.mr_ucl == pytest.approx(3.267 * 2.25)               # 7.35075


def test_capability_hand():
    # LSL = 5, USL = 20  ->  width = 15, nearest limit distance = min(20-11, 11-5) = 6
    c = capability_indices(X, lsl=5, usl=20, d2=1.128)
    assert c["Cp"] == pytest.approx(15 / (6 * 1.994681), rel=1e-5)    # 1.25333
    assert c["Cpk"] == pytest.approx(6 / (3 * 1.994681), rel=1e-5)    # 1.00267
    assert c["Pp"] == pytest.approx(15 / (6 * 1.581139), rel=1e-5)    # 1.58114
    assert c["Ppk"] == pytest.approx(6 / (3 * 1.581139), rel=1e-5)    # 1.26491
    assert c["Cp"] == pytest.approx(1.253333, rel=1e-5)
    assert c["Cpk"] == pytest.approx(1.002667, rel=1e-5)
    assert c["pct_outside"] == 0


def test_cpk_uses_nearer_limit_and_can_go_negative():
    # mean 11 sits ABOVE USL 10.5 -> Cpk = (10.5 - 11) / (3 * 1.994681) < 0
    c = capability_indices(X, lsl=0, usl=10.5)
    assert c["Cpk"] == pytest.approx(-0.5 / (3 * 1.994681), rel=1e-5)
    assert c["pct_outside"] == pytest.approx(3 / 5)   # 12, 11, 13 are above 10.5


def test_spec_limits_mean_sd_and_percentile():
    base = np.array([1, 2, 3, 4, 5, np.nan])   # mean 3, sd sqrt(2.5)
    lsl, usl = derive_spec_limits(base, {"spec_method": "mean_sd", "mean_sd_k": 3})
    assert (lsl, usl) == pytest.approx((3 - 3 * 1.581139, 3 + 3 * 1.581139), rel=1e-5)
    lsl, usl = derive_spec_limits(base, {"spec_method": "percentile",
                                         "lower_percentile": 0, "upper_percentile": 100})
    assert (lsl, usl) == (1, 5)


def test_we_rule1_beyond_3_sigma():
    f = western_electric([0, 0, 3.5, -3.1, 2.9], 0, 1)
    assert f["rule1"].tolist() == [False, False, True, True, False]


def test_we_rule2_two_of_three_same_side():
    assert western_electric([2.5, 0, 2.5], 0, 1)["rule2"].tolist() == [False, False, True]
    assert not western_electric([2.5, -2.5, 0], 0, 1)["rule2"].any()   # opposite sides


def test_we_rule3_four_of_five_beyond_1_sigma():
    assert western_electric([1.5, 1.5, 0, 1.5, 1.5], 0, 1)["rule3"].tolist() == [False] * 4 + [True]
    assert not western_electric([1.5, 1.5, 0, 0, 1.5], 0, 1)["rule3"].any()   # only 3 of 5


def test_we_rule4_eight_same_side():
    assert western_electric([0.5] * 8, 0, 1)["rule4"].tolist() == [False] * 7 + [True]
    assert not western_electric([0.5] * 7 + [-0.5], 0, 1)["rule4"].any()


@pytest.fixture
def series():
    # 20 runs; baseline = first 50% = runs 0-9. Run 3 is a failing run with a wild value,
    # which must NOT enter the baseline limits. Run 6 is missing.
    vals = [10, 12, 11, 500, 13, 9, np.nan, 10, 12, 11] + [11, 12] * 5
    return pd.DataFrame({
        "run_id": range(20),
        "timestamp": pd.date_range("2008-08-01", periods=20, freq="h"),
        "fail": [0, 0, 0, 1] + [0] * 16,
        "s": vals,
    })


def test_chart_uses_only_passing_baseline_runs(series):
    r = chart_sensor(series, "s", {"baseline_frac": 0.5, "d2": 1.128, "D4": 3.267,
                                   "sigma_multiplier": 3, "rules": [1, 2, 3, 4]})
    base = [10, 12, 11, 13, 9, 10, 12, 11]          # passing, measured, in time order
    assert r["limits"].n_baseline == 8
    assert r["limits"].center == pytest.approx(np.mean(base))
    assert r["limits"].mr_bar == pytest.approx(np.abs(np.diff(base)).mean())
    assert r["summary"]["n_points"] == 19             # the missing run is skipped
    assert bool(r["points"].loc[r["points"]["run_id"] == 3, "rule1"].iloc[0])


def test_capability_uses_spc_baseline_window(series):
    cfg_s = {"baseline_frac": 0.5, "d2": 1.128, "D4": 3.267, "sigma_multiplier": 3,
             "rules": [1, 2, 3, 4]}
    chart = chart_sensor(series, "s", cfg_s)
    cfg = {"capability": {"spec_method": "mean_sd", "mean_sd_k": 3, "normality_alpha": 0.05,
                          "lower_percentile": 0.135, "upper_percentile": 99.865},
           "spc": cfg_s}
    t = run_capability(series, {"s": chart}, cfg)
    # limits must come from the passing baseline only (run 3's 500 excluded)
    base = np.array([10, 12, 11, 13, 9, 10, 12, 11], dtype=float)
    assert t["lsl_assumed"].iloc[0] == pytest.approx(base.mean() - 3 * base.std(ddof=1))
    assert t["period"].iloc[0] == "baseline window, passing runs"


def test_zero_spread_gives_nan_not_infinity():
    c = capability_indices([5, 5, 5, 5], lsl=0, usl=10)
    assert np.isnan(c["Cp"]) and np.isnan(c["Pp"])


def test_p_chart_hand():
    # week 1: 10 runs, 1 fail; week 2: 40 runs, 4 fails -> p-bar = 5/50 = 0.10
    # week 1 UCL = 0.10 + 3*sqrt(0.1*0.9/10) = 0.10 + 3*0.094868 = 0.384605, LCL = max(0, -0.18) = 0
    # week 2 UCL = 0.10 + 3*sqrt(0.1*0.9/40) = 0.10 + 3*0.047434 = 0.242302
    from src.spc import p_chart
    t1 = pd.date_range("2008-08-04", periods=10, freq="h")      # Monday, week 1
    t2 = pd.date_range("2008-08-11", periods=40, freq="h")      # Monday, week 2
    raw = pd.DataFrame({"timestamp": t1.append(t2),
                        "fail": [1] + [0] * 9 + [1] * 4 + [0] * 36})
    t = p_chart(raw, "W", 3)
    assert t["n"].tolist() == [10, 40]
    assert t["p_bar"].iloc[0] == pytest.approx(0.10)
    assert t["ucl"].tolist() == pytest.approx([0.384605, 0.242302], rel=1e-5)
    assert t["lcl"].tolist() == [0.0, 0.0]
    assert not t["out"].any()


def test_stability_ratio_hand():
    # trend 0..5: MRs all 1 -> sigma_within = 1/1.128 = 0.88652; SD = sqrt(17.5/5) = 1.87083
    # ratio = 1.87083 / 0.88652 = 2.1103 (drift inflates SD, not moving ranges)
    assert stability_ratio([0, 1, 2, 3, 4, 5], 1.128) == pytest.approx(2.1103, abs=1e-4)
    # alternating 0,1: MRs all 1 -> 0.88652; SD = sqrt(1.5/5) = 0.54772 -> ratio 0.6178
    assert stability_ratio([0, 1, 0, 1, 0, 1], 1.128) == pytest.approx(0.6178, abs=1e-4)


def test_find_stable_window_picks_the_steady_stretch():
    x = [0, 1, 2, 3, 4, 5] + [0, 1, 0, 1, 0, 1]
    w = find_stable_window(np.array(x, float), size=6, step=6, d2=1.128)
    assert (w["start"], w["end"]) == (6, 12)
    assert w["ratio"] == pytest.approx(0.6178, abs=1e-4)


def test_find_stable_window_includes_final_stretch():
    # with step 4 the starts would be 0, 4 only; the last stretch (start 5) must be tried too
    x = [0, 5, 10, 15, 20, 25, 0, 1, 0, 1, 0]
    w = find_stable_window(np.array(x, float), size=6, step=4, d2=1.128)
    assert w["start"] == 5


def test_percentile_indices_hand():
    # x = 0..1000 (1,001 values): P0.135 = 1.35, median = 500, P99.865 = 998.65
    # LSL = -100, USL = 1100 -> Pp = 1200 / 997.3 = 1.20325
    # Ppk = min((1100-500)/(998.65-500), (500+100)/(500-1.35)) = 600/498.65 = 1.20325
    r = percentile_indices(np.arange(1001), -100, 1100)
    assert r["Pp_pct"] == pytest.approx(1200 / 997.3, rel=1e-6)
    assert r["Ppk_pct"] == pytest.approx(600 / 498.65, rel=1e-6)


def test_anderson_darling_separates_normal_from_skewed():
    rng = np.random.default_rng(0)
    assert anderson_darling_p(rng.normal(size=500)) > 0.05
    assert anderson_darling_p(rng.exponential(size=500)) < 0.001
