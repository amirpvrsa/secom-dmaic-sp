"""Tests for src/clean.py on a small hand-built table where every answer is
worked out by hand (see comments)."""
import numpy as np
import pandas as pd
import pytest

from src.clean import clean, dataset_overview, impute, sensor_profile

N = np.nan
CFG = {"clean": {"max_missing_frac": 0.50, "constant_max_unique": 1,
                 "near_constant_dominant_frac": 0.90, "impute_strategy": "median"}}


@pytest.fixture
def df():
    sensors = pd.DataFrame({
        # 2 missing of 10 -> keep; median of 1..8 = 4.5
        "s_keep":      [1, 2, 3, 4, 5, 6, 7, 8, N, N],
        # every value missing -> all_missing
        "s_all_nan":   [N] * 10,
        # 6 of 10 missing = 60% > 50% -> too_missing
        "s_missing60": [1, 2, 3, 4, N, N, N, N, N, N],
        # exactly 50% missing is NOT more than 50% -> keep; median of 1..5 = 3
        "s_missing50": [1, 2, 3, 4, 5, N, N, N, N, N],
        # one distinct value -> constant
        "s_const":     [7.0] * 10,
        # one distinct value plus 30% gaps -> still constant (not too_missing)
        "s_const_nan": [7, 7, 7, 7, 7, 7, 7, N, N, N],
        # top value is 9 of 10 = 90% >= 90% -> near_constant
        "s_near":      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        # top value is 8 of 10 = 80% < 90% -> keep
        "s_almost":    [0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
    }, dtype=float)
    meta = pd.DataFrame({
        "run_id": range(10),
        "timestamp": pd.date_range("2008-08-01", periods=10, freq="h"),
        "label": [-1] * 8 + [1] * 2,
        "fail": [0] * 8 + [1] * 2,
    })
    return pd.concat([meta, sensors], axis=1)


def test_profile_hand_values(df):
    p = sensor_profile(df.drop(columns=["run_id", "timestamp", "label", "fail"]))
    assert p.loc["s_missing60", "missing_frac"] == pytest.approx(0.6)
    assert p.loc["s_const_nan", "n_unique"] == 1
    assert p.loc["s_near", "dominant_frac"] == pytest.approx(0.9)
    assert p.loc["s_almost", "dominant_frac"] == pytest.approx(0.8)


def test_each_sensor_gets_the_right_rule(df):
    r = clean(df, CFG).by_sensor.set_index("sensor")["rule"]
    assert r.to_dict() == {
        "s_keep": "keep", "s_all_nan": "all_missing", "s_missing60": "too_missing",
        "s_missing50": "keep", "s_const": "constant", "s_const_nan": "constant",
        "s_near": "near_constant", "s_almost": "keep",
    }


def test_summary_counts_add_up(df):
    s = clean(df, CFG).summary.set_index("step")
    assert (s["sensors_before"] - s["sensors_removed"] == s["sensors_after"]).all()
    assert s.loc["start", "sensors_after"] == 8
    assert s.loc["drop: all_missing", "sensors_removed"] == 1
    assert s.loc["drop: too_missing", "sensors_removed"] == 1
    assert s.loc["drop: constant", "sensors_removed"] == 2
    assert s.loc["drop: near_constant", "sensors_removed"] == 1
    assert s.loc["impute: median", "sensors_after"] == 3
    # gaps left in kept sensors: 2 (s_keep) + 5 (s_missing50) + 0 (s_almost) = 7 of 30 cells
    assert s.loc["impute: median", "cells_imputed"] == 7
    assert s.loc["impute: median", "cells_imputed_pct_of_kept"] == pytest.approx(7 / 30)


def test_imputed_values_are_hand_medians(df):
    res = clean(df, CFG)
    assert res.data["s_keep"].tolist()[-2:] == [4.5, 4.5]
    assert res.data["s_missing50"].tolist()[-5:] == [3.0] * 5
    assert not res.data.isna().any().any()


def test_mask_marks_exactly_the_original_gaps(df):
    res = clean(df, CFG)
    pd.testing.assert_frame_equal(res.missing_mask, df[res.missing_mask.columns].isna())


def test_rows_and_meta_untouched(df):
    res = clean(df, CFG)
    assert len(res.data) == len(df)
    pd.testing.assert_frame_equal(res.data[["run_id", "timestamp", "label", "fail"]],
                                  df[["run_id", "timestamp", "label", "fail"]])


def test_changing_threshold_in_config_changes_result(df):
    loose = {"clean": {**CFG["clean"], "max_missing_frac": 0.65}}
    r = clean(df, loose).by_sensor.set_index("sensor")["rule"]
    assert r["s_missing60"] == "keep"


def test_mean_strategy_and_bad_strategy():
    X = pd.DataFrame({"a": [1.0, 2.0, 6.0, N]})
    filled, mask, fill = impute(X, "mean")
    assert filled["a"].iloc[-1] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        impute(X, "zero")


def test_overview_hand_values(df):
    o = dataset_overview(df).set_index("metric")["value"]
    assert o["runs"] == 10
    assert o["failed runs"] == 2
    assert o["fail rate"] == pytest.approx(0.2)
    assert o["sensors"] == 8
    # NaNs: 2 + 10 + 6 + 5 + 0 + 3 + 0 + 0 = 26
    assert o["missing cells"] == 26
    assert o["runs sharing a timestamp with another run"] == 0
