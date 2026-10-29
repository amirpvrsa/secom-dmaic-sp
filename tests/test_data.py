"""Tests for src/data.py. No network: fake data is passed in directly."""
import numpy as np
import pandas as pd
import pytest

from src.data import (assemble, load_secom, parse_labels, parse_timestamps,
                      read_raw_files, sensor_names)

FMT = "%d/%m/%Y %H:%M:%S"


def test_sensor_names_are_zero_padded():
    assert sensor_names(3) == ["sensor_000", "sensor_001", "sensor_002"]
    assert sensor_names(590)[-1] == "sensor_589"


def test_labels_map_fail_to_one_and_pass_to_zero():
    out = parse_labels(pd.Series([-1, 1, -1, 1, 1]), pass_value=-1, fail_value=1)
    assert out.tolist() == [0, 1, 0, 1, 1]


def test_unexpected_label_raises():
    with pytest.raises(ValueError):
        parse_labels(pd.Series([-1, 0, 1]), pass_value=-1, fail_value=1)


def test_timestamps_are_day_first():
    # 01/08/2008 must be 1 August, not 8 January
    ts = parse_timestamps(pd.Series(["01/08/2008 10:00:00"]), FMT)
    assert ts.iloc[0] == pd.Timestamp("2008-08-01 10:00:00")


def test_ambiguous_or_wrong_timestamp_raises_instead_of_guessing():
    with pytest.raises(ValueError):
        parse_timestamps(pd.Series(["2008/31/12 10:00"]), FMT)


def test_assemble_sorts_by_time_and_keeps_run_id():
    feats = pd.DataFrame({0: [1.0, 2.0, 3.0], 1: [10.0, np.nan, 30.0]})
    labels = pd.Series([-1, 1, -1])
    times = pd.Series(["03/08/2008 00:00:00", "01/08/2008 00:00:00", "02/08/2008 00:00:00"])
    df = assemble(feats, labels, times, -1, 1, FMT)
    assert df["run_id"].tolist() == [1, 2, 0]          # chronological order
    assert df["fail"].tolist() == [1, 0, 0]
    assert df["sensor_000"].tolist() == [2.0, 3.0, 1.0]  # sensor values move with their run
    assert np.isnan(df.loc[0, "sensor_001"])


def test_tied_timestamps_keep_file_order():
    feats = pd.DataFrame({0: [1.0, 2.0, 3.0]})
    times = pd.Series(["01/08/2008 00:00:00"] * 3)
    df = assemble(feats, pd.Series([-1, -1, 1]), times, -1, 1, FMT)
    assert df["run_id"].tolist() == [0, 1, 2]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        assemble(pd.DataFrame({0: [1.0, 2.0]}), pd.Series([-1]),
                 pd.Series(["01/08/2008 00:00:00"]), -1, 1, FMT)


def test_read_raw_files_matches_uci_format(tmp_path):
    (tmp_path / "secom.data").write_text("1.5 NaN 3\n4 5 6\n")
    (tmp_path / "secom_labels.data").write_text(
        '-1 "19/07/2008 11:55:00"\n1 "19/07/2008 12:32:00"\n')
    feats, labels, times = read_raw_files(tmp_path)
    assert feats.shape == (2, 3)
    assert np.isnan(feats.iloc[0, 1])
    assert labels.tolist() == ["-1", "1"]
    assert times.tolist() == ["19/07/2008 11:55:00", "19/07/2008 12:32:00"]


def _cfg(tmp_path):
    return {"data": {"uci_id": 179, "cache_dir": str(tmp_path / "cache"),
                     "cache_file": "secom.csv.gz", "raw_dir": str(tmp_path / "raw"),
                     "pass_value": -1, "fail_value": 1, "timestamp_format": FMT}}


def test_cache_round_trip_is_identical(tmp_path):
    cfg = _cfg(tmp_path)
    calls = []

    def fake_fetch():
        calls.append(1)
        return (pd.DataFrame({0: [0.123456789012, np.nan], 1: [2.0, 3.0]}),
                pd.Series([1, -1]),
                pd.Series(["02/08/2008 08:00:00", "01/08/2008 09:30:00"]))

    first = load_secom(cfg, fetcher=fake_fetch)
    second = load_secom(cfg, fetcher=fake_fetch)  # should come from cache
    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second, check_dtype=False)
