"""Fetch the SECOM dataset, attach timestamps and labels, sort runs by time, cache.

Output of load_secom(): one DataFrame, one row per production run, with columns
    run_id     original row position in the UCI files (0-based), kept for traceability
    timestamp  when the run happened
    label      raw SECOM label (-1 pass, 1 fail by default; set in config)
    fail       1 if the run failed, 0 if it passed
    sensor_000 ... sensor_NNN   one column per sensor, in the original order
Rows are sorted by timestamp (ties keep the original file order).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from src.config import resolve

META_COLS = ["run_id", "timestamp", "label", "fail"]


# ---------- small, pure helpers (these are what the tests check) ----------

def sensor_names(n: int) -> list[str]:
    """sensor_000, sensor_001, ... Zero-padded so they sort correctly as text."""
    width = max(3, len(str(n - 1)))
    return [f"sensor_{i:0{width}d}" for i in range(n)]


def parse_labels(labels: pd.Series, pass_value: int, fail_value: int) -> pd.Series:
    """Map raw labels to fail = 1 / pass = 0. Stops on any unexpected value."""
    values = pd.to_numeric(labels, errors="coerce")
    bad = ~values.isin([pass_value, fail_value])
    if bad.any():
        examples = labels[bad].unique()[:5]
        raise ValueError(f"{bad.sum()} labels are neither {pass_value} nor {fail_value}: {examples}")
    return (values == fail_value).astype(int).rename("fail")


def parse_timestamps(ts: pd.Series, fmt: str) -> pd.Series:
    """Strict timestamp parsing.

    Tries the configured format first (SECOM's own files are day-first), then ISO 8601
    (what pandas writes when we cache). Never guesses day/month order: if neither
    format matches every value, it raises.
    """
    if pd.api.types.is_datetime64_any_dtype(ts):
        return ts
    text = ts.astype(str).str.strip().str.strip('"')
    for f in (fmt, "ISO8601"):
        try:
            return pd.to_datetime(text, format=f, errors="raise")
        except (ValueError, TypeError):
            continue
    raise ValueError(f"Timestamps match neither '{fmt}' nor ISO 8601. Example: {text.iloc[0]!r}")


def assemble(features: pd.DataFrame, raw_labels: pd.Series, timestamps: pd.Series,
             pass_value: int, fail_value: int, fmt: str) -> pd.DataFrame:
    """Combine sensors + labels + timestamps into one frame sorted by time."""
    n = len(features)
    if not (len(raw_labels) == len(timestamps) == n):
        raise ValueError(f"Row counts differ: features {n}, labels {len(raw_labels)}, "
                         f"timestamps {len(timestamps)}")

    sensors = features.reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
    sensors.columns = sensor_names(sensors.shape[1])

    meta = pd.DataFrame({
        "run_id": range(n),
        "timestamp": parse_timestamps(pd.Series(timestamps).reset_index(drop=True), fmt),
        "label": pd.to_numeric(pd.Series(raw_labels).reset_index(drop=True)).astype(int),
    })
    meta["fail"] = parse_labels(meta["label"], pass_value, fail_value)

    df = pd.concat([meta, sensors], axis=1)
    # mergesort is stable: runs with identical timestamps keep their file order
    df = df.sort_values(["timestamp", "run_id"], kind="mergesort").reset_index(drop=True)
    return df


# ---------- the two ways to get the data ----------

def read_raw_files(raw_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Read the original UCI files: secom.data (space-separated sensors, 'NaN' for
    missing) and secom_labels.data (label, then a quoted 'dd/mm/yyyy HH:MM:SS')."""
    features = pd.read_csv(raw_dir / "secom.data", sep=r"\s+", header=None, na_values=["NaN"])
    # each label line looks like:  -1 "19/07/2008 11:55:00"
    # split on the FIRST whitespace only, because the timestamp itself contains a space
    lines = (raw_dir / "secom_labels.data").read_text().strip().splitlines()
    parts = [ln.strip().split(maxsplit=1) for ln in lines if ln.strip()]
    labels = pd.Series([p[0] for p in parts], name="label")
    timestamps = pd.Series([p[1].strip().strip('"') for p in parts], name="timestamp")
    return features, labels, timestamps


def fetch_from_uci(uci_id: int, fmt: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Download through ucimlrepo. ucimlrepo labels each column as Feature, Target,
    ID or Other; the timestamp is found as the non-sensor column that parses as a date."""
    from ucimlrepo import fetch_ucirepo  # imported here so tests never need the network

    ds = fetch_ucirepo(id=uci_id)
    features = ds.data.features.copy()
    targets = ds.data.targets
    if targets is None or targets.shape[1] != 1:
        raise ValueError("Expected exactly one target column (pass/fail) from ucimlrepo")
    raw_labels = targets.iloc[:, 0]

    # candidate timestamp columns: anything in the original table that is not a
    # numeric sensor, plus any non-numeric column ucimlrepo filed under features
    original = ds.data.original
    non_numeric_feats = [c for c in features.columns
                         if not pd.api.types.is_numeric_dtype(features[c])]
    candidates = [c for c in original.columns
                  if c not in features.columns and c not in targets.columns] + non_numeric_feats
    for col in candidates:
        try:
            ts = parse_timestamps(original[col], fmt)
        except ValueError:
            continue
        features = features.drop(columns=[col], errors="ignore")
        return features, raw_labels, ts
    raise ValueError(f"No timestamp column found in ucimlrepo data (checked {candidates}). "
                     "Put secom.data and secom_labels.data in the raw_dir from config.yaml instead.")


# ---------- main entry point ----------

def load_secom(cfg: dict, refresh: bool = False,
               fetcher: Callable[[], tuple[pd.DataFrame, pd.Series, pd.Series]] | None = None
               ) -> pd.DataFrame:
    """Return the assembled, time-sorted SECOM table.

    Order of preference: local cache -> raw UCI files in raw_dir -> download via ucimlrepo.
    `fetcher` lets tests inject fake data without touching the network.
    """
    d = cfg["data"]
    cache = resolve(d["cache_dir"]) / d["cache_file"]

    if cache.exists() and not refresh:
        df = pd.read_csv(cache)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
        return df

    raw_dir = resolve(d["raw_dir"])
    if fetcher is None:
        if (raw_dir / "secom.data").exists() and (raw_dir / "secom_labels.data").exists():
            fetcher = lambda: read_raw_files(raw_dir)  # noqa: E731
        else:
            fetcher = lambda: fetch_from_uci(d["uci_id"], d["timestamp_format"])  # noqa: E731

    features, raw_labels, timestamps = fetcher()
    df = assemble(features, raw_labels, timestamps,
                  d["pass_value"], d["fail_value"], d["timestamp_format"])

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]
