"""Sensor screening and imputation, with every decision logged.

Rules are applied in a fixed order, and each sensor is assigned to the FIRST rule it
hits, so no sensor is counted twice:
    1. all_missing    - no values at all
    2. too_missing    - missing fraction > clean.max_missing_frac
    3. constant       - <= clean.constant_max_unique distinct values
    4. near_constant  - most common value is >= clean.near_constant_dominant_frac of values
    5. keep
Kept sensors then have their remaining gaps filled (median or mean, from config).
Rows (runs) are never dropped, so the time sequence stays complete for SPC.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import META_COLS, sensor_columns

RULE_ORDER = ["all_missing", "too_missing", "constant", "near_constant"]


@dataclass
class CleanResult:
    data: pd.DataFrame          # meta columns + kept, imputed sensors
    missing_mask: pd.DataFrame  # same shape as kept sensors; True where a value was imputed
    summary: pd.DataFrame       # one row per step, with counts
    by_sensor: pd.DataFrame     # one row per sensor: profile + decision + reason
    fill_values: pd.Series      # the value used to fill each kept sensor


def sensor_profile(X: pd.DataFrame) -> pd.DataFrame:
    """Per-sensor statistics used by the screening rules."""
    n_runs = len(X)
    rows = []
    for col in X.columns:
        s = X[col]
        present = s.dropna()
        n_present = len(present)
        if n_present:
            top_count = int(present.value_counts().iloc[0])
            dominant_frac = top_count / n_present
        else:
            dominant_frac = np.nan
        rows.append({
            "sensor": col,
            "n_missing": int(n_runs - n_present),
            "missing_frac": (n_runs - n_present) / n_runs if n_runs else np.nan,
            "n_unique": int(present.nunique()),
            "dominant_frac": dominant_frac,
        })
    return pd.DataFrame(rows).set_index("sensor")


def decide(profile: pd.DataFrame, cfg_clean: dict) -> pd.DataFrame:
    """Attach a decision ('drop'/'keep') and the rule responsible to each sensor."""
    out = profile.copy()
    # np.select checks the conditions top to bottom and takes the FIRST that is true,
    # which is exactly the "first rule wins" order in RULE_ORDER.
    conditions = [
        out["missing_frac"] == 1.0,
        out["missing_frac"] > cfg_clean["max_missing_frac"],
        out["n_unique"] <= cfg_clean["constant_max_unique"],
        out["dominant_frac"] >= cfg_clean["near_constant_dominant_frac"],
    ]
    out["rule"] = np.select(conditions, RULE_ORDER, default="keep")
    out["decision"] = np.where(out["rule"] == "keep", "keep", "drop")
    return out


def impute(X: pd.DataFrame, strategy: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Fill gaps column by column. Returns (filled data, mask of filled cells, fill values)."""
    if strategy == "median":
        fill = X.median()
    elif strategy == "mean":
        fill = X.mean()
    else:
        raise ValueError(f"Unknown impute_strategy {strategy!r}; use 'median' or 'mean'")
    mask = X.isna()
    return X.fillna(fill), mask, fill


def clean(df: pd.DataFrame, cfg: dict) -> CleanResult:
    c = cfg["clean"]
    sensors = sensor_columns(df)
    X = df[sensors]
    n_runs = len(X)

    by_sensor = decide(sensor_profile(X), c)
    thresholds = {
        "all_missing": "100% missing",
        "too_missing": f"missing > {c['max_missing_frac']:.0%}",
        "constant": f"distinct values <= {c['constant_max_unique']}",
        "near_constant": f"top value >= {c['near_constant_dominant_frac']:.0%} of values",
    }

    rows = [{"step": "start", "rule": "all sensors", "sensors_before": len(sensors),
             "sensors_removed": 0, "sensors_after": len(sensors), "cells_imputed": 0}]
    remaining = len(sensors)
    for rule in RULE_ORDER:
        hit = by_sensor.index[by_sensor["rule"] == rule]
        rows.append({"step": f"drop: {rule}", "rule": thresholds[rule],
                     "sensors_before": remaining, "sensors_removed": len(hit),
                     "sensors_after": remaining - len(hit), "cells_imputed": 0})
        remaining -= len(hit)

    kept = list(by_sensor.index[by_sensor["decision"] == "keep"])
    filled, mask, fill_values = impute(X[kept], c["impute_strategy"])
    n_imputed = int(mask.values.sum())
    rows.append({"step": f"impute: {c['impute_strategy']}",
                 "rule": "fill remaining gaps in kept sensors",
                 "sensors_before": remaining, "sensors_removed": 0,
                 "sensors_after": remaining, "cells_imputed": n_imputed})

    summary = pd.DataFrame(rows)
    # share of the kept data that is imputed rather than measured
    kept_cells = remaining * n_runs
    summary["cells_imputed_pct_of_kept"] = summary["cells_imputed"] / kept_cells if kept_cells else np.nan

    by_sensor["n_imputed"] = 0
    by_sensor.loc[kept, "n_imputed"] = mask.sum().astype(int)
    by_sensor["fill_value"] = np.nan
    by_sensor.loc[kept, "fill_value"] = fill_values

    data = pd.concat([df[[m for m in META_COLS if m in df.columns]], filled], axis=1)
    return CleanResult(data=data, missing_mask=mask, summary=summary,
                       by_sensor=by_sensor.reset_index(), fill_values=fill_values)


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Run-level facts for the Define/Measure section. All values computed, none typed in."""
    sensors = sensor_columns(df)
    X = df[sensors]
    n = len(df)
    n_fail = int(df["fail"].sum())
    gaps = df["timestamp"].diff().dropna()
    facts = [
        ("runs", n),
        ("sensors", len(sensors)),
        ("failed runs", n_fail),
        ("passed runs", n - n_fail),
        ("fail rate", n_fail / n if n else np.nan),
        ("first run", df["timestamp"].min()),
        ("last run", df["timestamp"].max()),
        ("runs sharing a timestamp with another run", int(df["timestamp"].duplicated(keep=False).sum())),
        ("median time between runs", gaps.median() if len(gaps) else pd.NaT),
        ("missing cells", int(X.isna().values.sum())),
        ("missing cells, % of all", X.isna().values.mean() if X.size else np.nan),
    ]
    return pd.DataFrame(facts, columns=["metric", "value"])
