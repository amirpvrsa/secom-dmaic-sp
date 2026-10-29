"""Improve phase, honestly scoped.

Observational data cannot show that changing a sensor reading would reduce failures.
What it CAN test is a monitoring idea: "if we had flagged every run where sensor S
crossed threshold T, how many failures would we have caught, and at what false-alarm cost?"

To keep the test honest, the threshold and its direction are chosen on TRAINING runs
(the earlier part of the timeline) and the rule is scored on the LATER held-out runs.

    Youden's J = true-positive rate - false-positive rate
               = (fails flagged / all fails) - (passes flagged / all passes)
    lift       = fail rate among flagged runs / overall fail rate   (1 = no better than random)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.analyze import benjamini_hochberg, time_split


def best_threshold(x, y) -> dict:
    """Threshold and direction that maximise Youden's J on (x, y), measured values only.
    direction +1 flags x >= threshold; -1 flags x <= threshold."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    n_pos, n_neg = y.sum(), (1 - y).sum()
    if n_pos == 0 or n_neg == 0:
        return {"threshold": np.nan, "direction": 0, "j": np.nan}
    direction = 1 if x[y == 1].mean() >= x[y == 0].mean() else -1
    best = {"threshold": np.nan, "direction": direction, "j": -np.inf}
    for t in np.unique(x):
        flag = direction * x >= direction * t
        j = flag[y == 1].sum() / n_pos - flag[y == 0].sum() / n_neg
        if j > best["j"]:
            best = {"threshold": float(t), "direction": direction, "j": float(j)}
    return best


def evaluate_rule(x, y, threshold: float, direction: int) -> dict:
    """Score a fixed rule. Runs where the sensor is missing are counted as not flagged."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    flag = ~np.isnan(x) & (direction * np.nan_to_num(x, nan=0.0) >= direction * threshold)
    tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum()); tn = int((~flag & (y == 0)).sum())
    base = y.mean() if len(y) else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    return {
        "runs": len(y), "fails": int(y.sum()), "flagged": tp + fp,
        "fails_caught": tp, "false_alarms": fp,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else np.nan,
        "precision": prec,
        "lift": prec / base if base and not np.isnan(prec) else np.nan,
        "fisher_p": float(stats.fisher_exact([[tp, fp], [fn, tn]]).pvalue),
    }


def run_improve(raw: pd.DataFrame, sensors: list[str], cfg: dict) -> pd.DataFrame:
    tr, te = time_split(len(raw), cfg["analyze"]["holdout_frac"])
    y = raw["fail"].to_numpy()
    rows = []
    for s in sensors[: cfg["improve"]["n_sensors"]]:
        x = raw[s].to_numpy()
        rule = best_threshold(x[tr], y[tr])
        if rule["direction"] == 0:
            continue
        train = evaluate_rule(x[tr], y[tr], rule["threshold"], rule["direction"])
        test = evaluate_rule(x[te], y[te], rule["threshold"], rule["direction"])
        rows.append({"sensor": s, "rule": f"{'>=' if rule['direction'] > 0 else '<='} {rule['threshold']:.4g}",
                     "threshold": rule["threshold"], "direction": rule["direction"],
                     "train_j": rule["j"],
                     **{f"train_{k}": v for k, v in train.items()},
                     **{f"test_{k}": v for k, v in test.items()}})
    out = pd.DataFrame(rows)
    if len(out):
        # several sensors are tried, so correct the held-out p-values for multiple testing
        out["test_fisher_q"] = benjamini_hochberg(out["test_fisher_p"])
        out["holds_up"] = (out["test_fisher_q"] < cfg["analyze"]["fdr_q"]) & (out["test_lift"] > 1)
    return out
