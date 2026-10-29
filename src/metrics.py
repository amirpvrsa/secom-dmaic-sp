"""Baseline CTQ performance (yield, DPMO, sigma level) and the drift test.

    yield         = 1 - p                      (p = fail rate)
    DPMO          = p / opportunities_per_run * 1,000,000
    Z_long_term   = Phi^-1(1 - DPMO / 1,000,000)   (Phi^-1 = inverse standard normal CDF)
    sigma level   = Z_long_term + shift           (shift = 1.5 by Six Sigma convention)

Two-proportion z-test (is the fail rate different between two periods?):
    p_pool = (x1 + x2) / (n1 + n2)
    z      = (p1 - p2) / sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    95% CI for p1 - p2 = (p1 - p2) +/- 1.96 * sqrt(p1(1-p1)/n1 + p2(1-p2)/n2)
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def ctq_metrics(n_runs: int, n_fail: int, opportunities_per_run: int = 1, shift: float = 1.5) -> dict:
    p = n_fail / n_runs
    dpmo = p / opportunities_per_run * 1_000_000
    z_lt = float(stats.norm.ppf(1 - dpmo / 1_000_000))
    return {"runs": n_runs, "fails": n_fail, "fail_rate": p, "yield": 1 - p,
            "dpmo": dpmo, "z_long_term": z_lt, "sigma_level": z_lt + shift, "shift": shift}


def two_proportion_test(x1: int, n1: int, x2: int, n2: int, conf: float = 0.95) -> dict:
    p1, p2 = x1 / n1, x2 / n2
    pool = (x1 + x2) / (n1 + n2)
    se_pool = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se_pool if se_pool > 0 else np.nan
    p_value = float(2 * stats.norm.sf(abs(z))) if se_pool > 0 else np.nan
    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zc = stats.norm.ppf(0.5 + conf / 2)
    return {"p1": p1, "p2": p2, "n1": n1, "n2": n2, "x1": x1, "x2": x2, "diff": p1 - p2,
            "z": float(z), "p_value": p_value,
            "ci_low": float(p1 - p2 - zc * se_diff), "ci_high": float(p1 - p2 + zc * se_diff),
            "conf": conf}


def run_metrics(raw, cfg) -> dict:
    m = cfg["metrics"]
    fail = raw["fail"].to_numpy()
    cut = int(round(len(fail) * cfg["spc"]["baseline_frac"]))
    early, late = fail[:cut], fail[cut:]
    drift = two_proportion_test(int(early.sum()), len(early), int(late.sum()), len(late))
    drift["significant"] = drift["p_value"] < m["drift_alpha"]
    drift["early_end"] = raw["timestamp"].iloc[cut - 1]
    drift["late_start"] = raw["timestamp"].iloc[cut]
    return {
        "overall": ctq_metrics(len(fail), int(fail.sum()), m["opportunities_per_run"], m["sigma_shift"]),
        "early": ctq_metrics(len(early), int(early.sum()), m["opportunities_per_run"], m["sigma_shift"]),
        "late": ctq_metrics(len(late), int(late.sum()), m["opportunities_per_run"], m["sigma_shift"]),
        "drift": drift,
    }
