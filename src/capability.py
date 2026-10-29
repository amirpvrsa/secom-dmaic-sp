"""Process capability: Cp, Cpk, Pp, Ppk.

    Cp  = (USL - LSL) / (6 * sigma_within)
    Cpk = min(USL - mean, mean - LSL) / (3 * sigma_within)
    Pp  = (USL - LSL) / (6 * sigma_overall)
    Ppk = min(USL - mean, mean - LSL) / (3 * sigma_overall)

    sigma_within  = MR-bar / d2   (short-term: run-to-run variation only)
    sigma_overall = sample standard deviation, ddof = 1 (long-term: includes drift)

ASSUMPTION (stated in the report): SECOM publishes no specification limits.
LSL/USL here are DERIVED from passing runs in the baseline window. Because the
limits come from the data, the baseline's own capability is ~1 by construction
(for the mean_sd method); the informative number is how the LATER runs compare.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.spc import moving_ranges


def derive_spec_limits(baseline_pass: np.ndarray, cfg_c: dict) -> tuple[float, float]:
    x = np.asarray(baseline_pass, dtype=float)
    x = x[~np.isnan(x)]
    if cfg_c["spec_method"] == "mean_sd":
        m, s, k = x.mean(), x.std(ddof=1), cfg_c["mean_sd_k"]
        return m - k * s, m + k * s
    if cfg_c["spec_method"] == "percentile":
        return (float(np.percentile(x, cfg_c["lower_percentile"])),
                float(np.percentile(x, cfg_c["upper_percentile"])))
    raise ValueError(f"Unknown spec_method {cfg_c['spec_method']!r}")


def capability_indices(x, lsl: float, usl: float, d2: float = 1.128) -> dict:
    """x must be in time order (sigma_within uses consecutive moving ranges)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    mean = x.mean()
    s_within = moving_ranges(x).mean() / d2
    s_overall = x.std(ddof=1)
    width = usl - lsl
    nearest = min(usl - mean, mean - lsl)

    def ratio(num, den):  # a sensor with zero spread has no meaningful index
        return num / den if den > 0 else np.nan

    return {
        "n": len(x), "mean": mean, "sigma_within": s_within, "sigma_overall": s_overall,
        "Cp": ratio(width, 6 * s_within), "Cpk": ratio(nearest, 3 * s_within),
        "Pp": ratio(width, 6 * s_overall), "Ppk": ratio(nearest, 3 * s_overall),
        "pct_outside": float(((x < lsl) | (x > usl)).mean()),
    }


def anderson_darling_p(x) -> float:
    """p-value of the Anderson-Darling normality test (small p = not normal)."""
    from statsmodels.stats.diagnostic import normal_ad
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 8 or np.ptp(x) == 0:
        return np.nan
    return float(normal_ad(x)[1])


def percentile_indices(x, lsl: float, usl: float, lo_pct: float = 0.135, hi_pct: float = 99.865) -> dict:
    """Non-normal (percentile / ISO 22514 style) performance indices.

        Pp_pct  = (USL - LSL) / (P_hi - P_lo)
        Ppk_pct = min( (USL - median) / (P_hi - median), (median - LSL) / (median - P_lo) )

    P_lo, P_hi = the 0.135th and 99.865th percentiles, which bracket 99.73% of the data,
    the same share that +/-3 sigma covers for a normal distribution. No normality assumed.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    p_lo, med, p_hi = np.percentile(x, [lo_pct, 50, hi_pct])
    upper = (usl - med) / (p_hi - med) if p_hi > med else np.nan
    lower = (med - lsl) / (med - p_lo) if med > p_lo else np.nan
    return {"Pp_pct": (usl - lsl) / (p_hi - p_lo) if p_hi > p_lo else np.nan,
            "Ppk_pct": float(np.nanmin([upper, lower])) if not (np.isnan(upper) and np.isnan(lower)) else np.nan}


def run_capability(raw: pd.DataFrame, spc_charts: dict, cfg: dict) -> pd.DataFrame:
    """Capability per priority sensor. The baseline window for each sensor is the one the
    SPC step chose, so the control chart and the capability study use the same reference."""
    c, s = cfg["capability"], cfg["spc"]
    pos = np.arange(len(raw))
    rows = []
    for sensor, chart in spc_charts.items():
        start, end = chart["baseline_pos"]
        in_win = (pos >= start) & (pos <= end)
        passing = raw["fail"].to_numpy() == 0
        lsl, usl = derive_spec_limits(raw.loc[in_win & passing, sensor].to_numpy(), c)
        for label, mask in (("baseline window, passing runs", in_win & passing),
                            ("baseline window, failing runs", in_win & ~passing),
                            ("outside baseline window, all runs", ~in_win)):
            vals = raw.loc[mask, sensor].to_numpy()
            if np.sum(~np.isnan(vals)) < 8:
                continue
            ad_p = anderson_darling_p(vals)
            rows.append({"sensor": sensor, "period": label, "lsl_assumed": lsl, "usl_assumed": usl,
                         **capability_indices(vals, lsl, usl, s["d2"]),
                         "ad_p": ad_p, "normal": bool(ad_p >= c["normality_alpha"]) if not np.isnan(ad_p) else False,
                         **percentile_indices(vals, lsl, usl, c["lower_percentile"], c["upper_percentile"])})
    return pd.DataFrame(rows)
