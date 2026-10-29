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


def run_capability(raw: pd.DataFrame, sensors: list[str], cfg: dict) -> pd.DataFrame:
    c, s = cfg["capability"], cfg["spc"]
    cut = int(round(len(raw) * s["baseline_frac"]))
    base, later = raw.iloc[:cut], raw.iloc[cut:]
    rows = []
    for sensor in sensors:
        lsl, usl = derive_spec_limits(base.loc[base["fail"] == 0, sensor].to_numpy(), c)
        for label, part in (("baseline, passing runs", base[base["fail"] == 0]),
                            ("after baseline, all runs", later),
                            ("after baseline, passing runs", later[later["fail"] == 0]),
                            ("after baseline, failing runs", later[later["fail"] == 1])):
            vals = part[sensor].to_numpy()
            if np.sum(~np.isnan(vals)) < 3:
                continue
            rows.append({"sensor": sensor, "period": label, "lsl_assumed": lsl,
                         "usl_assumed": usl, **capability_indices(vals, lsl, usl, s["d2"])})
    return pd.DataFrame(rows)
