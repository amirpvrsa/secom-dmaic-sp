"""Control phase: Individuals and Moving Range (I-MR) charts with Western Electric rules.

How the limits are set (Phase I -> Phase II):
  * Baseline = PASSING runs in the first `baseline_frac` of the timeline, measured values only.
  * From the baseline:   X-bar = mean,  MR-bar = mean of |x_i - x_(i-1)|,
                         sigma_within = MR-bar / d2      (d2 = 1.128 for ranges of 2 points)
                         I chart:  UCL/LCL = X-bar +/- 3 * sigma_within
                         MR chart: UCL = D4 * MR-bar (D4 = 3.267), LCL = 0
  * Those fixed limits are then applied to every run (pass and fail) in time order.
Missing readings are skipped: the chart shows the runs where the sensor was measured.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# chart colors (reference palette: slot 1 blue, slot 2 orange; ink for text/limits)
C_PASS, C_FAIL, C_INK, C_MUTED, C_GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


@dataclass
class IMRLimits:
    center: float
    sigma: float      # within-subgroup sigma estimated as MR-bar / d2
    ucl: float
    lcl: float
    mr_bar: float
    mr_ucl: float
    n_baseline: int


def moving_ranges(x: np.ndarray) -> np.ndarray:
    """|x_i - x_(i-1)| for consecutive points. One fewer value than x."""
    x = np.asarray(x, dtype=float)
    return np.abs(np.diff(x))


def imr_limits(baseline: np.ndarray, d2: float, D4: float, k: float) -> IMRLimits:
    x = np.asarray(baseline, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        raise ValueError("Need at least 2 baseline points for I-MR limits")
    mr_bar = moving_ranges(x).mean()
    sigma = mr_bar / d2
    c = x.mean()
    return IMRLimits(center=c, sigma=sigma, ucl=c + k * sigma, lcl=c - k * sigma,
                     mr_bar=mr_bar, mr_ucl=D4 * mr_bar, n_baseline=len(x))


def western_electric(x: np.ndarray, center: float, sigma: float, rules=(1, 2, 3, 4)) -> pd.DataFrame:
    """Flag each point that COMPLETES a Western Electric pattern.

    Rule 1: the point is beyond 3 sigma.
    Rule 2: 2 of the last 3 points beyond 2 sigma on the same side (this point one of them).
    Rule 3: 4 of the last 5 points beyond 1 sigma on the same side (this point one of them).
    Rule 4: the last 8 points all on the same side of the center line.
    """
    x = np.asarray(x, dtype=float)
    z = (x - center) / sigma if sigma > 0 else np.zeros_like(x)
    n = len(z)
    out = pd.DataFrame(False, index=range(n), columns=[f"rule{r}" for r in rules])

    def k_of_m(threshold: float, k: int, m: int) -> np.ndarray:
        hit = np.zeros(n, dtype=bool)
        for side in (1, -1):
            beyond = side * z > threshold
            for i in range(m - 1, n):
                if beyond[i] and beyond[i - m + 1:i + 1].sum() >= k:
                    hit[i] = True
        return hit

    if 1 in rules:
        out["rule1"] = np.abs(z) > 3
    if 2 in rules:
        out["rule2"] = k_of_m(2, 2, 3)
    if 3 in rules:
        out["rule3"] = k_of_m(1, 4, 5)
    if 4 in rules:
        hit = np.zeros(n, dtype=bool)
        for side in (1, -1):
            same = side * z > 0
            for i in range(7, n):
                if same[i - 7:i + 1].all():
                    hit[i] = True
        out["rule4"] = hit
    out["any"] = out.any(axis=1)
    return out


def chart_sensor(df: pd.DataFrame, sensor: str, cfg_s: dict) -> dict:
    """Build the I-MR result for one sensor. df must be time-sorted and NOT imputed."""
    # baseline = the first baseline_frac of ALL runs by position (same rule capability.py uses)
    cut = int(round(len(df) * cfg_s["baseline_frac"]))
    d = df[["run_id", "timestamp", "fail", sensor]].copy()
    d["in_base"] = np.arange(len(d)) < cut
    d = d.dropna(subset=[sensor]).reset_index(drop=True)
    cut_time = df["timestamp"].iloc[cut - 1]
    in_base = d.pop("in_base")
    base = d.loc[in_base & (d["fail"] == 0), sensor].to_numpy()
    lim = imr_limits(base, cfg_s["d2"], cfg_s["D4"], cfg_s["sigma_multiplier"])

    x = d[sensor].to_numpy()
    flags = western_electric(x, lim.center, lim.sigma, tuple(cfg_s["rules"]))
    mr = np.concatenate([[np.nan], moving_ranges(x)])
    d = pd.concat([d, flags], axis=1)
    d["mr"] = mr
    d["mr_out"] = d["mr"] > lim.mr_ucl
    d["baseline"] = in_base

    after = d[~in_base]
    flagged = d[d["any"]]
    # Phase I check: if the baseline's overall SD is much larger than its within SD
    # (MR-bar/d2), the baseline itself was not stable, so its limits are suspect.
    sd_overall = float(np.std(base, ddof=1))
    summary = {
        "sensor": sensor, "center": lim.center, "sigma_within": lim.sigma,
        "baseline_sd_overall": sd_overall,
        "baseline_stability_ratio": sd_overall / lim.sigma if lim.sigma > 0 else np.nan,
        "lcl": lim.lcl, "ucl": lim.ucl, "mr_bar": lim.mr_bar, "mr_ucl": lim.mr_ucl,
        "n_baseline": lim.n_baseline, "n_points": len(d),
        **{f"n_{c}": int(d[c].sum()) for c in flags.columns if c != "any"},
        "n_any_rule": int(d["any"].sum()),
        "n_mr_out": int(d["mr_out"].sum()),
        "pct_after_baseline_beyond_3sigma": float(after["rule1"].mean()) if len(after) else np.nan,
        "fail_rate_flagged": float(flagged["fail"].mean()) if len(flagged) else np.nan,
        "fail_rate_not_flagged": float(d.loc[~d["any"], "fail"].mean()),
    }
    return {"points": d, "limits": lim, "summary": summary, "baseline_end": cut_time}


def plot_imr(res: dict, path) -> None:
    d, lim = res["points"], res["limits"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    sensor = res["summary"]["sensor"]
    for ax in (ax1, ax2):
        ax.axvspan(d["timestamp"].min(), res["baseline_end"], color=C_GRID, alpha=0.6, lw=0)
        ax.grid(axis="y", color=C_GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=C_MUTED, labelsize=8)

    passed, failed = d[d["fail"] == 0], d[d["fail"] == 1]
    ax1.plot(d["timestamp"], d[sensor], color=C_PASS, lw=0.6, alpha=0.5, zorder=1)
    ax1.scatter(passed["timestamp"], passed[sensor], s=8, color=C_PASS, label="Passing run", zorder=2)
    ax1.scatter(failed["timestamp"], failed[sensor], s=28, color=C_FAIL, marker="x",
                linewidths=1.5, label="Failing run", zorder=3)
    # only zone-A signals (rules 1-2) are ringed; rules 3-4 can flag hundreds of points
    # after a shift and would bury the chart. All four rules are counted in the table.
    zone_a = d[d.get("rule1", False) | d.get("rule2", False)]
    ax1.scatter(zone_a["timestamp"], zone_a[sensor], s=70, facecolors="none", edgecolors=C_INK,
                linewidths=0.9, label="Rule 1 or 2 signal", zorder=4)
    for y, ls in ((lim.ucl, "--"), (lim.center, "-"), (lim.lcl, "--")):
        ax1.axhline(y, color=C_MUTED, ls=ls, lw=1)
    ax1.set_ylabel("Individual value", color=C_MUTED, fontsize=9)
    ax1.set_title(f"{sensor}: I-MR chart (shaded = baseline window)", loc="left",
                  fontsize=11, color=C_INK)
    ax1.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)

    ax2.plot(d["timestamp"], d["mr"], color=C_PASS, lw=0.7)
    ax2.axhline(lim.mr_ucl, color=C_MUTED, ls="--", lw=1)
    ax2.axhline(lim.mr_bar, color=C_MUTED, ls="-", lw=1)
    ax2.set_ylabel("Moving range", color=C_MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- p-chart of fail rate

def p_chart(raw: pd.DataFrame, period: str, k: float = 3) -> pd.DataFrame:
    """Fail proportion per time period, with limits that vary with each period's size.

        p-bar   = total fails / total runs           (over all periods)
        UCL_i   = p-bar + k * sqrt(p-bar * (1 - p-bar) / n_i),   LCL_i = max(0, p-bar - ...)
    where n_i is the number of runs in period i.
    """
    g = raw.set_index("timestamp")["fail"].resample(period)
    t = pd.DataFrame({"n": g.size(), "fails": g.sum()})
    t = t[t["n"] > 0].copy()
    p_bar = t["fails"].sum() / t["n"].sum()
    t["p"] = t["fails"] / t["n"]
    half = k * np.sqrt(p_bar * (1 - p_bar) / t["n"])
    t["p_bar"] = p_bar
    t["ucl"] = p_bar + half
    t["lcl"] = np.maximum(0.0, p_bar - half)
    t["out"] = (t["p"] > t["ucl"]) | (t["p"] < t["lcl"])
    return t.rename_axis("period_start").reset_index()


def plot_p_chart(t: pd.DataFrame, path, period_word: str = "period") -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.step(t["period_start"], t["ucl"], where="mid", color=C_MUTED, ls="--", lw=1, label="UCL / LCL")
    ax.step(t["period_start"], t["lcl"], where="mid", color=C_MUTED, ls="--", lw=1)
    ax.axhline(t["p_bar"].iloc[0], color=C_MUTED, lw=1, label="Average fail rate")
    ax.plot(t["period_start"], t["p"], color=C_PASS, lw=2, marker="o", ms=5, label="Fail rate")
    out = t[t["out"]]
    ax.scatter(out["period_start"], out["p"], s=90, facecolors="none", edgecolors=C_INK,
               lw=1.2, label="Outside limits", zorder=4)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", color=C_GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=C_MUTED, labelsize=8)
    ax.set_title(f"Fail rate per {period_word} (p-chart)", loc="left", fontsize=11, color=C_INK)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_spc(raw: pd.DataFrame, sensors: list[str], cfg: dict, fig_dir) -> dict:
    results, rows = {}, []
    for s in sensors:
        r = chart_sensor(raw, s, cfg["spc"])
        path = fig_dir / f"imr_{s}.png"
        plot_imr(r, path)
        r["figure"] = path
        results[s] = r
        rows.append(r["summary"])
    pc = p_chart(raw, cfg["spc"]["p_chart_period"], cfg["spc"]["sigma_multiplier"])
    words = {"D": "day", "W": "week", "MS": "month", "ME": "month", "M": "month"}
    plot_p_chart(pc, fig_dir / "p_chart.png", words.get(cfg["spc"]["p_chart_period"], "period"))
    return {"charts": results, "summary": pd.DataFrame(rows), "p_chart": pc,
            "p_chart_figure": fig_dir / "p_chart.png"}
