"""Analyze phase: which sensors differ between passing and failing runs?

Four parts, all observational (they find ASSOCIATIONS, not causes):
  1. Univariate screening: per sensor, Welch t-test and Mann-Whitney U on MEASURED
     values only (imputed values excluded), with Benjamini-Hochberg correction.
  2. Missingness screening: does a sensor being missing go with a different fail rate?
  3. L1 logistic regression: all kept sensors together, imputation fitted inside the
     model on training runs only, time-ordered holdout for an honest test score.
  4. Stability + Pareto: how often each sensor is selected across bootstrap refits,
     and a Pareto of the model's weight across sensors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------- multiple testing

def benjamini_hochberg(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values). NaNs stay NaN and are not counted.

    Sort the m p-values ascending; the i-th smallest gets q = p * m / i; then make the
    q-values non-decreasing from the largest down (a running minimum), capped at 1.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


# ---------------------------------------------------------------- 1. univariate screen

def screen_sensors(X: pd.DataFrame, fail: pd.Series, min_group_n: int, fdr_q: float) -> pd.DataFrame:
    """Compare passing vs failing runs sensor by sensor, using measured values only."""
    fail = fail.astype(bool).to_numpy()
    rows = []
    for col in X.columns:
        v = X[col].to_numpy(dtype=float)
        a, b = v[~fail], v[fail]                 # a = pass, b = fail
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        row = {"sensor": col, "n_pass": len(a), "n_fail": len(b),
               "mean_pass": np.nan, "mean_fail": np.nan,
               "median_pass": np.nan, "median_fail": np.nan,
               "effect_size_d": np.nan, "welch_p": np.nan, "mw_p": np.nan}
        if len(a) >= min_group_n and len(b) >= min_group_n:
            row.update(mean_pass=a.mean(), mean_fail=b.mean(),
                       median_pass=np.median(a), median_fail=np.median(b))
            sa, sb = a.std(ddof=1), b.std(ddof=1)
            pooled = np.sqrt(((len(a) - 1) * sa**2 + (len(b) - 1) * sb**2) / (len(a) + len(b) - 2))
            if pooled > 0:
                row["effect_size_d"] = (b.mean() - a.mean()) / pooled
                row["welch_p"] = stats.ttest_ind(b, a, equal_var=False).pvalue
            if np.ptp(np.concatenate([a, b])) > 0:
                row["mw_p"] = stats.mannwhitneyu(b, a, alternative="two-sided").pvalue
        rows.append(row)
    out = pd.DataFrame(rows)
    out["welch_q"] = benjamini_hochberg(out["welch_p"])
    out["mw_q"] = benjamini_hochberg(out["mw_p"])
    out["significant_both"] = (out["welch_q"] < fdr_q) & (out["mw_q"] < fdr_q)
    return out.sort_values("mw_p", na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------- 2. missingness screen

def missingness_screen(X_all: pd.DataFrame, fail: pd.Series, min_count: int, fdr_q: float) -> pd.DataFrame:
    """Fisher's exact test: fail rate when the sensor is missing vs when it is measured.

    Sensors with an identical missing pattern (missing in exactly the same runs) get the
    same pattern_id, because they are really one finding, not many.
    """
    f = fail.astype(bool).to_numpy()
    patterns: dict[bytes, int] = {}
    rows = []
    for col in X_all.columns:
        miss = X_all[col].isna().to_numpy()
        n_miss = int(miss.sum())
        if n_miss < min_count or n_miss == len(miss):
            continue
        key = np.packbits(miss).tobytes()
        pid = patterns.setdefault(key, len(patterns) + 1)
        a = int((miss & f).sum()); b = int((miss & ~f).sum())     # missing: fail, pass
        c = int((~miss & f).sum()); d = int((~miss & ~f).sum())   # measured: fail, pass
        rows.append({"sensor": col, "pattern_id": pid, "n_missing": n_miss,
                     "fail_rate_missing": a / (a + b), "fail_rate_measured": c / (c + d),
                     "fisher_p": stats.fisher_exact([[a, b], [c, d]]).pvalue})
    out = pd.DataFrame(rows, columns=["sensor", "pattern_id", "n_missing", "fail_rate_missing",
                                      "fail_rate_measured", "fisher_p"])
    out["fisher_q"] = benjamini_hochberg(out["fisher_p"])
    out["significant"] = out["fisher_q"] < fdr_q
    return out.sort_values("fisher_p").reset_index(drop=True)


def missing_patterns_summary(miss: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct missing pattern among significant sensors."""
    sig = miss[miss["significant"]]
    if sig.empty:
        return pd.DataFrame(columns=["pattern_id", "n_sensors", "n_missing", "fail_rate_missing",
                                     "fail_rate_measured", "fisher_q", "example_sensors"])
    g = sig.groupby("pattern_id")
    return pd.DataFrame({
        "n_sensors": g.size(),
        "n_missing": g["n_missing"].first(),
        "fail_rate_missing": g["fail_rate_missing"].first(),
        "fail_rate_measured": g["fail_rate_measured"].first(),
        "fisher_q": g["fisher_q"].min(),
        "example_sensors": g["sensor"].apply(lambda s: ", ".join(s.head(4))),
    }).reset_index().sort_values("fisher_q").reset_index(drop=True)


# ---------------------------------------------------------------- 3. L1 logistic model

def time_split(n: int, holdout_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Rows are already in time order: first part trains, last part tests."""
    cut = int(round(n * (1 - holdout_frac)))
    return np.arange(cut), np.arange(cut, n)


def make_pipeline(C: float, class_weight, seed: int) -> Pipeline:
    # Imputer and scaler live INSIDE the pipeline: when the pipeline is fitted on
    # training runs, medians/means/SDs come from training runs only.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=C, l1_ratio=1.0, solver="liblinear",
                                     class_weight=class_weight, random_state=seed,
                                     max_iter=5000)),
    ])


def fit_l1_model(X: pd.DataFrame, y: pd.Series, cfg_a: dict) -> tuple[Pipeline, pd.DataFrame]:
    """Choose C by time-ordered cross-validation, then refit on all training runs."""
    search = GridSearchCV(
        make_pipeline(1.0, cfg_a["class_weight"], cfg_a["random_seed"]),
        {"model__C": cfg_a["C_grid"]},
        cv=TimeSeriesSplit(n_splits=cfg_a["cv_folds"]),
        scoring=cfg_a["cv_scoring"], refit=True,
    )
    search.fit(X, y)
    cv = pd.DataFrame({"C": search.cv_results_["param_model__C"].astype(float),
                       "cv_mean": search.cv_results_["mean_test_score"],
                       "cv_sd": search.cv_results_["std_test_score"]})
    return search.best_estimator_, cv


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    has_both = y.nunique() == 2
    return {
        "n_runs": int(len(y)), "n_fail": int(y.sum()), "fail_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, prob)) if has_both else np.nan,
        "average_precision": float(average_precision_score(y, prob)) if has_both else np.nan,
        "recall_at_0.5": float(recall_score(y, pred, zero_division=0)),
        "precision_at_0.5": float(precision_score(y, pred, zero_division=0)),
        "flagged_at_0.5": int(pred.sum()),
    }


def coefficients(model: Pipeline, names: list[str]) -> pd.Series:
    """Coefficients on the standardized scale: comparable across sensors."""
    return pd.Series(model.named_steps["model"].coef_.ravel(), index=names, name="coef")


def selection_frequency(X: pd.DataFrame, y: pd.Series, C: float, cfg_a: dict) -> pd.Series:
    """Fraction of bootstrap refits in which each sensor gets a non-zero coefficient.

    L1 picks one sensor out of a group of correlated ones somewhat arbitrarily, so a
    sensor chosen in 90% of resamples is a much firmer result than one chosen in 20%.
    """
    rng = np.random.default_rng(cfg_a["random_seed"])
    counts = np.zeros(X.shape[1])
    n = len(y)
    done = 0
    for _ in range(cfg_a["bootstrap_n"]):
        idx = rng.integers(0, n, n)
        yb = y.iloc[idx]
        if yb.nunique() < 2:
            continue
        m = make_pipeline(C, cfg_a["class_weight"], cfg_a["random_seed"]).fit(X.iloc[idx], yb)
        counts += m.named_steps["model"].coef_.ravel() != 0
        done += 1
    return pd.Series(counts / max(done, 1), index=X.columns, name="selection_freq")


# ---------------------------------------------------------------- 4. Pareto + ranking

def pareto_table(coef: pd.Series, freq: pd.Series, top_n: int) -> pd.DataFrame:
    """Sensors ordered by |coefficient|, with cumulative share of total |coefficient|."""
    t = pd.DataFrame({"coef": coef, "abs_coef": coef.abs(), "selection_freq": freq})
    t = t[t["abs_coef"] > 0].sort_values("abs_coef", ascending=False)
    total = t["abs_coef"].sum()
    t["share"] = t["abs_coef"] / total if total else np.nan
    t["cum_share"] = t["share"].cumsum()
    t["direction"] = np.where(t["coef"] > 0, "higher value, more fails", "lower value, more fails")
    return t.head(top_n).rename_axis("sensor").reset_index()


def rank_sensors(pareto: pd.DataFrame, screen: pd.DataFrame) -> list[str]:
    """Order of sensors for SPC charts. The strongest evidence comes first:
      1. significant in BOTH univariate tests AND selected by the model (by selection freq)
      2. significant in both univariate tests only (by Mann-Whitney q-value)
      3. selected by the model only (by selection freq)
      4. everything else, by Mann-Whitney q-value
    """
    sig = screen.loc[screen["significant_both"]].sort_values("mw_q")["sensor"].tolist()
    model = pareto.sort_values(["selection_freq", "abs_coef"], ascending=False)["sensor"].tolist()
    tier1 = [s for s in model if s in sig]
    tier2 = [s for s in sig if s not in tier1]
    tier3 = [s for s in model if s not in sig]
    # 4. if nothing above, fall back to the strongest univariate results so SPC still has input
    tier4 = [s for s in screen.sort_values("mw_q")["sensor"] if s not in sig and s not in model]
    return tier1 + tier2 + tier3 + tier4


def plot_pareto(pareto: pd.DataFrame, path) -> None:
    """Bars = each sensor's share of total model weight; line = cumulative share.
    Both on the same 0-100% axis (no second y-axis)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    blue, ink, muted, grid = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(pareto))
    ax.bar(x, pareto["share"], color=blue, width=0.7, label="Share of model weight")
    ax.plot(x, pareto["cum_share"], color=ink, marker="o", ms=4, lw=1.5, label="Cumulative share")
    for i, f in enumerate(pareto["selection_freq"]):
        ax.text(i + 0.38, pareto["share"].iloc[i] + 0.015, f"{f:.0%}", ha="right", fontsize=7, color=muted)
    ax.set_xticks(x, pareto["sensor"], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color=grid, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=muted, labelsize=8)
    ax.set_title("Pareto of model weight (label above bar = bootstrap selection frequency)",
                 loc="left", fontsize=11, color=ink)
    ax.legend(fontsize=8, frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- orchestration

def run_analysis(raw: pd.DataFrame, kept: list[str], all_sensors: list[str], cfg: dict) -> dict:
    """raw: time-sorted table with NaNs (not imputed). kept: sensors that passed cleaning."""
    a = cfg["analyze"]
    y = raw["fail"].astype(int)
    X = raw[kept]

    screen = screen_sensors(X, y, a["min_group_n"], a["fdr_q"])
    miss = missingness_screen(raw[all_sensors], y, a["missing_min_count"], a["fdr_q"])
    patterns = missing_patterns_summary(miss)

    tr, te = time_split(len(raw), a["holdout_frac"])
    model, cv = fit_l1_model(X.iloc[tr], y.iloc[tr], a)
    best_C = float(model.named_steps["model"].C)
    metrics = {"train": evaluate(model, X.iloc[tr], y.iloc[tr]),
               "test": evaluate(model, X.iloc[te], y.iloc[te])}
    coef = coefficients(model, kept)
    freq = selection_frequency(X.iloc[tr], y.iloc[tr], best_C, a)
    pareto = pareto_table(coef, freq, a["pareto_top_n"])

    return {
        "screen": screen, "missingness": miss, "missing_patterns": patterns,
        "cv": cv, "best_C": best_C, "metrics": metrics, "coef": coef, "freq": freq,
        "pareto": pareto, "ranking": rank_sensors(pareto, screen),
        "n_selected": int((coef != 0).sum()),
        "raw_fail": y.to_numpy(),
        "split": {"train_end": raw["timestamp"].iloc[tr[-1]], "test_start": raw["timestamp"].iloc[te[0]],
                  "n_train": len(tr), "n_test": len(te)},
    }
