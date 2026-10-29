"""Render the single-page DMAIC report.

Rule: no number is typed into the template. report.py hands the template the
computed tables and values; the template only formats them. Sentences whose
meaning depends on a result (e.g. "the model does / does not hold up on later
runs") are chosen here from the numbers, with the threshold in config.yaml.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _img(path) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def pct(x, digits=1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.{digits}f}%"


def num(x, digits=3) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    ax = abs(x)
    if ax != 0 and (ax < 10 ** -digits or ax >= 1e6):
        return f"{x:.{digits}g}"
    return f"{x:,.{digits}f}"


def sig(x) -> str:
    """p/q-values: scientific notation when tiny."""
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:.2e}" if x < 1e-3 else f"{x:.3f}"


def duration(td) -> str:
    if td is None or pd.isna(td):
        return "n/a"
    mins = pd.Timedelta(td).total_seconds() / 60
    return f"{mins:.0f} min" if mins < 120 else f"{mins / 60:.1f} h"


def overview_value(metric: str, value) -> str:
    if isinstance(value, pd.Timedelta):
        return duration(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M")
    if "rate" in metric or "%" in metric:
        return pct(value, 2)
    return num(value)


PERIOD_WORDS = {"D": "day", "W": "week", "MS": "month", "ME": "month", "M": "month"}


def build_context(cfg, overview, cleaning, analysis, spc, capability) -> dict:
    a, rep = analysis, cfg["report"]
    ov = overview.set_index("metric")["value"]
    screen = a["screen"]
    tested = screen["welch_p"].notna() | screen["mw_p"].notna()
    test = a["metrics"]["test"]
    q = cfg["analyze"]["fdr_q"]

    # result-dependent wording, decided by thresholds in config
    holds_up = (not pd.isna(test["roc_auc"])) and test["roc_auc"] >= rep["min_useful_auc"]
    unstable = spc["summary"][spc["summary"]["baseline_stability_ratio"] > rep["max_stability_ratio"]]

    top_sensors = a["ranking"][: cfg["spc"]["n_sensors"]]
    hyp = screen.set_index("sensor").loc[[s for s in top_sensors if s in set(screen["sensor"])]]

    cut = int(round(len(a["raw_fail"]) * cfg["spc"]["baseline_frac"]))
    fail_early, fail_late = float(a["raw_fail"][:cut].mean()), float(a["raw_fail"][cut:].mean())

    return {
        "fail_early": fail_early, "fail_late": fail_late,
        "cfg": cfg, "title": rep["title"],
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "ov": ov, "overview": overview,
        "quality": cleaning.summary, "n_kept": int((cleaning.by_sensor["decision"] == "keep").sum()),
        "n_imputed_pct": cleaning.summary["cells_imputed_pct_of_kept"].iloc[-1],
        "screen_top": screen[tested].head(rep["screen_rows"]),
        "n_tested": int(tested.sum()),
        "n_sig_welch": int((screen["welch_q"] < q).sum()),
        "n_sig_mw": int((screen["mw_q"] < q).sum()),
        "n_sig_both": int(screen["significant_both"].sum()),
        "fdr_q": q,
        "miss_patterns": a["missing_patterns"],
        "n_miss_tested": len(a["missingness"]),
        "cv": a["cv"], "best_C": a["best_C"], "n_selected": a["n_selected"],
        "train": a["metrics"]["train"], "test": test, "split": a["split"],
        "holds_up": holds_up,
        "pareto": a["pareto"],
        "top_sensors": top_sensors, "hyp": hyp,
        "spc": spc["summary"], "unstable": unstable,
        "pchart": spc["p_chart"], "n_p_out": int(spc["p_chart"]["out"].sum()),
        "cap": capability,
        "img_pareto": _img(a["pareto_figure"]),
        "img_p": _img(spc["p_chart_figure"]),
        "img_imr": {s: _img(r["figure"]) for s, r in spc["charts"].items()},
        "period_word": PERIOD_WORDS.get(cfg["spc"]["p_chart_period"], cfg["spc"]["p_chart_period"]),
        "n_flag_lower": int((spc["summary"]["fail_rate_flagged"] < spc["summary"]["fail_rate_not_flagged"]).sum()),
        "pct": pct, "num": num, "sig": sig, "duration": duration, "overview_value": overview_value,
    }


def render(context: dict, out_path) -> Path:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), undefined=StrictUndefined,
                      autoescape=True)
    html = env.get_template("report.html.j2").render(**context)
    out_path = Path(out_path)
    out_path.write_text(html, encoding="utf-8")
    return out_path
