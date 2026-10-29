"""Build everything with one command:  python run.py   (add --refresh to re-download)

Pipeline: data -> clean -> analyze -> spc -> capability -> report.
"""
from __future__ import annotations

import argparse
import json

from src.analyze import plot_pareto, run_analysis
from src.capability import run_capability
from src.clean import clean, dataset_overview
from src.config import load_config, resolve
from src.data import load_secom, sensor_columns
from src.report import build_context, render
from src.spc import run_spc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and re-fetch")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    out = cfg["outputs"]
    out_dir = resolve(out["dir"])
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Define / Measure
    raw = load_secom(cfg, refresh=args.refresh)          # time-sorted, NOT imputed
    overview = dataset_overview(raw)
    cleaning = clean(raw, cfg)
    kept = list(cleaning.missing_mask.columns)
    print(f"[measure] {len(raw)} runs, {len(kept)} sensors kept")

    # Analyze (uses raw values: gaps handled inside each method, never the global imputation)
    analysis = run_analysis(raw, kept, sensor_columns(raw), cfg)
    analysis["pareto_figure"] = fig_dir / "pareto.png"
    plot_pareto(analysis["pareto"], analysis["pareto_figure"])
    print(f"[analyze] held-out ROC AUC {analysis['metrics']['test']['roc_auc']:.3f}")

    # Control
    sensors = analysis["ranking"][: cfg["spc"]["n_sensors"]]
    spc = run_spc(raw, sensors, cfg, fig_dir)
    capability = run_capability(raw, sensors, cfg)
    print(f"[control] charted {', '.join(sensors)}")

    # Tables for anyone who wants the numbers behind the report
    overview.to_csv(out_dir / out["overview"], index=False)
    cleaning.summary.to_csv(out_dir / out["quality_summary"], index=False)
    cleaning.by_sensor.to_csv(out_dir / out["quality_by_sensor"], index=False)
    cleaning.data.to_csv(out_dir / out["clean_data"], index=False)
    cleaning.missing_mask.to_csv(out_dir / out["missing_mask"], index=False)
    analysis["screen"].to_csv(out_dir / "screening.csv", index=False)
    analysis["missingness"].to_csv(out_dir / "missingness.csv", index=False)
    analysis["pareto"].to_csv(out_dir / "pareto.csv", index=False)
    analysis["cv"].to_csv(out_dir / "model_cv.csv", index=False)
    spc["summary"].to_csv(out_dir / "spc_summary.csv", index=False)
    spc["p_chart"].to_csv(out_dir / "p_chart.csv", index=False)
    capability.to_csv(out_dir / "capability.csv", index=False)
    (out_dir / "model_metrics.json").write_text(json.dumps(
        {**analysis["metrics"], "best_C": analysis["best_C"], "n_selected": analysis["n_selected"]},
        indent=2))

    path = render(build_context(cfg, overview, cleaning, analysis, spc, capability),
                  out_dir / cfg["report"]["file"])
    print(f"[report] {path}")


if __name__ == "__main__":
    main()
