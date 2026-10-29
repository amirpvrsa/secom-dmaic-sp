"""Build everything with one command:  python run.py   (add --refresh to re-download)

Stages built so far: data -> clean -> data-quality tables.
Analyze, SPC, capability and the HTML report are added in later sessions.
"""
from __future__ import annotations

import argparse

from src.clean import clean, dataset_overview
from src.config import load_config, resolve
from src.data import load_secom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and re-fetch")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    out = cfg["outputs"]
    out_dir = resolve(out["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_secom(cfg, refresh=args.refresh)
    overview = dataset_overview(df)
    result = clean(df, cfg)

    overview.to_csv(out_dir / out["overview"], index=False)
    result.summary.to_csv(out_dir / out["quality_summary"], index=False)
    result.by_sensor.to_csv(out_dir / out["quality_by_sensor"], index=False)
    result.data.to_csv(out_dir / out["clean_data"], index=False)
    result.missing_mask.to_csv(out_dir / out["missing_mask"], index=False)

    print("Data overview\n" + overview.to_string(index=False))
    print("\nData-quality summary\n" + result.summary.to_string(index=False))
    print(f"\nOutputs written to {out_dir}")


if __name__ == "__main__":
    main()
