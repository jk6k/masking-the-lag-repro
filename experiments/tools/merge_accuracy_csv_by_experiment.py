#!/usr/bin/env python3
"""Merge accuracy CSVs by replacing experiment rows with later overrides."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing CSV: {path}")
    df = pd.read_csv(path)
    if "experiment_id" not in df.columns:
        raise SystemExit(f"CSV is missing experiment_id column: {path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge accuracy CSVs by experiment_id precedence.")
    parser.add_argument("--base_csv", type=Path, required=True)
    parser.add_argument("--override_csv", type=Path, action="append", default=[])
    parser.add_argument("--keep_experiments", default=None, help="Optional comma-separated experiment IDs to keep.")
    parser.add_argument("--out_csv", type=Path, required=True)
    args = parser.parse_args()

    merged = _read_csv(args.base_csv)
    for override_csv in args.override_csv:
        override = _read_csv(override_csv)
        override_experiments = set(override["experiment_id"].astype(str))
        merged = merged[~merged["experiment_id"].astype(str).isin(override_experiments)].copy()
        merged = pd.concat([merged, override], ignore_index=True)

    if args.keep_experiments:
        keep = {item.strip() for item in args.keep_experiments.split(",") if item.strip()}
        merged = merged[merged["experiment_id"].astype(str).isin(keep)].copy()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"Wrote merged CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
