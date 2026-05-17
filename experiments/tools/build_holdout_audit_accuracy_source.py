#!/usr/bin/env python3
"""Build a split-aware accuracy source for holdout auditing."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _normalize_eval_workload(value: str) -> str:
    text = str(value or "").strip()
    if text.endswith("_eval") or text.endswith("_holdout"):
        return text
    return f"{text}_eval"


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge eval/holdout accuracy CSVs for holdout audit input.")
    parser.add_argument("--eval_accuracy_csv", type=Path, required=True)
    parser.add_argument("--holdout_accuracy_csv", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    args = parser.parse_args()

    eval_df = pd.read_csv(args.eval_accuracy_csv)
    holdout_df = pd.read_csv(args.holdout_accuracy_csv)

    required = {"E0", "E6"}
    eval_df = eval_df.loc[eval_df["experiment_id"].isin(required)].copy()
    holdout_df = holdout_df.loc[holdout_df["experiment_id"].isin(required)].copy()

    eval_df["split"] = "eval"
    eval_df["workload"] = eval_df["workload"].map(_normalize_eval_workload)
    holdout_df["split"] = "holdout"

    out_df = pd.concat([eval_df, holdout_df], ignore_index=True, sort=False)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
