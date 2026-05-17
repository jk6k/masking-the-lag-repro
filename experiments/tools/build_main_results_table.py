#!/usr/bin/env python3
"""Build the canonical Table 1 CSV/TeX assets from the unified main-claim summary."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

try:
    from experiments.tools.path_policy import MAIN_PROJECT_REPORT_DATA_DIR, MAIN_PROJECT_REPORT_TABLE_DIR
except ModuleNotFoundError:  # direct script execution
    from path_policy import MAIN_PROJECT_REPORT_DATA_DIR, MAIN_PROJECT_REPORT_TABLE_DIR


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAIN_CLAIM_CSV = MAIN_PROJECT_REPORT_DATA_DIR / "main_claim_unified_summary_20260310.csv"
DEFAULT_OUT_CSV = MAIN_PROJECT_REPORT_TABLE_DIR / "Table1_MainResults_E0_E6.csv"
DEFAULT_OUT_TEX = MAIN_PROJECT_REPORT_TABLE_DIR / "Table1_MainResults_E0_E6.tex"
ORDER = ["E0", "E2", "E3", "E4", "E6"]


def _format_accuracy_evidence(value: str) -> str:
    norm = str(value or "").strip().lower()
    if norm == "measured_full_eval":
        return "measured"
    if norm in {"modeled_full_eval", "proxy_config_conditioned"}:
        return "modeled (config-conditioned)"
    if norm == "shared_e0_full_eval_reference":
        return "shared E0 reference"
    return str(value or "")


def _format_reference_basis(value: str) -> str:
    text = str(value or "").strip()
    if "modeled config-conditioned full-eval" in text:
        return "modeled full-eval / local E0 quant"
    if "config-conditioned FP32 proxy" in text:
        return "config FP32 proxy / local E0 quant"
    if text.startswith("local E0 quant"):
        return "local E0 quant"
    if "config-conditioned FP32" in text:
        return "config FP32 / local E0 quant"
    return text


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "config",
        "latency_ms",
        "energy_j",
        "tops_w",
        "top1_value",
        "accuracy_evidence",
        "reference_basis",
        "speedup_vs_E0",
        "energy_reduction_pct_vs_E0",
        "acc_drop_pp_vs_E0",
        "tops_w_gain_vs_E0",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\caption{Retained headline chain on the local ImageNet eval split. Performance comes from paired \texttt{master\_metrics.csv} runs. Table entries explicitly report Top-1 evidence type and reference basis for each retained configuration.}",
        r"\label{tab:main_e0e6}",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{lrrrrll}",
        r"\toprule",
        r"Cfg & L(ms) & E(J) & TOPS/W & Top-1 (\%) & Accuracy evidence & Reference basis \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['config']} & "
            f"{float(row['latency_ms']):.3f} & "
            f"{float(row['energy_j']):.4f} & "
            f"{float(row['tops_w']):.3f} & "
            f"{float(row['top1_value']):.2f} & "
            f"{row['accuracy_evidence']} & "
            f"{row['reference_basis']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_rows(main_claim_csv: Path) -> list[dict[str, object]]:
    df = pd.read_csv(main_claim_csv)
    top1_col = "top1_value_mean" if "top1_value_mean" in df.columns else "top1_measured_mean"
    rows: list[dict[str, object]] = []
    for experiment_id in ORDER:
        subset = df.loc[df["experiment_id"] == experiment_id]
        if subset.empty:
            raise SystemExit(f"Missing experiment_id={experiment_id} in {main_claim_csv}")
        row = subset.iloc[0]
        rows.append(
            {
                "config": experiment_id,
                "latency_ms": float(row["latency_ms_mean"]),
                "energy_j": float(row["energy_j_mean"]),
                "tops_w": float(row["tops_w_mean"]),
                "top1_value": float(row[top1_col]),
                "accuracy_evidence": _format_accuracy_evidence(str(row["accuracy_evidence"])),
                "reference_basis": _format_reference_basis(str(row["baseline_definition"])),
                "speedup_vs_E0": float(row["speedup_vs_e0_mean"]),
                "energy_reduction_pct_vs_E0": float(row["energy_reduction_pct_vs_e0_mean"]),
                "acc_drop_pp_vs_E0": float(row["acc_drop_pp_vs_e0_mean"]),
                "tops_w_gain_vs_E0": float(row["tops_w_gain_vs_e0_mean"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical main-results Table 1 assets.")
    parser.add_argument("--main_claim_csv", type=Path, default=DEFAULT_MAIN_CLAIM_CSV)
    parser.add_argument("--out_csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out_tex", type=Path, default=DEFAULT_OUT_TEX)
    args = parser.parse_args()

    rows = _build_rows(args.main_claim_csv)
    _write_csv(args.out_csv, rows)
    _write_tex(args.out_tex, rows)


if __name__ == "__main__":
    main()
