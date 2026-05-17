#!/usr/bin/env python3
"""Build the unified main-claim summary for the retained E0/E2/E3/E4/E6 chain."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

try:
    from experiments.tools.path_policy import MAIN_PROJECT_REPORT_DATA_DIR
except ModuleNotFoundError:  # direct script execution
    from path_policy import MAIN_PROJECT_REPORT_DATA_DIR


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HEADLINE_SUMMARY_CSV = MAIN_PROJECT_REPORT_DATA_DIR / "headline_statistics_summary_20260310.csv"
DEFAULT_HEADLINE_PAIR_CI_CSV = MAIN_PROJECT_REPORT_DATA_DIR / "headline_statistics_pair_ci_20260310.csv"
DEFAULT_HEADLINE_PER_SEED_CSV = MAIN_PROJECT_REPORT_DATA_DIR / "headline_statistics_per_seed_20260310.csv"
DEFAULT_HEADLINE_PER_MODEL_CSV = MAIN_PROJECT_REPORT_DATA_DIR / "headline_statistics_per_model_20260310.csv"
DEFAULT_OUT_DIR = MAIN_PROJECT_REPORT_DATA_DIR
DEFAULT_TAG = "20260310"
RETAINED_ORDER = ["E0", "E2", "E3", "E4", "E6"]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def _is_modeled_accuracy_evidence(value: object) -> bool:
    return str(value or "").strip().lower() in {"modeled_full_eval", "proxy_config_conditioned"}


def _format_experiment_list(experiments: list[str]) -> str:
    return ", ".join(f"`{exp}`" for exp in experiments)


def _load_main_rows(
    summary_csv: Path,
    pair_ci_csv: Path,
    per_seed_csv: Path,
    per_model_csv: Path,
) -> list[dict[str, object]]:
    summary = pd.read_csv(summary_csv)
    pair_ci = pd.read_csv(pair_ci_csv)
    per_seed = pd.read_csv(per_seed_csv)
    per_model = pd.read_csv(per_model_csv)

    rows: list[dict[str, object]] = []
    for experiment_id in RETAINED_ORDER:
        row = summary.loc[summary["experiment_id"] == experiment_id]
        if row.empty:
            raise SystemExit(f"Missing summary row for {experiment_id} in {summary_csv}")
        row = row.iloc[0]

        seed_sub = per_seed.loc[per_seed["experiment_id"] == experiment_id]
        model_sub = per_model.loc[per_model["experiment_id"] == experiment_id]
        split = "eval" if experiment_id in {"E0", "E2", "E3", "E4", "E6"} else "unknown"
        models = ",".join(sorted(model_sub["model"].astype(str).unique().tolist()))
        seeds = ",".join(str(int(seed)) for seed in sorted(seed_sub["seed"].astype(int).unique().tolist()))

        merged: dict[str, object] = {
            "experiment_id": experiment_id,
            "performance_run_id": row["performance_run_id"],
            "performance_run_path": row["performance_run_path"],
            "performance_split": split,
            "accuracy_source_csv": str(seed_sub["accuracy_source_csv"].dropna().iloc[0]),
            "accuracy_split": "eval",
            "models": models,
            "seeds": seeds,
            "seed_count": int(seed_sub["seed"].astype(int).nunique()),
            "pair_basis": row["pair_basis"],
            "baseline_definition": str(row.get("reference_basis") or "local E0 8-bit SC reference chain"),
            "config_conditioned_accuracy": bool(experiment_id in {"E3", "E4", "E6"}),
            "accuracy_note": row["accuracy_note"],
            "accuracy_evidence": str(row.get("accuracy_evidence") or ""),
            "latency_ms_mean": float(row["latency_ms_mean"]),
            "energy_j_mean": float(row["energy_j_mean"]),
            "tops_w_mean": float(row["tops_w_mean"]),
            "top1_value_mean": float(row["top1_measured_mean"]),
            "acc_drop_pp_vs_fp32_mean": float(row["acc_drop_pp_vs_fp32_mean"]),
        }
        if experiment_id == "E0":
            merged.update(
                {
                    "speedup_vs_e0_mean": 1.0,
                    "energy_reduction_pct_vs_e0_mean": 0.0,
                    "tops_w_gain_vs_e0_mean": 1.0,
                    "acc_drop_pp_vs_e0_mean": 0.0,
                }
            )
        else:
            pair_row = pair_ci.loc[pair_ci["experiment_id"] == experiment_id]
            if pair_row.empty:
                raise SystemExit(f"Missing pair-CI row for {experiment_id} in {pair_ci_csv}")
            pair_row = pair_row.iloc[0]
            merged.update(
                {
                    "speedup_vs_e0_mean": float(pair_row["speedup_vs_e0_mean"]),
                    "energy_reduction_pct_vs_e0_mean": float(pair_row["energy_reduction_pct_vs_e0_mean"]),
                    "tops_w_gain_vs_e0_mean": float(pair_row["tops_w_gain_vs_e0_mean"]),
                    "acc_drop_pp_vs_e0_mean": float(pair_row["acc_drop_pp_vs_e0_mean"]),
                }
            )
        rows.append(merged)
    return rows


def _write_report(out_path: Path, rows: list[dict[str, object]], tag: str) -> None:
    by_id = {str(row["experiment_id"]): row for row in rows}
    e2_uses_shared_e0_reference = str(by_id["E2"].get("accuracy_evidence") or "") == "shared_e0_full_eval_reference"
    e6_is_modeled = _is_modeled_accuracy_evidence(by_id["E6"].get("accuracy_evidence"))
    modeled_config_rows = [
        str(row["experiment_id"])
        for row in rows
        if bool(row.get("config_conditioned_accuracy"))
        and _is_modeled_accuracy_evidence(row.get("accuracy_evidence"))
    ]
    measured_config_rows = [
        str(row["experiment_id"])
        for row in rows
        if bool(row.get("config_conditioned_accuracy"))
        and str(row.get("accuracy_evidence") or "").strip().lower() == "measured_full_eval"
    ]
    if e2_uses_shared_e0_reference:
        e2_decision_line = (
            f"- `E2` retains a measured latency gain (`{_fmt(by_id['E2']['speedup_vs_e0_mean'], 2)}x`), "
            f"and its Top-1 evidence is explicitly carried by the shared local `E0` quantized full-eval reference."
        )
    else:
        e2_decision_line = (
            f"- `E2` retains a measured latency gain (`{_fmt(by_id['E2']['speedup_vs_e0_mean'], 2)}x`) "
            f"and now carries dedicated `measured_full_eval` Top-1 evidence under the same repaired protocol."
        )
    modeled_decision_line = None
    if modeled_config_rows:
        modeled_decision_line = (
            f"- Modeled config-conditioned rows in this chain ({_format_experiment_list(modeled_config_rows)}) "
            "can be cited as modeled full-eval evidence for their own configuration, but they must not be promoted as direct measured headline evidence."
        )
    measured_decision_line = None
    if measured_config_rows:
        measured_decision_line = (
            f"- Measured config-conditioned rows in this chain ({_format_experiment_list(measured_config_rows)}) "
            "can be cited as measured full-eval evidence for their own configuration, but they do not upgrade the modeled configurations by implication."
        )
    lines = [
        f"# Unified Main-Claim Summary ({tag})",
        "",
        "Scope",
        "- Retained headline chain: `E0/E2/E3/E4/E6` only.",
        "- Performance and energy are taken from the released paired `master_metrics.csv` runs on the ImageNet eval split.",
        "- Accuracy metadata is taken from the repaired chain and tagged as `measured_full_eval`, `modeled_full_eval`, or `shared_e0_full_eval_reference`.",
        "- Summary rows report the full governed seed release recorded in the per-seed statistics exports rather than silently collapsing to the first available seed.",
        "",
        "Unified chain record",
    ]
    for row in rows:
        lines.extend(
            [
                (
                    f"- `{row['experiment_id']}`: perf=`{row['performance_run_path']}`, "
                    f"acc=`{row['accuracy_source_csv']}`, split=`{row['performance_split']}`, "
                    f"models=`{row['models']}`, seeds=`{row['seeds']}`, "
                    f"seed-count=`{row['seed_count']}`, "
                    f"config-conditioned=`{row['config_conditioned_accuracy']}`, "
                    f"accuracy-evidence=`{row['accuracy_evidence']}`."
                ),
                (
                    f"  headline: latency `{_fmt(row['latency_ms_mean'], 3)} ms`, energy `{_fmt(row['energy_j_mean'], 4)} J`, "
                    f"TOPS/W `{_fmt(row['tops_w_mean'], 3)}`, Top-1 `{_fmt(row['top1_value_mean'], 2)}%`, "
                    f"delta vs local E0 quant `{_fmt(row['acc_drop_pp_vs_e0_mean'], 2)} pp`."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "Decision",
            e2_decision_line,
            (
                f"- `E6` remains a strong efficiency endpoint (`{_fmt(by_id['E6']['speedup_vs_e0_mean'], 2)}x`, "
                f"`{_fmt(by_id['E6']['energy_reduction_pct_vs_e0_mean'], 2)}%` energy reduction) but carries "
                f"`{_fmt(by_id['E6']['acc_drop_pp_vs_e0_mean'], 2)} pp` "
                f"{'modeled' if e6_is_modeled else 'measured'} Top-1 loss versus the local E0 quantized chain."
            ),
            "- The paper should therefore use a tradeoff/case-study headline rather than present E6 as a recommended accuracy-safe preset.",
            "",
            "P0 Closure",
            "- This note provides one retained source path for every headline number in Table 1 and the abstract.",
            "- Any remaining manuscript claim should point back to this chain instead of mixing legacy protocol-pack numbers with repaired full-eval accuracy rows.",
        ]
    )
    if modeled_decision_line is not None:
        lines.insert(-5, modeled_decision_line)
    if measured_decision_line is not None:
        lines.insert(-5, measured_decision_line)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the unified main-claim summary note.")
    parser.add_argument("--headline_summary_csv", type=Path, default=DEFAULT_HEADLINE_SUMMARY_CSV)
    parser.add_argument("--headline_pair_ci_csv", type=Path, default=DEFAULT_HEADLINE_PAIR_CI_CSV)
    parser.add_argument("--headline_per_seed_csv", type=Path, default=DEFAULT_HEADLINE_PER_SEED_CSV)
    parser.add_argument("--headline_per_model_csv", type=Path, default=DEFAULT_HEADLINE_PER_MODEL_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    args = parser.parse_args()

    rows = _load_main_rows(
        summary_csv=args.headline_summary_csv,
        pair_ci_csv=args.headline_pair_ci_csv,
        per_seed_csv=args.headline_per_seed_csv,
        per_model_csv=args.headline_per_model_csv,
    )

    csv_path = args.out_dir / f"main_claim_unified_summary_{args.tag}.csv"
    md_path = args.out_dir / f"main_claim_unified_summary_{args.tag}.md"
    _write_csv(
        csv_path,
        rows,
        [
            "experiment_id",
            "performance_run_id",
            "performance_run_path",
            "performance_split",
            "accuracy_source_csv",
            "accuracy_split",
            "models",
            "seeds",
            "seed_count",
            "pair_basis",
            "baseline_definition",
            "config_conditioned_accuracy",
            "accuracy_note",
            "accuracy_evidence",
            "latency_ms_mean",
            "energy_j_mean",
            "tops_w_mean",
            "top1_value_mean",
            "acc_drop_pp_vs_fp32_mean",
            "speedup_vs_e0_mean",
            "energy_reduction_pct_vs_e0_mean",
            "tops_w_gain_vs_e0_mean",
            "acc_drop_pp_vs_e0_mean",
        ],
    )
    _write_report(md_path, rows, args.tag)


if __name__ == "__main__":
    main()
