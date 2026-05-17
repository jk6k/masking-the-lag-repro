#!/usr/bin/env python3
"""Build reviewer-facing appendix assets for PASCAL VOC config-conditioned segmentation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from experiments.tools.path_policy import MAIN_PROJECT_REPORT_DATA_DIR, MAIN_PROJECT_REPORT_FIG_DIR
except ModuleNotFoundError:  # direct script execution
    from path_policy import MAIN_PROJECT_REPORT_DATA_DIR, MAIN_PROJECT_REPORT_FIG_DIR


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = ROOT / "experiments" / "results" / "accuracy" / "pascalvoc_seg_config_conditioned_fullval_20260308.csv"
DEFAULT_OUT_DATA = MAIN_PROJECT_REPORT_DATA_DIR
DEFAULT_OUT_FIG = MAIN_PROJECT_REPORT_FIG_DIR
DEFAULT_TAG = "20260308_pascalvocconfig"


def _normalize_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _format_opt_float(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _validate_unit_interval_series(frame: pd.DataFrame, column: str, source: Path) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    invalid_mask = values.isna() | ~values.map(math.isfinite) | (values < 0.0) | (values > 1.0)
    if invalid_mask.any():
        invalid_rows = (frame.index[invalid_mask] + 2).tolist()
        bad_values = ", ".join(str(frame.loc[idx, column]) for idx in invalid_mask[invalid_mask].index[:3])
        raise ValueError(
            f"{source} has invalid {column} values outside [0, 1] at CSV rows {invalid_rows}: {bad_values}"
        )
    return values


def _validate_sparse_metadata(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    sparse_rows = frame[frame["sparse_perturbation"]].copy()
    if sparse_rows.empty:
        return frame

    sparse_rows["sparse_tau_global"] = _validate_unit_interval_series(sparse_rows, "sparse_tau_global", source)
    sparse_rows["sparse_active_fraction"] = _validate_unit_interval_series(
        sparse_rows,
        "sparse_active_fraction",
        source,
    )
    for experiment_id, group in sparse_rows.groupby("experiment_id", sort=False):
        for column in ("sparse_tau_global", "sparse_active_fraction"):
            unique_values = sorted({round(float(value), 12) for value in group[column].tolist()})
            if len(unique_values) > 1:
                raise ValueError(
                    f"{source} has inconsistent {column} values for experiment_id {experiment_id}: {unique_values}"
                )

    frame.loc[sparse_rows.index, "sparse_tau_global"] = sparse_rows["sparse_tau_global"]
    frame.loc[sparse_rows.index, "sparse_active_fraction"] = sparse_rows["sparse_active_fraction"]
    return frame


def _mechanism_label(row: pd.Series) -> str:
    det_enabled = bool(row["det_perturbation"])
    sparse_enabled = bool(row["sparse_perturbation"])
    if det_enabled and sparse_enabled:
        return (
            f"DET(k={int(float(row['det_k_global']))}) + "
            f"SPARSE(tau={float(row['sparse_tau_global']):.2f}, compat_active={float(row['sparse_active_fraction']):.2f})"
        )
    if det_enabled:
        return f"DET(k={int(float(row['det_k_global']))})"
    if sparse_enabled:
        return (
            f"SPARSE(tau={float(row['sparse_tau_global']):.2f}, "
            f"compat_active={float(row['sparse_active_fraction']):.2f})"
        )
    return "baseline"


def _load_summary(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df = df[df["workload_id"] == "mobilevit_s_pascalvoc_seg"].copy()
    if df.empty:
        raise ValueError(f"No PASCAL VOC config-conditioned rows found in {input_csv}")

    df["baseline_flag"] = _normalize_bool(df["baseline"])
    df["det_perturbation"] = _normalize_bool(df["det_perturbation"])
    df["sparse_perturbation"] = _normalize_bool(df["sparse_perturbation"])
    numeric_columns = [
        "sample_count",
        "global_correct_pct",
        "mean_iou_pct",
        "finite_mean_iou_pct",
        "finite_iou_class_count",
        "global_correct_delta_pp",
        "mean_iou_delta_pp",
        "finite_mean_iou_delta_pp",
        "det_k_global",
        "det_prefix_error_mean",
        "det_prefix_error_p95",
        "sparse_tau_global",
        "sparse_active_fraction",
        "seed",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = _validate_sparse_metadata(df, input_csv)

    baseline = (
        df[df["baseline_flag"]]
        .groupby("experiment_id", as_index=False)
        .agg(
            baseline_sample_count=("sample_count", "max"),
            baseline_global_correct_pct=("global_correct_pct", "mean"),
            baseline_mean_iou_pct=("mean_iou_pct", "mean"),
            baseline_finite_mean_iou_pct=("finite_mean_iou_pct", "mean"),
            baseline_finite_iou_class_count=("finite_iou_class_count", "max"),
        )
    )
    perturbed = (
        df[~df["baseline_flag"]]
        .groupby("experiment_id", as_index=False)
        .agg(
            perturbed_sample_count=("sample_count", "max"),
            perturbed_global_correct_pct=("global_correct_pct", "mean"),
            perturbed_mean_iou_pct=("mean_iou_pct", "mean"),
            perturbed_finite_mean_iou_pct=("finite_mean_iou_pct", "mean"),
            perturbed_finite_iou_class_count=("finite_iou_class_count", "max"),
            global_correct_delta_pp=("global_correct_delta_pp", "mean"),
            mean_iou_delta_pp=("mean_iou_delta_pp", "mean"),
            finite_mean_iou_delta_pp=("finite_mean_iou_delta_pp", "mean"),
            det_k_global=("det_k_global", "first"),
            det_prefix_error_mean=("det_prefix_error_mean", "first"),
            det_prefix_error_p95=("det_prefix_error_p95", "first"),
            sparse_tau_global=("sparse_tau_global", "first"),
            sparse_active_fraction=("sparse_active_fraction", "first"),
            det_perturbation=("det_perturbation", "first"),
            sparse_perturbation=("sparse_perturbation", "first"),
            seed_count=("seed", "nunique"),
        )
    )
    summary = baseline.merge(perturbed, on="experiment_id", how="inner", validate="one_to_one")
    summary["mechanism_label"] = summary.apply(_mechanism_label, axis=1)
    order = pd.Categorical(summary["experiment_id"], categories=["E3", "E4", "E6"], ordered=True)
    return summary.assign(_order=order).sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _render_figure(summary: pd.DataFrame, out_base: Path) -> tuple[Path, Path, Path]:
    labels = summary["experiment_id"].tolist()
    x = list(range(len(labels)))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), gridspec_kw={"width_ratios": [1.0, 1.0]})
    colors = ["#4e79a7", "#f28e2b", "#e15759"]

    axes[0].bar(x, summary["finite_mean_iou_delta_pp"], color=colors)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Delta mIoU (pp)")
    axes[0].set_title("Panel A: finite-class mIoU delta")
    axes[0].grid(axis="y", linestyle=":", alpha=0.3)

    axes[1].bar(x, summary["global_correct_delta_pp"], color=colors)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Delta global correct (pp)")
    axes[1].set_title("Panel B: global-correct delta")
    axes[1].grid(axis="y", linestyle=":", alpha=0.3)

    for ax, column in zip(axes, ["finite_mean_iou_delta_pp", "global_correct_delta_pp"]):
        for xpos, value in enumerate(summary[column]):
            ax.text(
                xpos,
                float(value) + (-0.12 if float(value) < 0 else 0.08),
                f"{float(value):.2f}",
                ha="center",
                va="top" if float(value) < 0 else "bottom",
                fontsize=8,
            )

    fig.suptitle("PASCAL VOC Config-Conditioned Segmentation Full-Val", y=1.02, fontsize=11)
    fig.text(
        0.5,
        0.01,
        "All rows use the same 1449-image local validation mirror; E3=DET, E4=SPARSE, E6=DET+SPARSE.",
        ha="center",
        fontsize=7,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    svg_path = out_base.with_suffix(".svg")
    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return svg_path, pdf_path, png_path


def _write_report(report_path: Path, summary_csv: Path, figure_pdf: Path, source_csv: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# PASCAL VOC Config-Conditioned Segmentation Note",
        "",
        "Scope",
        f"- Source CSV: `{source_csv}`",
        f"- Summary CSV: `{summary_csv}`",
        f"- Figure: `{figure_pdf}`",
        "",
        "Headline",
        "- The DET/SPARSE perturbation chain now extends beyond ImageNet classification into a full-validation PASCAL VOC segmentation workload.",
        "- All rows use the same `1449`-image local PASCAL VOC 2012 validation mirror and the official MobileViT-S DeepLabv3 checkpoint path.",
        "",
        "Per-condition summary",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- `{row.experiment_id}` (`{row.mechanism_label}`): "
            f"baseline mIoU `{row.baseline_finite_mean_iou_pct:.2f}%`, "
            f"perturbed mIoU `{row.perturbed_finite_mean_iou_pct:.2f}%` "
            f"({row.finite_mean_iou_delta_pp:+.2f} pp); "
            f"global correct `{row.baseline_global_correct_pct:.2f}% -> {row.perturbed_global_correct_pct:.2f}%` "
            f"({row.global_correct_delta_pp:+.2f} pp)."
        )
    lines.extend(
        [
            "",
            "Traceability",
            f"- `E3`: DET only, `k={int(float(summary.loc[summary['experiment_id'] == 'E3', 'det_k_global'].iloc[0]))}`, "
            f"prefix-error mean `{_format_opt_float(summary.loc[summary['experiment_id'] == 'E3', 'det_prefix_error_mean'].iloc[0], 6)}`, "
            f"P95 `{_format_opt_float(summary.loc[summary['experiment_id'] == 'E3', 'det_prefix_error_p95'].iloc[0], 6)}`.",
            f"- `E4`: SPARSE only, `tau={_format_opt_float(summary.loc[summary['experiment_id'] == 'E4', 'sparse_tau_global'].iloc[0])}`, "
            f"compatibility active fraction `{_format_opt_float(summary.loc[summary['experiment_id'] == 'E4', 'sparse_active_fraction'].iloc[0])}`.",
            f"- `E6`: DET + SPARSE, `k={int(float(summary.loc[summary['experiment_id'] == 'E6', 'det_k_global'].iloc[0]))}`, "
            f"tau `{_format_opt_float(summary.loc[summary['experiment_id'] == 'E6', 'sparse_tau_global'].iloc[0])}`, "
            f"compatibility active fraction `{_format_opt_float(summary.loc[summary['experiment_id'] == 'E6', 'sparse_active_fraction'].iloc[0])}`.",
            "",
            "Interpretation",
            "- This closes the non-hardware broader-task extension target in a concrete in-house workload rather than leaving DET/SPARSE evidence limited to ImageNet classification.",
            "- `E3` shows the lightest segmentation degradation on the full validation mirror, while `E6` remains the most accuracy-negative operating point in this workload.",
            "- The broader-task extension therefore supports execution-chain validity and workload reach, not any accuracy-preservation claim.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build appendix assets for PASCAL VOC config-conditioned segmentation.")
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--out_data_dir", type=Path, default=DEFAULT_OUT_DATA)
    parser.add_argument("--out_fig_dir", type=Path, default=DEFAULT_OUT_FIG)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    args = parser.parse_args()

    args.out_data_dir.mkdir(parents=True, exist_ok=True)
    args.out_fig_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(args.input_csv)
    summary_csv = args.out_data_dir / f"pascalvoc_seg_config_conditioned_summary_{args.tag}.csv"
    report_md = args.out_data_dir / f"pascalvoc_seg_config_conditioned_report_{args.tag}.md"
    fig_base = args.out_fig_dir / f"Pascal_VOC_ConfigConditioned_{args.tag}"

    summary.to_csv(summary_csv, index=False)
    _, figure_pdf, _ = _render_figure(summary, fig_base)
    _write_report(report_md, summary_csv, figure_pdf, args.input_csv, summary)

    print(f"[pascalvoc-config] summary={summary_csv}")
    print(f"[pascalvoc-config] report={report_md}")
    print(f"[pascalvoc-config] figure={figure_pdf}")


if __name__ == "__main__":
    main()
