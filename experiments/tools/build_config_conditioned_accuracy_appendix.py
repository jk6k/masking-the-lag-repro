"""Build appendix-ready config-conditioned accuracy repair assets.

This script consolidates repaired config-conditioned reruns for E3/E4/E6 into:

- a per-model summary CSV
- an aggregate summary CSV
- a markdown report
- a compact figure for appendix/rebuttal use
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from experiments.tools.path_policy import MAIN_PROJECT_REPORT_DATA_DIR, MAIN_PROJECT_REPORT_FIG_DIR
except ModuleNotFoundError:  # direct script execution
    from path_policy import MAIN_PROJECT_REPORT_DATA_DIR, MAIN_PROJECT_REPORT_FIG_DIR


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_CSV = ROOT / "experiments/results/accuracy/accuracy_config_conditioned_cuda_batch128_20260306.csv"
DEFAULT_E6_FIX_CSV = ROOT / "experiments/results/accuracy/accuracy_config_conditioned_cuda_e6_sparsefix_batch128_20260306.csv"
DEFAULT_FULL_CSV = ROOT / "experiments/results/accuracy/accuracy_config_conditioned_cuda_batch4096_fullrepaired_20260307.csv"
DEFAULT_OUT_DATA = MAIN_PROJECT_REPORT_DATA_DIR
DEFAULT_OUT_FIG = MAIN_PROJECT_REPORT_FIG_DIR


def _normalize_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _load_repaired_rows(batch_csv: Path, e6_fix_csv: Path) -> pd.DataFrame:
    batch_df = pd.read_csv(batch_csv)
    e6_fix_df = pd.read_csv(e6_fix_csv)
    merged = pd.concat(
        [
            batch_df[batch_df["experiment_id"] != "E6"],
            e6_fix_df,
        ],
        ignore_index=True,
    )
    merged["baseline_flag"] = _normalize_bool(merged["baseline"])
    merged["top1"] = pd.to_numeric(merged["top1"], errors="coerce")
    merged["seed"] = pd.to_numeric(merged["seed"], errors="coerce")
    return merged


def _load_full_rows(full_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(full_csv)
    df["baseline_flag"] = _normalize_bool(df["baseline"])
    df["top1"] = pd.to_numeric(df["top1"], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    return df


def _build_per_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        df[df["baseline_flag"]]
        .groupby(["experiment_id", "model"], as_index=False)
        .agg(
            baseline_top1_mean=("top1", "mean"),
            baseline_top1_std=("top1", "std"),
            baseline_n_rows=("top1", "size"),
        )
    )
    nonbaseline = (
        df[~df["baseline_flag"]]
        .groupby(["experiment_id", "model"], as_index=False)
        .agg(
            nonbaseline_top1_mean=("top1", "mean"),
            nonbaseline_top1_std=("top1", "std"),
            nonbaseline_n_rows=("top1", "size"),
            det_k_global=("det_k_global", "first"),
            sparse_tau_global=("sparse_tau_global", "first"),
            sparse_active_fraction=("sparse_active_fraction", "first"),
            det_perturbation=("det_perturbation", "first"),
            sparse_perturbation=("sparse_perturbation", "first"),
        )
    )
    summary = baseline.merge(nonbaseline, on=["experiment_id", "model"], how="outer")
    e0_quant = (
        summary.loc[summary["experiment_id"] == "E0", ["model", "nonbaseline_top1_mean"]]
        .rename(columns={"nonbaseline_top1_mean": "e0_quant_top1_mean"})
    )
    summary = summary.merge(e0_quant, on="model", how="left")
    summary["delta_vs_fp32_pp"] = summary["nonbaseline_top1_mean"] - summary["baseline_top1_mean"]
    summary["delta_vs_e0_quant_pp"] = summary["nonbaseline_top1_mean"] - summary["e0_quant_top1_mean"]
    return summary.sort_values(["experiment_id", "model"]).reset_index(drop=True)


def _build_aggregate_summary(per_model: pd.DataFrame) -> pd.DataFrame:
    tracked = per_model[per_model["experiment_id"].isin(["E3", "E4", "E6"])].copy()
    agg = (
        tracked.groupby("experiment_id", as_index=False)
        .agg(
            paired_model_mean_delta_vs_fp32_pp=("delta_vs_fp32_pp", "mean"),
            paired_model_std_delta_vs_fp32_pp=("delta_vs_fp32_pp", "std"),
            paired_model_mean_delta_vs_e0_quant_pp=("delta_vs_e0_quant_pp", "mean"),
            paired_model_std_delta_vs_e0_quant_pp=("delta_vs_e0_quant_pp", "std"),
            model_count=("model", "nunique"),
        )
    )
    order = pd.Categorical(agg["experiment_id"], categories=["E3", "E4", "E6"], ordered=True)
    agg = agg.assign(_order=order).sort_values("_order").drop(columns="_order")
    return agg.reset_index(drop=True)


def _write_report(
    *,
    report_path: Path,
    per_model_csv: Path,
    aggregate_csv: Path,
    figure_pdf: Path,
    source_note: str,
    source_kind: str,
    batch_csv: Path,
    e6_fix_csv: Path,
    aggregate: pd.DataFrame,
) -> None:
    is_full_eval = source_kind == "full_eval"
    lines = [
        f"# Config-Conditioned Accuracy Appendix Note ({source_note})",
        "",
        "Scope",
        f"- Source batch: `{batch_csv}`",
    ]
    if e6_fix_csv != batch_csv:
        lines.append(f"- E6 sparse-active override: `{e6_fix_csv}`")
    lines.extend(
        [
        f"- Per-model summary: `{per_model_csv}`",
        f"- Aggregate summary: `{aggregate_csv}`",
        f"- Figure: `{figure_pdf}`",
        "",
        "Method",
        (
            "- Rows come from repaired config-conditioned full-eval runs on the "
            "ImageNet eval split."
            if is_full_eval
            else "- Rows come from a repaired config-conditioned batch using "
            "fixed seeds and the ImageNet eval split subset configured for this run."
        ),
        (
            "- No E6 override is used here because the reported rows already come "
            "from the repaired full-eval path."
            if is_full_eval
            else "- When a separate E6 override is provided, E6 rows are replaced "
            "with the sparse-active rerun so the appendix evidence does not retain "
            "the earlier runner bug."
        ),
        "- Baseline rows are the per-config FP32 reference generated by the same evaluation path; non-baseline rows are the quantized/config-conditioned runs.",
        "- Reported deltas are descriptive paired model means across `mobilevit_xxs`, `mobilevit_xs`, and `mobilevit_s`.",
        "",
        (
            "Aggregate full-eval summary"
            if is_full_eval
            else "Aggregate repaired-batch summary"
        ),
        ]
    )
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"- `{row.experiment_id}`: "
            f"delta vs FP32 = `{row.paired_model_mean_delta_vs_fp32_pp:.2f} pp`, "
            f"delta vs E0 quantized = `{row.paired_model_mean_delta_vs_e0_quant_pp:.2f} pp`."
        )
    lines.extend(
        [
            "",
            "Interpretation",
            "- `E3` now has a stable negative separation from `E0`, confirming that DET is no longer metadata-only in the repaired path.",
            (
                "- `E4` now reflects the repaired direct sparse-semantics path on the "
                "released full-eval chain; its severity should be read from the "
                "reported deltas rather than assumed to remain moderate."
                if is_full_eval
                else "- `E4` now reflects the repaired direct sparse-semantics path on "
                "the larger repaired batch; its severity should be read from the "
                "reported deltas rather than assumed to remain moderate."
            ),
            (
                "- `E6` remains clearly efficiency-positive on the main paper path, "
                "but the repaired full-eval rows still do not justify any accuracy-"
                "preservation claim."
                if is_full_eval
                else "- `E6` remains clearly efficiency-positive on the main paper "
                "path, but the sparse-active repaired batch still does not justify "
                "any accuracy-preservation claim."
            ),
            (
                "- These full-eval rows close the `E0/E3/E4/E6` configuration-conditioned "
                "accuracy path for the supported MobileViT workload, but they close it "
                "in the negative direction: the manuscript still cannot claim accuracy "
                "preservation."
                if is_full_eval
                else "- These appendix assets strengthen traceability and rebuttal "
                "readiness, but they do not close full-eval accuracy for the "
                "manuscript headline."
            ),
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_figure(aggregate: pd.DataFrame, out_base: Path) -> tuple[Path, Path, Path]:
    labels = aggregate["experiment_id"].tolist()
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    bars_fp32 = ax.bar(
        x - width / 2,
        aggregate["paired_model_mean_delta_vs_fp32_pp"],
        width,
        label="vs config FP32",
        color="#4e79a7",
    )
    bars_e0 = ax.bar(
        x + width / 2,
        aggregate["paired_model_mean_delta_vs_e0_quant_pp"],
        width,
        label="vs E0 quantized",
        color="#e15759",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Paired Top-1 Delta (pp)")
    ax.set_title("Repaired Config-Conditioned Accuracy Batch")
    ymin = min(
        aggregate["paired_model_mean_delta_vs_fp32_pp"].min(),
        aggregate["paired_model_mean_delta_vs_e0_quant_pp"].min(),
    )
    ymax = max(
        aggregate["paired_model_mean_delta_vs_fp32_pp"].max(),
        aggregate["paired_model_mean_delta_vs_e0_quant_pp"].max(),
    )
    ax.set_ylim(ymin - 0.45, ymax + 0.55)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    for bars in (bars_fp32, bars_e0):
        for bar in bars:
            height = float(bar.get_height())
            va = "bottom" if height >= 0 else "top"
            offset = 0.08 if height >= 0 else -0.08
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f"{height:.2f}",
                ha="center",
                va=va,
                fontsize=8,
            )

    fig.tight_layout()
    svg_path = out_base.with_suffix(".svg")
    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return svg_path, pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build appendix-ready repaired config-conditioned accuracy assets.")
    parser.add_argument("--full_csv", type=Path, default=None)
    parser.add_argument("--batch_csv", type=Path, default=DEFAULT_BATCH_CSV)
    parser.add_argument("--e6_fix_csv", type=Path, default=DEFAULT_E6_FIX_CSV)
    parser.add_argument("--out_data_dir", type=Path, default=DEFAULT_OUT_DATA)
    parser.add_argument("--out_fig_dir", type=Path, default=DEFAULT_OUT_FIG)
    parser.add_argument("--tag", default="20260306")
    args = parser.parse_args()

    args.out_data_dir.mkdir(parents=True, exist_ok=True)
    args.out_fig_dir.mkdir(parents=True, exist_ok=True)

    if args.full_csv is not None:
        repaired = _load_full_rows(args.full_csv)
        source_batch = args.full_csv
        e6_source = args.full_csv
        source_kind = "full_eval"
    else:
        repaired = _load_repaired_rows(args.batch_csv, args.e6_fix_csv)
        source_batch = args.batch_csv
        e6_source = args.e6_fix_csv
        source_kind = "subset_batch"
    per_model = _build_per_model_summary(repaired)
    aggregate = _build_aggregate_summary(per_model)

    per_model_csv = args.out_data_dir / f"config_conditioned_accuracy_per_model_{args.tag}.csv"
    aggregate_csv = args.out_data_dir / f"config_conditioned_accuracy_summary_{args.tag}.csv"
    report_md = args.out_data_dir / f"config_conditioned_accuracy_report_{args.tag}.md"
    fig_base = args.out_fig_dir / f"ConfigConditioned_Accuracy_Repair_{args.tag}"

    per_model.to_csv(per_model_csv, index=False)
    aggregate.to_csv(aggregate_csv, index=False)
    _, fig_pdf, _ = _render_figure(aggregate, fig_base)
    _write_report(
        report_path=report_md,
        per_model_csv=per_model_csv,
        aggregate_csv=aggregate_csv,
        figure_pdf=fig_pdf,
        source_note=args.tag,
        source_kind=source_kind,
        batch_csv=source_batch,
        e6_fix_csv=e6_source,
        aggregate=aggregate,
    )

    print(f"[appendix-accuracy] wrote {per_model_csv}")
    print(f"[appendix-accuracy] wrote {aggregate_csv}")
    print(f"[appendix-accuracy] wrote {report_md}")
    print(f"[appendix-accuracy] wrote {fig_base.with_suffix('.svg')}")
    print(f"[appendix-accuracy] wrote {fig_base.with_suffix('.pdf')}")
    print(f"[appendix-accuracy] wrote {fig_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
