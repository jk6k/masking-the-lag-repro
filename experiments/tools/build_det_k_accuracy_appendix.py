"""Build appendix-ready DET-k config-conditioned accuracy sweep assets."""

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
DEFAULT_SWEEP_CSV = ROOT / "experiments/results/accuracy/accuracy_config_conditioned_cuda_detk_sweep4096_seed0_20260307.csv"
DEFAULT_EFFICIENCY_CSV = ROOT / "experiments/results/quick_reports/20260305_stage2_seedtrue_fullgrid/quickscan_e3_k_sweep.csv"
DEFAULT_OUT_DATA = MAIN_PROJECT_REPORT_DATA_DIR
DEFAULT_OUT_FIG = MAIN_PROJECT_REPORT_FIG_DIR


def _normalize_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _load_accuracy_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["baseline_flag"] = _normalize_bool(df["baseline"])
    df["top1"] = pd.to_numeric(df["top1"], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["det_k_global"] = pd.to_numeric(df["det_k_global"], errors="coerce")
    return df


def _build_pair_rows(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["experiment_id"] == "E0"].copy()
    rows: list[dict[str, float | str]] = []
    tracked = (
        df[(df["experiment_id"] == "E3") & (~df["baseline_flag"])]
        .copy()
        .sort_values(["det_k_global", "seed", "model"])
    )
    for row in tracked.itertuples(index=False):
        fp32 = df[
            (df["experiment_id"] == "E3")
            & (df["seed"] == row.seed)
            & (df["model"] == row.model)
            & (df["baseline_flag"])
            & (df["det_k_global"] == row.det_k_global)
        ]["top1"].iloc[0]
        e0q = base[
            (base["seed"] == row.seed)
            & (base["model"] == row.model)
            & (~base["baseline_flag"])
        ]["top1"].iloc[0]
        rows.append(
            {
                "det_k_global": float(row.det_k_global),
                "seed": int(row.seed),
                "model": row.model,
                "delta_vs_fp32_pp": float(row.top1 - fp32),
                "delta_vs_e0_quant_pp": float(row.top1 - e0q),
            }
        )
    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        return pair_df
    zero_rows = []
    for seed in sorted(base["seed"].dropna().unique()):
        for model in sorted(base["model"].dropna().unique()):
            zero_rows.append(
                {
                    "det_k_global": 129.0,
                    "seed": int(seed),
                    "model": model,
                    "delta_vs_fp32_pp": 0.0,
                    "delta_vs_e0_quant_pp": 0.0,
                }
            )
    return pd.concat([pair_df, pd.DataFrame(zero_rows)], ignore_index=True)


def _build_per_k_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        pair_df.groupby("det_k_global", as_index=False)
        .agg(
            paired_model_mean_delta_vs_fp32_pp=("delta_vs_fp32_pp", "mean"),
            paired_model_std_delta_vs_fp32_pp=("delta_vs_fp32_pp", "std"),
            paired_model_mean_delta_vs_e0_quant_pp=("delta_vs_e0_quant_pp", "mean"),
            paired_model_std_delta_vs_e0_quant_pp=("delta_vs_e0_quant_pp", "std"),
            model_count=("model", "nunique"),
            seed_count=("seed", "nunique"),
        )
        .sort_values("det_k_global")
        .reset_index(drop=True)
    )
    return agg


def _merge_efficiency(summary: pd.DataFrame, efficiency_csv: Path) -> pd.DataFrame:
    eff = pd.read_csv(efficiency_csv)
    eff["k_global"] = pd.to_numeric(eff["k_global"], errors="coerce")
    merged = summary.merge(
        eff[
            [
                "k_global",
                "speedup_vs_E0",
                "det_net_gain_j",
                "prefix_error_mean",
                "prefix_error_p95",
            ]
        ],
        left_on="det_k_global",
        right_on="k_global",
        how="left",
    ).drop(columns="k_global")
    return merged


def _write_report(
    *,
    report_path: Path,
    source_csv: Path,
    efficiency_csv: Path,
    per_k_csv: Path,
    per_model_csv: Path,
    figure_pdf: Path,
    merged: pd.DataFrame,
) -> None:
    lines = [
        "# DET-k Config-Conditioned Accuracy Sweep Note",
        "",
        "Scope",
        f"- Accuracy source: `{source_csv}`",
        f"- Efficiency source: `{efficiency_csv}`",
        f"- Per-k summary: `{per_k_csv}`",
        f"- Per-model rows: `{per_model_csv}`",
        f"- Figure: `{figure_pdf}`",
        "",
        "Method",
        "- Rows come from a repaired CUDA config-conditioned accuracy sweep over DET truncation settings on a 4096-image/model subset.",
        "- Reported deltas are paired against each configuration's own FP32 rows and against the E0 quantized reference generated under the same evaluation path.",
        "- The 129-point is included as the zero-approximation reference line from E0.",
        "",
        "Aggregate summary",
    ]
    for row in merged.itertuples(index=False):
        lines.append(
            f"- `k={int(row.det_k_global)}`: "
            f"delta vs FP32 = `{row.paired_model_mean_delta_vs_fp32_pp:.2f} pp`, "
            f"delta vs E0 quantized = `{row.paired_model_mean_delta_vs_e0_quant_pp:.2f} pp`, "
            f"speedup = `{row.speedup_vs_E0:.2f}x`, "
            f"prefix error mean = `{row.prefix_error_mean:.5f}`."
        )
    lines.extend(
        [
            "",
            "Interpretation",
            "- This sweep provides real configuration-conditioned measured separation across DET-k settings instead of a flat attached measured line.",
            "- The measured accuracy penalty steepens as k decreases, while speedup and net DET gain improve, yielding an explicit efficiency-versus-degradation tradeoff.",
            "- The k=64 point can now be defended as a mid-curve compromise rather than a proxy-only choice, but it still should not be overstated as a globally optimal accuracy-validated point.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_figure(merged: pd.DataFrame, out_base: Path) -> tuple[Path, Path, Path]:
    x = merged["det_k_global"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))

    ax = axes[0]
    ax.plot(
        x,
        merged["paired_model_mean_delta_vs_fp32_pp"],
        marker="o",
        linewidth=1.8,
        color="#4e79a7",
        label="vs config FP32",
    )
    ax.plot(
        x,
        merged["paired_model_mean_delta_vs_e0_quant_pp"],
        marker="s",
        linewidth=1.8,
        color="#e15759",
        label="vs E0 quantized",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("DET truncation k")
    ax.set_ylabel("Paired Top-1 Delta (pp)")
    ax.set_title("Measured DET-k Accuracy Sweep")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax2 = axes[1]
    ax2.plot(
        x,
        merged["speedup_vs_E0"],
        marker="o",
        linewidth=1.8,
        color="#59a14f",
        label="Speedup vs E0",
    )
    ax2.set_xlabel("DET truncation k")
    ax2.set_ylabel("Speedup (x)", color="#59a14f")
    ax2.tick_params(axis="y", labelcolor="#59a14f")
    ax2.set_title("Efficiency / Proxy Context")
    ax2b = ax2.twinx()
    ax2b.plot(
        x,
        merged["prefix_error_mean"],
        marker="^",
        linewidth=1.5,
        color="#f28e2b",
        label="Prefix error mean",
    )
    ax2b.set_ylabel("Prefix Error Mean", color="#f28e2b")
    ax2b.tick_params(axis="y", labelcolor="#f28e2b")

    handles1, labels1 = ax2.get_legend_handles_labels()
    handles2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=8, loc="upper right")

    for axis in axes:
        axis.grid(alpha=0.18, linewidth=0.5)

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
    parser = argparse.ArgumentParser(description="Build appendix-ready DET-k config-conditioned accuracy sweep assets.")
    parser.add_argument("--sweep_csv", type=Path, default=DEFAULT_SWEEP_CSV)
    parser.add_argument("--efficiency_csv", type=Path, default=DEFAULT_EFFICIENCY_CSV)
    parser.add_argument("--out_data_dir", type=Path, default=DEFAULT_OUT_DATA)
    parser.add_argument("--out_fig_dir", type=Path, default=DEFAULT_OUT_FIG)
    parser.add_argument("--tag", default="20260307")
    args = parser.parse_args()

    args.out_data_dir.mkdir(parents=True, exist_ok=True)
    args.out_fig_dir.mkdir(parents=True, exist_ok=True)

    accuracy = _load_accuracy_rows(args.sweep_csv)
    pairs = _build_pair_rows(accuracy)
    summary = _build_per_k_summary(pairs)
    merged = _merge_efficiency(summary, args.efficiency_csv)

    per_model_csv = args.out_data_dir / f"det_k_accuracy_per_model_{args.tag}.csv"
    per_k_csv = args.out_data_dir / f"det_k_accuracy_summary_{args.tag}.csv"
    report_md = args.out_data_dir / f"det_k_accuracy_report_{args.tag}.md"
    fig_base = args.out_fig_dir / f"DET_k_Accuracy_Sweep_{args.tag}"

    pairs.to_csv(per_model_csv, index=False)
    merged.to_csv(per_k_csv, index=False)
    _, fig_pdf, _ = _render_figure(merged, fig_base)
    _write_report(
        report_path=report_md,
        source_csv=args.sweep_csv,
        efficiency_csv=args.efficiency_csv,
        per_k_csv=per_k_csv,
        per_model_csv=per_model_csv,
        figure_pdf=fig_pdf,
        merged=merged,
    )

    print(f"[det-k-appendix] wrote {per_model_csv}")
    print(f"[det-k-appendix] wrote {per_k_csv}")
    print(f"[det-k-appendix] wrote {report_md}")
    print(f"[det-k-appendix] wrote {fig_base.with_suffix('.svg')}")
    print(f"[det-k-appendix] wrote {fig_base.with_suffix('.pdf')}")
    print(f"[det-k-appendix] wrote {fig_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
