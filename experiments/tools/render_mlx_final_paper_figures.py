#!/usr/bin/env python3
"""Render the MLX final-freeze figure pack with updated DET/SPARSE support wording."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.tools import render_paper_figures as base


_ORIG_PLOT_FIG8 = base.plot_fig8_det_acc
_ORIG_PLOT_FIG15 = base.plot_fig15_pareto
_ORIG_PLOT_FIG19 = base.plot_fig19_ablation


def plot_fig8_det_acc(
    quick_dir: Path,
    out_dir: Path,
    trace_rows: list[dict[str, Any]],
    run_tag: str,
    cmd: str,
    det_summary_csv: Path | None = None,
) -> None:
    if det_summary_csv is None:
        auto_summary = quick_dir / "fig8_det_k_summary.csv"
        if auto_summary.exists():
            det_summary_csv = auto_summary
    if det_summary_csv is None or not det_summary_csv.exists():
        return _ORIG_PLOT_FIG8(quick_dir, out_dir, trace_rows, run_tag, cmd, det_summary_csv)

    df = base._load_csv(det_summary_csv)
    required = {"det_k_global", "paired_model_mean_delta_vs_e0_quant_pp", "speedup_vs_E0"}
    if not required.issubset(df.columns):
        return _ORIG_PLOT_FIG8(quick_dir, out_dir, trace_rows, run_tag, cmd, det_summary_csv)

    df["det_k_global"] = base._to_num(df["det_k_global"])
    df["paired_model_mean_delta_vs_e0_quant_pp"] = base._to_num(df["paired_model_mean_delta_vs_e0_quant_pp"])
    df["speedup_vs_E0"] = base._to_num(df["speedup_vs_E0"])
    if "paired_model_std_delta_vs_e0_quant_pp" in df.columns:
        df["paired_model_std_delta_vs_e0_quant_pp"] = base._to_num(df["paired_model_std_delta_vs_e0_quant_pp"])
    df = df.dropna(subset=["det_k_global", "paired_model_mean_delta_vs_e0_quant_pp", "speedup_vs_E0"]).sort_values("det_k_global")
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(3.60, 2.70))
    if "paired_model_std_delta_vs_e0_quant_pp" in df.columns:
        std = df["paired_model_std_delta_vs_e0_quant_pp"].fillna(0)
        ax.fill_between(
            df["det_k_global"],
            df["paired_model_mean_delta_vs_e0_quant_pp"] - std,
            df["paired_model_mean_delta_vs_e0_quant_pp"] + std,
            color=base.PALETTE[0],
            alpha=0.2,
            zorder=2,
            label="± 1 std dev",
        )
    ax.plot(
        df["det_k_global"],
        df["paired_model_mean_delta_vs_e0_quant_pp"],
        marker="o",
        markersize=5.0,
        markerfacecolor="white",
        markeredgecolor=base.PALETTE[0],
        markeredgewidth=1.2,
        color=base.PALETTE[0],
        linewidth=1.5,
        zorder=3,
        label="delta vs E0 Baseline",
    )
    ax.axhline(0.0, color="#444444", linewidth=1.2, linestyle="--", zorder=1, label="E0 Baseline (0.0)")
    ax.set_xlabel("DET truncation k")
    ax.set_ylabel("Top-1 Delta vs E0 (pp)")
    ax.margins(x=0.05, y=0.35)

    ax2 = ax.twinx()
    ax2.plot(
        df["det_k_global"],
        df["speedup_vs_E0"],
        marker="s",
        markersize=4.2,
        markerfacecolor=base.PALETTE[2],
        markeredgecolor=base.PALETTE[2],
        color=base.PALETTE[2],
        linewidth=1.3,
        linestyle="--",
        zorder=2,
        label="speedup (x)",
    )
    ax2.set_ylabel("Speedup vs E0", color=base.PALETTE[2])
    ax2.tick_params(axis="y", labelcolor=base.PALETTE[2])
    ax2.margins(x=0.05, y=0.35)

    k64 = df[df["det_k_global"] == 64]
    if not k64.empty:
        row = k64.iloc[0]
        ax.scatter(
            [row["det_k_global"]],
            [row["paired_model_mean_delta_vs_e0_quant_pp"]],
            marker="*",
            s=55,
            color=base.PALETTE[6],
            zorder=4,
            label="Promoted point (k=64)",
        )

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=7.2)

    evidence = set(df.get("evidence_tier", pd.Series("", index=df.index)).astype(str).str.lower())
    fully_measured = bool(evidence) and not any("projection" in item for item in evidence)
    footer = (
        "Displayed DET-k points are measured under the active MLX support sweep; k=64 is the promoted operating point and k=129 is the full-BSL reference."
        if fully_measured
        else "k=64 is the repaired full-eval anchor; k=129 is the E0-equivalence anchor."
    )
    fig.text(0.5, -0.05, footer, ha="center", fontsize=7.0, color="#666666")
    svg_path, _, _ = base._save_fig(fig, "Fig8_DET_AccVsBSL", out_dir)

    trace_note = (
        "Displayed DET-k points are measured under the active MLX support sweep."
        if fully_measured
        else "k=64 repaired full-eval anchor; k=129 E0-equivalence anchor; remaining DET-k points use prefix-error-bounded projections."
    )
    base._record_trace(
        trace_rows,
        fig_id="Fig8",
        figure_file=svg_path,
        input_csvs=[det_summary_csv],
        run_tag=run_tag,
        command=cmd,
        params_summary="mode=mlx_final_det_k_summary; delta_ref=E0_quantized; speedup_ref=E0",
        notes=trace_note,
    )


def plot_fig15_pareto(quick_dir: Path, out_dir: Path, trace_rows: list[dict[str, Any]], run_tag: str, cmd: str) -> None:
    path = quick_dir / "fig_j_sparse_tau_pareto.csv"
    df = base._load_csv(path)
    if df.empty:
        return

    if "measured_acc_drop_pp_vs_E0_mean" not in df.columns:
        return _ORIG_PLOT_FIG15(quick_dir, out_dir, trace_rows, run_tag, cmd)

    df["tau"] = base._to_num(df["tau"])
    df["measured_acc_drop_pp_vs_E0_mean"] = base._to_num(df["measured_acc_drop_pp_vs_E0_mean"])
    df["energy_saved_pct"] = base._to_num(df["energy_saved_pct"])
    df = df.dropna(subset=["tau", "measured_acc_drop_pp_vs_E0_mean", "energy_saved_pct"]).sort_values("tau")
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(3.62, 2.45))
    ax.plot(
        df["measured_acc_drop_pp_vs_E0_mean"],
        df["energy_saved_pct"],
        linestyle="-",
        color=base.PALETTE[0],
        linewidth=1.3,
        zorder=2,
        label="measured MLX sweep",
    )
    ax.scatter(
        df["measured_acc_drop_pp_vs_E0_mean"],
        df["energy_saved_pct"],
        s=46,
        marker="o",
        facecolors="white",
        edgecolors=base.PALETTE[0],
        linewidths=1.0,
        zorder=3,
    )
    label_offsets = {0.00: (5, 4), 0.25: (5, -10), 0.50: (-30, 4)}
    for tau in [0.00, 0.25, 0.50]:
        sub = df[np.isclose(df["tau"], tau)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        ax.annotate(
            f"tau={row['tau']:.2f}",
            (row["measured_acc_drop_pp_vs_E0_mean"], row["energy_saved_pct"]),
            xytext=label_offsets.get(float(tau), (4, 4)),
            textcoords="offset points",
            fontsize=7.4,
        )
    if len(df) >= 2:
        start = df.iloc[1]
        end = df.iloc[-1]
        ax.annotate(
            "higher tau",
            (end["measured_acc_drop_pp_vs_E0_mean"], end["energy_saved_pct"]),
            xytext=(start["measured_acc_drop_pp_vs_E0_mean"], start["energy_saved_pct"] + 7.0),
            arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 0.8},
            fontsize=7.2,
            color="#555555",
        )
    ax.set_xlabel("Top-1 Drop vs E0 (pp)")
    ax.set_ylabel("Energy Saved (%)")
    ax.legend(frameon=False, fontsize=7.0, loc="lower right")
    fig.text(0.5, 0.015, "All displayed points are measured under the active MLX support sweep.", ha="center", fontsize=7.0, color="#666666")
    svg_path, _, _ = base._save_fig(fig, "Fig15_SparsePareto", out_dir)
    base._record_trace(
        trace_rows,
        fig_id="Fig15",
        figure_file=svg_path,
        input_csvs=[path],
        run_tag=run_tag,
        command=cmd,
        params_summary="x=measured_acc_drop_pp_vs_E0_mean; y=energy_saved_pct; measured_sparse_sweep=true",
        notes="Displayed tau sweep points are measured under the active MLX support accuracy surface.",
    )


def plot_fig17_overall_pareto(quick_dir: Path, out_dir: Path, trace_rows: list[dict[str, Any]], run_tag: str, cmd: str) -> None:
    base.plot_fig17_overall_pareto(quick_dir, out_dir, trace_rows, run_tag, cmd)


def plot_fig19_ablation(quick_dir: Path, out_dir: Path, trace_rows: list[dict[str, Any]], run_tag: str, cmd: str) -> None:
    contract_path = quick_dir / "fig19_ablation_contract_summary.csv"
    if contract_path.exists():
        return _ORIG_PLOT_FIG19(quick_dir, out_dir, trace_rows, run_tag, cmd)

    path = quick_dir / "ablation_summary.csv"
    df = base._load_csv(path)
    if df.empty:
        return
    keep = ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]
    df = df[df["experiment_id"].isin(keep)].copy()
    if df.empty:
        return
    acc_col = "acc_drop_pp_mean" if "acc_drop_pp_mean" in df.columns else base._pick_col(df, ["measured_acc_drop_pp_mean"])
    if acc_col is None:
        return
    for col in ["speedup_vs_E0", "energy_ratio_vs_E0", acc_col]:
        df[col] = base._to_num(df[col])
    df = df.dropna(subset=["speedup_vs_E0", "energy_ratio_vs_E0", acc_col])
    if df.empty:
        return
    df["experiment_id"] = pd.Categorical(df["experiment_id"], categories=keep, ordered=True)
    df = df.sort_values("experiment_id")
    fig, ax1 = plt.subplots(figsize=(3.60, 2.48))
    x = np.arange(len(df))
    w = 0.34
    b1 = ax1.bar(x - w / 2, df["speedup_vs_E0"], width=w, color=base.PALETTE[0], hatch="///", edgecolor="black", label="speedup")
    b2 = ax1.bar(x + w / 2, df["energy_ratio_vs_E0"], width=w, color="#b0b0b0", hatch="...", edgecolor="black", label="energy ratio")
    ax1.set_ylabel("Speedup / Energy Ratio (x)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["experiment_id"].astype(str))
    ax2 = ax1.twinx()
    evidence = df.get("accuracy_evidence", pd.Series("", index=df.index)).astype(str).str.lower()
    modeled_mask = evidence.str.contains("modeled").to_numpy(dtype=bool)
    ax2.plot(x, df[acc_col], linestyle="--", color="#d62728", linewidth=1.2, alpha=0.85)
    measured_pts = ax2.scatter(x[~modeled_mask], df.loc[~modeled_mask, acc_col], s=28, marker="o", facecolors="#d62728", edgecolors="#d62728", linewidths=0.8, label="measured / shared-ref acc", zorder=4)
    modeled_pts = ax2.scatter(x[modeled_mask], df.loc[modeled_mask, acc_col], s=30, marker="o", facecolors="white", edgecolors="#d62728", linewidths=1.2, label="modeled acc", zorder=5)
    ax2.set_ylabel("Accuracy Drop (pp)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.legend([b1, b2, measured_pts, modeled_pts], ["speedup vs E0", "energy ratio vs E0", "measured / shared-ref acc", "modeled acc"], loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False, fontsize=7.0)
    fig.text(0.5, 0.015, "Hollow red markers indicate modeled accuracy rows.", ha="center", fontsize=7.0, color="#666666")
    svg_path, _, _ = base._save_fig(fig, "Fig19_Ablation", out_dir)
    base._record_trace(trace_rows, fig_id="Fig19", figure_file=svg_path, input_csvs=[path], run_tag=run_tag, command=cmd, params_summary=f"acc_col={acc_col}; hollow_markers=modeled_accuracy", notes="Measured/shared-reference accuracy and modeled accuracy are rendered with separate marker treatments.")


base.plot_fig8_det_acc = plot_fig8_det_acc
base.plot_fig15_pareto = plot_fig15_pareto
base.plot_fig17_overall_pareto = plot_fig17_overall_pareto
base.plot_fig19_ablation = plot_fig19_ablation


if __name__ == "__main__":
    base.main()
