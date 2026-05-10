"""Render the repaired SUDS Q2+ paper figure pack.

Usage:
  python3 experiments/tools/render_suds_figures.py \
    --phase-dir experiments/results/runs \
    --output-dir figures/paper_figures_20260510_suds_q2_repaired \
    --figure all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
RUN_TAG = "20260510_suds_q2_repaired"
TRACEABILITY_NAME = "figure_traceability.csv"
REGISTRY_NAME = "figure_numbering_registry.csv"
VALIDATION_CSV = ROOT / "experiments/results/report_data/suds_bounded_mps_validation_20260510.csv"

FIG1_SRC_PDF = ROOT / "figures/suds_ai_redraw_20260510/fig1_saig_suds_interface/Fig1_SAIG_SUDS_Interface_academic_2864w.pdf"
FIG1_SRC_PNG = ROOT / "figures/suds_ai_redraw_20260510/fig1_saig_suds_interface/Fig1_SAIG_SUDS_Interface_academic_2864w.png"
FIG3_SRC_PDF = ROOT / "figures/suds_ai_redraw_20260510/fig3_suds_ternary_policy/Fig3_SUDS_Ternary_Policy_selected_literature_style_2864w.pdf"
FIG3_SRC_PNG = ROOT / "figures/suds_ai_redraw_20260510/fig3_suds_ternary_policy/Fig3_SUDS_Ternary_Policy_selected_literature_style_2864w.png"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8.5,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.4,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.5,
    }
)

COLORS = {
    "KEEP": "#2C7BB6",
    "DEGRADE": "#F0A202",
    "PRUNE": "#D7191C",
    "E2": "#B23A48",
    "E3": "#377EB8",
    "E4": "#2E8B57",
    "E7": "#7B3294",
    "BASE": "#333333",
    "GRAY": "#777777",
}


@dataclass(frozen=True)
class FigureMeta:
    figure_id: str
    manuscript_tier: str
    title: str
    canonical_stem: str
    figure_family: str
    evidence_label: str
    primary_source_path: str
    caption_scope_note: str


FIGURES: dict[str, FigureMeta] = {
    "fig1": FigureMeta(
        "Fig1",
        "main",
        "SAIG and SUDS interface",
        "Fig1_SAIGAndSUDSInterface",
        "ai_schematic",
        "ai_schematic",
        "docs/reports/task_briefs/20260510_suds_fig1_saig_interface_dual_reference_brief.md",
        "Modeled methodology schematic only; no silicon, SPICE, real GLUE, or hardware-measurement claim.",
    ),
    "fig2": FigureMeta(
        "Fig2",
        "main",
        "Slack signal availability and independence",
        "Fig2_SlackSignalAvailability",
        "data_figure",
        "modeled_analytical",
        "experiments/results/runs/slack_manifest.json",
        "Slack availability from HOPS/DPTC timeline-derived manifest; L1 independence is a scoped statistical annotation.",
    ),
    "fig3": FigureMeta(
        "Fig3",
        "main",
        "SUDS ternary quality-budget policy",
        "Fig3_SUDSTernaryPolicy",
        "ai_schematic",
        "ai_schematic",
        "docs/reports/task_briefs/20260510_suds_fig3_ternary_policy_dual_reference_brief.md",
        "Modeled policy schematic only; no silicon, SPICE, measured accuracy, real GLUE, or hardware-measurement claim.",
    ),
    "fig4": FigureMeta(
        "Fig4",
        "main",
        "Main modeled accuracy-energy trade-off",
        "Fig4_ModeledAccuracyEnergyTradeoff",
        "data_figure",
        "modeled_analytical",
        "experiments/results/runs/phase_c/phase_c_summary.json",
        "Modeled ADC energy reduction against analytical accuracy-impact drop; not measured hardware energy.",
    ),
    "fig5": FigureMeta(
        "Fig5",
        "main",
        "Tier distribution and ADC energy waterfall",
        "Fig5_TierDistributionADCWaterfall",
        "data_figure",
        "modeled_analytical",
        "experiments/results/runs/phase_c/phase_c_summary.json",
        "Mechanism explanation under the analytical ADC model; 4-bit ADC cost is modeled as 1x base and 8-bit as 16x base.",
    ),
    "fig6": FigureMeta(
        "Fig6",
        "main",
        "Bounded MPS validation sanity check",
        "Fig6_BoundedMPSValidation",
        "data_figure",
        "bounded_mps_validation",
        "experiments/results/report_data/suds_bounded_mps_validation_20260510.csv",
        "MobileViT-S, 5000 ImageNet samples, 4 seeds, MPS backend; bounded validation only.",
    ),
    "appf1": FigureMeta(
        "AppF1",
        "appendix",
        "Full SUDS threshold scan",
        "AppF1_ThresholdScan",
        "data_figure",
        "modeled_analytical",
        "experiments/results/runs/phase_c/phase_c_summary.json",
        "Full threshold-grid sensitivity under the analytical model.",
    ),
    "appf2": FigureMeta(
        "AppF2",
        "appendix",
        "SUDS + L1 overlay sensitivity",
        "AppF2_SUDSL1OverlaySensitivity",
        "data_figure",
        "modeled_analytical",
        "experiments/results/runs/phase_d/phase_d_summary.json",
        "Slack plus L1 overlay sensitivity; not a direct prior-work equivalence claim.",
    ),
    "appf3": FigureMeta(
        "AppF3",
        "appendix",
        "Synthetic BERT/GLUE-style stress test",
        "AppF3_SyntheticProfileStress",
        "data_figure",
        "synthetic_supporting",
        "experiments/results/runs/phase_e/phase_e_summary.json",
        "Synthetic BERT/GLUE-style profile only; not real BERT inference or real GLUE evaluation.",
    ),
    "appf4": FigureMeta(
        "AppF4",
        "appendix",
        "Parametric PHY link-budget check",
        "AppF4_ParametricPHYCheck",
        "data_figure",
        "parametric_supporting",
        "experiments/results/runs/phase_f/phase_f_summary.json",
        "Parametric optical link-budget check only; not SPICE-level or silicon PHY closure.",
    ),
}


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def load_phase_results(phase_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for phase in ["phase_b", "phase_c", "phase_d", "phase_e", "phase_f"]:
        path = phase_dir / phase / f"{phase}_summary.json"
        if path.exists():
            data[phase] = json.loads(path.read_text(encoding="utf-8"))
    slack_path = phase_dir / "slack_manifest.json"
    if slack_path.exists():
        data["slack"] = json.loads(slack_path.read_text(encoding="utf-8"))
    return data


def save_multi(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for ext in ["pdf", "svg", "png"]:
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path)
        outputs[ext] = rel(path)
    plt.close(fig)
    return outputs


def copy_asset(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Missing accepted schematic asset: {src}")
    shutil.copy2(src, dst)


def render_fig1(_data: dict[str, Any], output_dir: Path) -> dict[str, str]:
    meta = FIGURES["fig1"]
    pdf = output_dir / f"{meta.canonical_stem}.pdf"
    png = output_dir / f"{meta.canonical_stem}.png"
    copy_asset(FIG1_SRC_PDF, pdf)
    copy_asset(FIG1_SRC_PNG, png)
    return {"pdf": rel(pdf), "png": rel(png)}


def render_fig3(_data: dict[str, Any], output_dir: Path) -> dict[str, str]:
    meta = FIGURES["fig3"]
    pdf = output_dir / f"{meta.canonical_stem}.pdf"
    png = output_dir / f"{meta.canonical_stem}.png"
    copy_asset(FIG3_SRC_PDF, pdf)
    copy_asset(FIG3_SRC_PNG, png)
    return {"pdf": rel(pdf), "png": rel(png)}


def slack_arrays(data: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    slack_data = data.get("slack", {})
    names: list[str] = []
    bubbles: list[float] = []
    all_slack: list[float] = []
    for key, row in slack_data.items():
        if key == "_global":
            continue
        cols = row.get("total_slack_norm", [])
        if not cols:
            continue
        names.append(key.split(":")[-1])
        bubble = row.get("bubble_fraction", row.get("layer_slack_norm", 0.0))
        bubbles.append(float(np.mean(bubble)) if isinstance(bubble, list) else float(bubble))
        all_slack.extend(float(v) for v in cols)
    return names, np.asarray(bubbles), np.asarray(all_slack)


def representative_slack(data: dict[str, Any]) -> np.ndarray:
    best = np.array([])
    best_key = ""
    for key, row in data.get("slack", {}).items():
        if key == "_global":
            continue
        cols = np.asarray(row.get("total_slack_norm", []), dtype=float)
        if len(cols) > len(best):
            best = cols
            best_key = key
        if len(cols) == 192 and "fusion" in key.lower():
            return cols
    return best if len(best) else np.linspace(0.8, 0.0, 192)


def render_fig2(data: dict[str, Any], output_dir: Path) -> dict[str, str]:
    meta = FIGURES["fig2"]
    _names, bubbles, all_slack = slack_arrays(data)
    if all_slack.size == 0:
        raise ValueError("No slack data loaded for Fig2")
    col_slack = representative_slack(data)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.25), gridspec_kw={"width_ratios": [1.15, 1.0, 1.1]})
    order = np.argsort(bubbles)
    sorted_bubbles = bubbles[order]
    x = np.arange(len(sorted_bubbles))

    axes[0].plot(x, sorted_bubbles * 100.0, color=COLORS["E3"], linewidth=1.0)
    axes[0].fill_between(x, sorted_bubbles * 100.0, color=COLORS["E3"], alpha=0.18)
    axes[0].set_title("(a) Layer bubble range")
    axes[0].set_xlabel("Layer sorted by bubble fraction")
    axes[0].set_ylabel("Bubble fraction (%)")
    axes[0].annotate("4.6%-14.2%\nCV=0.237", xy=(0.05, 0.83), xycoords="axes fraction", fontsize=7.2)

    cx = np.arange(len(col_slack))
    axes[1].plot(cx, col_slack, color=COLORS["E4"], linewidth=1.0)
    axes[1].fill_between(cx, col_slack, color=COLORS["E4"], alpha=0.16)
    axes[1].axhline(0.10, color=COLORS["DEGRADE"], linestyle="--", linewidth=0.8)
    axes[1].axhline(0.70, color=COLORS["PRUNE"], linestyle="--", linewidth=0.8)
    axes[1].set_title("(b) Column slack gradient")
    axes[1].set_xlabel("Column index")
    axes[1].set_ylabel("Combined slack")
    axes[1].annotate(f"N={len(col_slack)}", xy=(0.06, 0.86), xycoords="axes fraction", fontsize=7.2)

    axes[2].hist(all_slack, bins=42, color=COLORS["E4"], edgecolor="white", linewidth=0.25, alpha=0.88)
    axes[2].axvline(0.10, color=COLORS["DEGRADE"], linestyle="--", linewidth=0.9, label=r"$\tau_{low}$")
    axes[2].axvline(0.70, color=COLORS["PRUNE"], linestyle="--", linewidth=0.9, label=r"$\tau_{high}$")
    axes[2].set_title("(c) Combined slack tiers")
    axes[2].set_xlabel("Combined slack")
    axes[2].set_ylabel("Column count")
    axes[2].legend(loc="upper right", frameon=False)
    axes[2].annotate(
        "CV=0.369\nrho(slack,L1)=-0.22\np=0.19, n=9,312",
        xy=(0.98, 0.58),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7.0,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#BBBBBB", alpha=0.92),
    )

    fig.suptitle("Slack signal availability and independence", y=1.03, fontsize=10.0)
    fig.tight_layout()
    return save_multi(fig, output_dir, meta.canonical_stem)


def render_fig4(data: dict[str, Any], output_dir: Path) -> dict[str, str]:
    meta = FIGURES["fig4"]
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.85))
    pc = data.get("phase_c", {})
    pb = data.get("phase_b", {})

    e2 = pc.get("E2", {})
    e2_points = [e2[k] for k in sorted(e2) if k.startswith("s_")]
    ax.plot(
        [p["acc_drop_pp"] for p in e2_points],
        [p["energy_reduction_ratio"] for p in e2_points],
        "s-",
        color=COLORS["E2"],
        markersize=4.5,
        linewidth=1.0,
        label="E2 L1 binary",
    )

    e3 = pb.get("E3", {})
    e3_points = [e3[k] for k in sorted(e3) if k.startswith("s_")]
    ax.plot(
        [p.get("avg_acc_drop_pp", p.get("acc_drop_pp", 0.0)) for p in e3_points],
        [p["energy_reduction_ratio"] for p in e3_points],
        "o--",
        color=COLORS["E3"],
        markersize=4.5,
        linewidth=1.0,
        label="E3 slack binary",
    )

    e4 = pc.get("E4", {})
    e4_points = sorted(e4.values(), key=lambda row: (row["tau_low"], row["tau_high"]))
    ax.scatter(
        [p["acc_drop_pp"] for p in e4_points],
        [p["energy_reduction_ratio"] for p in e4_points],
        marker="D",
        s=24,
        color=COLORS["E4"],
        edgecolor="white",
        linewidth=0.4,
        label="E4 SUDS ternary",
        zorder=4,
    )
    default = e4.get("tau_0.10_0.70")
    if default:
        ax.scatter(
            [default["acc_drop_pp"]],
            [default["energy_reduction_ratio"]],
            marker="D",
            s=58,
            color=COLORS["E4"],
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
        )
        ax.annotate(
            "default tau=(0.10,0.70)\n92.5% ADC reduction\n0.33 pp analytical drop",
            xy=(default["acc_drop_pp"], default["energy_reduction_ratio"]),
            xytext=(0.72, 0.54),
            textcoords="data",
            fontsize=6.8,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#B8B8B8", linewidth=0.45, alpha=0.94),
            arrowprops=dict(arrowstyle="->", linewidth=0.55, color=COLORS["GRAY"], shrinkB=3),
        )

    e7_best = None
    for mr in data.get("phase_d", {}).get("models", {}).values():
        s0 = mr.get("seeds", {}).get("seed_0", {})
        vals = list(s0.get("E7", {}).values())
        if vals:
            candidate = max(vals, key=lambda row: row["energy_reduction_ratio"])
            if e7_best is None or candidate["energy_reduction_ratio"] > e7_best["energy_reduction_ratio"]:
                e7_best = candidate
    if e7_best:
        ax.scatter(
            [e7_best["acc_drop_pp"]],
            [e7_best["energy_reduction_ratio"]],
            marker="*",
            s=82,
            color=COLORS["E7"],
            edgecolor="white",
            linewidth=0.4,
            label="E7 +L1 overlay",
            zorder=5,
        )

    ax.set_xlabel("Analytical accuracy-impact drop (pp)")
    ax.set_ylabel("Modeled ADC energy reduction")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Modeled accuracy-energy trade-off")
    fig.tight_layout()
    return save_multi(fig, output_dir, meta.canonical_stem)


def selected_configs(pc: dict[str, Any]) -> list[dict[str, Any]]:
    e4 = pc.get("E4", {})
    keys = ["tau_0.10_0.50", "tau_0.10_0.60", "tau_0.10_0.70", "tau_0.20_0.50"]
    return [e4[k] for k in keys if k in e4]


def render_fig5(data: dict[str, Any], output_dir: Path) -> dict[str, str]:
    meta = FIGURES["fig5"]
    pc = data.get("phase_c", {})
    configs = selected_configs(pc)
    if not configs:
        raise ValueError("No E4 configs loaded for Fig5")

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65), gridspec_kw={"width_ratios": [1.3, 1.0]})
    labels = [f"({c['tau_low']:.2f},{c['tau_high']:.2f})" for c in configs]
    x = np.arange(len(configs))
    keep = np.asarray([c["keep_ratio"] for c in configs])
    degrade = np.asarray([c["degrade_ratio"] for c in configs])
    prune = np.asarray([c["prune_ratio"] for c in configs])

    axes[0].bar(x, keep, color=COLORS["KEEP"], label="KEEP: 8-bit ADC")
    axes[0].bar(x, degrade, bottom=keep, color=COLORS["DEGRADE"], label="DEGRADE: 4-bit ADC")
    axes[0].bar(x, prune, bottom=keep + degrade, color=COLORS["PRUNE"], label="PRUNE: zero ADC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel(r"SUDS thresholds $(\tau_{low},\tau_{high})$")
    axes[0].set_ylabel("Fraction of columns")
    axes[0].set_ylim(0, 1)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_title("(a) Tier distribution")
    handles, legend_labels = axes[0].get_legend_handles_labels()

    default = pc.get("E4", {}).get("tau_0.10_0.70", configs[-1])
    dense = 16.0
    binary = 16.0 * (1.0 - default["prune_ratio"])
    suds_keep = default["keep_ratio"] * 16.0
    suds_degrade = default["degrade_ratio"] * 1.0
    bars_x = np.arange(3)
    axes[1].bar(bars_x[0], dense, color=COLORS["BASE"], width=0.58, label="8-bit energy")
    axes[1].bar(bars_x[1], binary, color=COLORS["E3"], width=0.58)
    axes[1].bar(bars_x[2], suds_keep, color=COLORS["KEEP"], width=0.58, label="KEEP 8-bit")
    axes[1].bar(bars_x[2], suds_degrade, bottom=suds_keep, color=COLORS["DEGRADE"], width=0.58, label="DEGRADE 4-bit")
    axes[1].set_xticks(bars_x)
    axes[1].set_xticklabels(["Uniform\n8-bit", "Binary\nprune", "SUDS\nternary"])
    axes[1].set_ylabel("ADC energy (x base)")
    axes[1].set_title("(b) ADC energy waterfall")
    for i, val in enumerate([dense, binary, suds_keep + suds_degrade]):
        reduction = 1.0 - val / dense
        axes[1].text(i, val + 0.45, f"{val:.2f}x\n({reduction:.1%} red.)", ha="center", fontsize=7.0)
    axes[1].annotate("4-bit=1x base\n8-bit=16x base", xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=7.1)
    axes[1].set_ylim(0, 18.2)

    fig.suptitle("Why modeled ADC energy falls", y=1.02, fontsize=10.0)
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    return save_multi(fig, output_dir, meta.canonical_stem)


def read_validation_rows() -> list[dict[str, str]]:
    with VALIDATION_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render_fig6(_data: dict[str, Any], output_dir: Path) -> dict[str, str]:
    meta = FIGURES["fig6"]
    rows = read_validation_rows()
    seed_rows = [r for r in rows if r["row_type"] == "seed"]
    summary_rows = {r["metric"]: r for r in rows if r["row_type"] == "summary"}
    seeds = [r["seed"] for r in seed_rows]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55), gridspec_kw={"width_ratios": [1.2, 1.0]})
    x = np.arange(len(seed_rows))
    e3_top1 = np.asarray([float(r["e3_delta_top1_pp"]) for r in seed_rows])
    e4_top1 = np.asarray([float(r["e4_delta_top1_pp"]) for r in seed_rows])
    axes[0].axhline(0, color="#999999", linewidth=0.6)
    axes[0].plot(x, e3_top1, "o-", color=COLORS["E3"], linewidth=1.0, label="E3 binary")
    axes[0].plot(x, e4_top1, "D-", color=COLORS["E4"], linewidth=1.0, label="E4 SUDS")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(seeds)
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Top-1 delta vs baseline (pp)")
    axes[0].set_title("(a) Four-seed bounded Top-1 check")
    axes[0].legend(loc="lower right", frameon=False)
    axes[0].annotate("ranking preserved\n4/4 seeds", xy=(0.03, 0.08), xycoords="axes fraction", fontsize=7.2)

    metrics = ["E3 Top-1", "E4 Top-1", "E3 Top-5", "E4 Top-5"]
    means = [
        float(summary_rows["e3_delta_top1_pp"]["mean_pp"]),
        float(summary_rows["e4_delta_top1_pp"]["mean_pp"]),
        float(summary_rows["e3_delta_top5_pp"]["mean_pp"]),
        float(summary_rows["e4_delta_top5_pp"]["mean_pp"]),
    ]
    stds = [
        float(summary_rows["e3_delta_top1_pp"]["std_pp"]),
        float(summary_rows["e4_delta_top1_pp"]["std_pp"]),
        float(summary_rows["e3_delta_top5_pp"]["std_pp"]),
        float(summary_rows["e4_delta_top5_pp"]["std_pp"]),
    ]
    colors = [COLORS["E3"], COLORS["E4"], COLORS["E3"], COLORS["E4"]]
    y = np.arange(len(metrics))
    axes[1].barh(y, means, xerr=stds, color=colors, alpha=0.88, capsize=3)
    axes[1].axvline(0, color="#999999", linewidth=0.6)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(metrics)
    axes[1].set_xlabel("Mean delta (pp), std error bar")
    axes[1].set_title("(b) Aggregate deltas")
    for yi, mean, std in zip(y, means, stds):
        axes[1].text(mean - 0.15, yi, f"{mean:.2f} +/- {std:.2f}", va="center", ha="right", fontsize=7.0, color="white")
    axes[1].annotate(
        "MobileViT-S, 5000 samples, MPS",
        xy=(0.03, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=7.1,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#BBBBBB", alpha=0.9),
    )

    fig.suptitle("Bounded MPS validation sanity check", y=1.04, fontsize=10.0)
    fig.tight_layout()
    return save_multi(fig, output_dir, meta.canonical_stem)


def render_appf1(data: dict[str, Any], output_dir: Path) -> dict[str, str]:
    meta = FIGURES["appf1"]
    e4 = list(data.get("phase_c", {}).get("E4", {}).values())
    tau_lows = sorted({row["tau_low"] for row in e4})
    tau_highs = sorted({row["tau_high"] for row in e4})
    energy = np.full((len(tau_lows), len(tau_highs)), np.nan)
    drop = np.full_like(energy, np.nan)
    for row in e4:
        i = tau_lows.index(row["tau_low"])
        j = tau_highs.index(row["tau_high"])
        energy[i, j] = row["energy_reduction_ratio"]
        drop[i, j] = row["acc_drop_pp"]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
    for ax, mat, title, fmt in [
        (axes[0], energy, "Modeled ADC energy reduction", ".1%"),
        (axes[1], drop, "Analytical accuracy-impact drop (pp)", ".2f"),
    ]:
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(tau_highs)))
        ax.set_xticklabels([f"{v:.2f}" for v in tau_highs])
        ax.set_yticks(np.arange(len(tau_lows)))
        ax.set_yticklabels([f"{v:.2f}" for v in tau_lows])
        ax.set_xlabel(r"$\tau_{high}$")
        ax.set_ylabel(r"$\tau_{low}$")
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=6.8, color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Full SUDS threshold scan", y=1.03, fontsize=10)
    fig.tight_layout()
    return save_multi(fig, output_dir, meta.canonical_stem)


def render_appf2(data: dict[str, Any], output_dir: Path) -> dict[str, str]:
    meta = FIGURES["appf2"]
    fig, ax = plt.subplots(1, 1, figsize=(4.7, 3.1))
    markers = {"mobilevit_xxs": "s", "mobilevit_xs": "^", "mobilevit_s": "D"}
    labels = {"mobilevit_xxs": "XXS", "mobilevit_xs": "XS", "mobilevit_s": "S"}
    for model, mr in data.get("phase_d", {}).get("models", {}).items():
        s0 = mr.get("seeds", {}).get("seed_0", {})
        for key, color, linestyle, name in [("E4", COLORS["E4"], "-", "SUDS"), ("E7", COLORS["E7"], "--", "SUDS+L1")]:
            vals = sorted(s0.get(key, {}).values(), key=lambda row: row["acc_drop_pp"])
            if not vals:
                continue
            ax.plot(
                [v["acc_drop_pp"] for v in vals],
                [v["energy_reduction_ratio"] for v in vals],
                marker=markers.get(model, "o"),
                linestyle=linestyle,
                linewidth=0.9,
                markersize=4.2,
                color=color,
                alpha=0.78,
                label=f"{name} {labels.get(model, model)}",
            )
    ax.set_xlabel("Analytical accuracy-impact drop (pp)")
    ax.set_ylabel("Modeled ADC energy reduction")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("SUDS + L1 overlay sensitivity")
    ax.legend(loc="lower right", ncol=2, frameon=False)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return save_multi(fig, output_dir, meta.canonical_stem)


def render_appf3(data: dict[str, Any], output_dir: Path) -> dict[str, str]:
    meta = FIGURES["appf3"]
    pe = data.get("phase_e", {})
    tasks = list(pe.get("tasks", {}).keys()) or ["MNLI", "QQP", "SST-2", "MRPC"]
    details = pe.get("C5a_gate", {}).get("task_details", {})
    e2_vals: list[float] = []
    e3_vals: list[float] = []
    e4_vals: list[float] = []
    for task in tasks:
        task_payload = pe.get("tasks", {}).get(task, {})
        row = details.get(task, {})
        e2_vals.append(float(row.get("E2_energy_red", task_payload.get("E2", {}).get("s_0.5", {}).get("energy_reduction_ratio", 0.0))))
        e3_vals.append(float(row.get("E3_energy_red", task_payload.get("E3", {}).get("s_0.5", {}).get("energy_reduction_ratio", 0.0))))
        e4_vals.append(float(row.get("E4_energy_red", 0.0)))

    fig, ax = plt.subplots(1, 1, figsize=(4.6, 2.75))
    x = np.arange(len(tasks))
    width = 0.24
    ax.bar(x - width, e2_vals, width, color=COLORS["E2"], label="E2 L1 binary")
    ax.bar(x, e3_vals, width, color=COLORS["E3"], label="E3 slack binary")
    ax.bar(x + width, e4_vals, width, color=COLORS["E4"], label="E4 SUDS ternary")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Modeled ADC energy reduction")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 0.9)
    ax.set_title("Synthetic BERT/GLUE-style profile stress", pad=8)
    ax.legend(loc="upper left", ncol=1, frameon=False)
    ax.annotate(
        "synthetic profile labels only\nnot real GLUE evaluation",
        xy=(0.98, 0.93),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7.0,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#BBBBBB", linewidth=0.45, alpha=0.92),
    )
    fig.tight_layout()
    return save_multi(fig, output_dir, meta.canonical_stem)


def render_appf4(data: dict[str, Any], output_dir: Path) -> dict[str, str]:
    meta = FIGURES["appf4"]
    phy_config = data.get("phase_f", {}).get("phy_config", {})
    configs = selected_configs(data.get("phase_c", {}))
    if not configs:
        configs = list(data.get("phase_c", {}).get("E4", {}).values())
    total_wdm = float(phy_config.get("wdm_channels_n", 64))
    xtalk_db = float(phy_config.get("crosstalk", {}).get("xtalk_db", -25.0))
    er_db = float(phy_config.get("er_db", 6.0))
    loss_db = float(sum(phy_config.get("loss_path_db", {}).values()) or 17.0)
    p_sens = float(phy_config.get("p_sensitivity_dbm", -22.0))
    pp_ext = float(phy_config.get("pp_extinction_db", 1.5))
    margin = float(phy_config.get("margin_db", 4.0))

    prune = np.asarray([c["prune_ratio"] for c in configs])
    active = np.asarray([max(1, int(round(total_wdm * (1.0 - c["prune_ratio"])))) for c in configs])
    xtalk_values: list[float] = []
    laser_values: list[float] = []
    er_linear = 10 ** (er_db / 10.0)
    xtalk_linear = 10 ** (xtalk_db / 10.0)
    for aw in active:
        interference = max(0.0, (float(aw) - 1.0) * xtalk_linear * (1.0 + 1.0 / er_linear))
        penalty = 10.0 * np.log10(1.0 + interference)
        xtalk_values.append(float(penalty))
        laser_values.append(float(p_sens + loss_db + penalty + pp_ext + margin))
    xtalk = np.asarray(xtalk_values)
    laser = np.asarray(laser_values)
    order = np.argsort(prune)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35))
    axes[0].plot(prune[order], active[order], "o-", color=COLORS["E3"], linewidth=1.0)
    axes[0].set_xlabel("PRUNE ratio")
    axes[0].set_ylabel("Active WDM channels")
    axes[0].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_title("(a) Active channels")

    axes[1].plot(prune[order], xtalk[order], "D-", color=COLORS["E4"], linewidth=1.0)
    axes[1].set_xlabel("PRUNE ratio")
    axes[1].set_ylabel("Crosstalk penalty (dB)")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_title("(b) Crosstalk penalty")

    axes[2].plot(prune[order], laser[order], "s-", color=COLORS["E7"], linewidth=1.0)
    axes[2].axhline(20, color=COLORS["PRUNE"], linestyle="--", linewidth=0.8, label="20 dBm ceiling")
    axes[2].set_xlabel("PRUNE ratio")
    axes[2].set_ylabel("Required laser power (dBm)")
    axes[2].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[2].set_title("(c) Laser range")
    axes[2].legend(loc="upper right", frameon=False)
    fig.suptitle("Parametric PHY link-budget check", y=1.04, fontsize=10)
    fig.tight_layout()
    return save_multi(fig, output_dir, meta.canonical_stem)


RENDERERS: dict[str, Callable[[dict[str, Any], Path], dict[str, str]]] = {
    "fig1": render_fig1,
    "fig2": render_fig2,
    "fig3": render_fig3,
    "fig4": render_fig4,
    "fig5": render_fig5,
    "fig6": render_fig6,
    "appf1": render_appf1,
    "appf2": render_appf2,
    "appf3": render_appf3,
    "appf4": render_appf4,
}


def render_command(figure_key: str, output_dir: Path, phase_dir: Path) -> str:
    return (
        "python3 experiments/tools/render_suds_figures.py "
        f"--phase-dir {rel(phase_dir)} --output-dir {rel(output_dir)} --figure {figure_key}"
    )


def write_traceability(output_dir: Path, outputs: dict[str, dict[str, str]], phase_dir: Path) -> None:
    path = output_dir / TRACEABILITY_NAME
    fields = [
        "figure_id",
        "run_tag",
        "manuscript_tier",
        "figure_file",
        "artifact_path",
        "all_outputs",
        "evidence_label",
        "primary_source_path",
        "render_command",
        "caption_scope_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in RENDERERS:
            if key not in outputs:
                continue
            meta = FIGURES[key]
            figure_file = outputs[key].get("pdf") or next(iter(outputs[key].values()))
            writer.writerow(
                {
                    "figure_id": meta.figure_id,
                    "run_tag": RUN_TAG,
                    "manuscript_tier": meta.manuscript_tier,
                    "figure_file": figure_file,
                    "artifact_path": figure_file,
                    "all_outputs": json.dumps(outputs[key], sort_keys=True),
                    "evidence_label": meta.evidence_label,
                    "primary_source_path": meta.primary_source_path,
                    "render_command": render_command(key, output_dir, phase_dir),
                    "caption_scope_note": meta.caption_scope_note,
                }
            )


def write_registry(output_dir: Path) -> None:
    path = output_dir / REGISTRY_NAME
    fields = [
        "figure_id",
        "numbering_status",
        "manuscript_tier",
        "figure_family",
        "title",
        "canonical_stem",
        "source_kind",
        "source_record",
        "notes",
    ]
    trace_rel = rel(output_dir / TRACEABILITY_NAME)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in RENDERERS:
            meta = FIGURES[key]
            writer.writerow(
                {
                    "figure_id": meta.figure_id,
                    "numbering_status": "active",
                    "manuscript_tier": meta.manuscript_tier,
                    "figure_family": meta.figure_family,
                    "title": meta.title,
                    "canonical_stem": meta.canonical_stem,
                    "source_kind": "traceability_csv",
                    "source_record": trace_rel,
                    "notes": meta.caption_scope_note,
                }
            )


def write_pack_metadata(output_dir: Path, outputs: dict[str, dict[str, str]], phase_dir: Path) -> None:
    payload = {
        "run_tag": RUN_TAG,
        "phase_dir": rel(phase_dir),
        "pack_dir": rel(output_dir),
        "main_figures": ["Fig1", "Fig2", "Fig3", "Fig4", "Fig5", "Fig6"],
        "appendix_figures": ["AppF1", "AppF2", "AppF3", "AppF4"],
        "outputs": outputs,
        "claim_boundary": "SUDS Q2+ repaired figure pack; modeled, analytical, bounded MPS, synthetic-supporting, and parametric-supporting evidence only.",
    }
    (output_dir / "pack_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render SUDS Q2+ repaired paper figures.")
    parser.add_argument("--phase-dir", type=Path, default=ROOT / "experiments/results/runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / f"figures/paper_figures_{RUN_TAG}")
    parser.add_argument("--figure", default="all", help="Figure key, e.g. fig2/appf1, or all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    phase_dir = (ROOT / args.phase_dir).resolve() if not args.phase_dir.is_absolute() else args.phase_dir.resolve()
    output_dir = (ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_phase_results(phase_dir)
    keys = list(RENDERERS.keys()) if args.figure == "all" else [args.figure.lower()]
    outputs: dict[str, dict[str, str]] = {}
    for key in keys:
        if key not in RENDERERS:
            raise SystemExit(f"Unknown figure key: {args.figure}")
        outputs[key] = RENDERERS[key](data, output_dir)
        print(f"[suds-figures] {key}: {outputs[key]}")

    if args.figure == "all":
        write_traceability(output_dir, outputs, phase_dir)
        write_registry(output_dir)
        write_pack_metadata(output_dir, outputs, phase_dir)
        print(f"[suds-figures] wrote {rel(output_dir / TRACEABILITY_NAME)}")
        print(f"[suds-figures] wrote {rel(output_dir / REGISTRY_NAME)}")
    print(f"[suds-figures] done: {rel(output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
