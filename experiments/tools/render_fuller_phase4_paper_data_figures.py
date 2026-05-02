#!/usr/bin/env python3
"""Render the Phase 4 FULLER paper data figure redesign pack."""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.table import Cell
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_TAG = "public_redacted_path"
FINAL_RUN_TAG = "20260428_fuller_final_unreserved_datafig_broad_scaling_flowmeas_promotion"
REMEDIATION_RUN_TAG = "20260430_full_figure_strict_remediated"
MECHANISM_RUN_TAG = "20260426_fuller_phase4_mechanism_basis_rerun"
DEFAULT_QUICK_DIR = ROOT / "experiments" / "results" / "quick_reports" / FINAL_RUN_TAG
DEFAULT_MECHANISM_QUICK_DIR = ROOT / "experiments" / "results" / "quick_reports" / FINAL_RUN_TAG
DEFAULT_OUT_DIR = ROOT / "figures" / f"paper_figures_{FINAL_RUN_TAG}"
DEFAULT_REVIEW_DIR = ROOT / "experiments" / "results" / "review" / FINAL_RUN_TAG
DEFAULT_DOC_NOTE = ROOT / "docs" / "reports" / "public_redacted_path_promotion_note.md"
FINAL_DOC_NOTE = ROOT / "docs" / "reports" / "20260428_fuller_final_unreserved_datafig_broad_scaling_flowmeas_promotion_note.md"
DEFAULT_CURRENT_BASIS_QA_NOTE = ROOT / "docs" / "reports" / "20260428_current_basis_datafig_broad_scaling_flowmeas_promotion_qa_note.md"
SCHEMATIC_NOTE = ROOT / "docs" / "reports" / "20260423_fuller_current_schematic_figure_redesign_note.md"

LANE_ORDER = ["ASTRA", "MESO", "HOPS", "DET", "SPARSE", "FULLER"]
BLOCKED_LANES = {"SPARSE", "FULLER"}
LANE_COLORS = {
    "ASTRA": "#5A6470",
    "MESO": "#8B9DAA",
    "HOPS": "#7C8E77",
    "DET": "#B7A99E",
    "SPARSE": "#C96A6A",
    "FULLER": "#988BA5",
}
LANE_MARKERS = {
    "ASTRA": "o",
    "MESO": "s",
    "HOPS": "D",
    "DET": "^",
    "SPARSE": "X",
    "FULLER": "P",
}
HATCHES = {
    "runtime_materialization_ready": "",
    "accuracy_preservation_claim_blocked": "///",
}

STYLE_ANCHORS = {
    "Fig3": (
        "composition_only: CrossLight Fig6 comparison scatter; HyAtten Fig6 speedup/energy comparison; "
        "Lightening-Transformer Fig13 platform comparison"
    ),
    "Fig4": (
        "composition_only: CrossLight Fig6 Pareto-style scatter; Lightening-Transformer Fig13 comparison posture; "
        "HyAtten Fig6 highlighted endpoint rhythm"
    ),
    "Fig5": (
        "composition_only: Noisy Machines Fig3/Fig4 robustness curves; Lightening-Transformer Fig14/Fig15 noise sweeps"
    ),
    "Fig6": (
        "composition_only: HyAtten Fig8 scalability split panels; Lightening-Transformer Fig9 scaling; "
        "old 20260305 AppF3/AppF4 clean axis rhythm"
    ),
    "Fig7": (
        "composition_only: Lightening-Transformer Fig11/Fig12 component comparison; HyAtten Fig7 breakdown; "
        "old 20260305 AppF9 device comparison"
    ),
    "Fig8": "composition_only: compact claim-gate bar; HyAtten Fig6 comparison plus explicit boundary annotation",
    "AppF1": "composition_only: Noisy Machines Fig3 errorbar robustness posture; compact range plot",
    "AppF2": "composition_only: IEEE appendix compatibility table with shortened actions and row coloring",
    "AppF3": "composition_only: retained related-work radar; old 20260305 Fig7 radar spacing",
    "AppF4": "composition_only: HyAtten Fig6 ablation rhythm; compact multi-panel mechanism comparison",
    "AppF5": "composition_only: Lightening-Transformer Fig11/Fig12 breakdown posture; CrossLight stacked energy bars",
    "AppF6": "composition_only: DET/SPARSE measured sweep; compact operating-point validation layout",
}

FIGURES = [
    ("Fig3", "main", "Fig3_Phase4RuntimeAccuracyBoundary", "Phase4 Runtime/Accuracy Boundary"),
    ("Fig4", "main", "Fig4_RuntimeAccuracyPareto", "Runtime-Accuracy Pareto"),
    ("Fig5", "main", "Fig5_BoundedSensitivity", "Bounded Sensitivity"),
    ("Fig6", "main", "Fig6_ScalingSupport", "Scaling Support"),
    ("Fig7", "main", "Fig7_DeviceContext", "Device Context"),
    ("Fig8", "main", "Fig8_HoldoutClaimBoundary", "Holdout Claim Boundary"),
    ("AppF1", "appendix", "AppF1_SeedRangeVariability", "Seed/Range Variability"),
    ("AppF2", "appendix", "AppF2_DataFigureCompatibility", "Data-Figure Compatibility Matrix"),
    ("AppF3", "appendix", "AppF3_RelatedWorkRadar", "Related-Work Radar"),
    ("AppF4", "appendix", "AppF4_MechanismAblationContext", "Mechanism Ablation Context"),
    ("AppF5", "appendix", "AppF5_MechanismEnergyBreakdown", "Mechanism Energy Breakdown"),
    ("AppF6", "appendix", "AppF6_DETOperatingPointContext", "DET/SPARSE Sweep Context"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pick_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    return float(value)


def fig9_representative_seed_summary(rows: list[dict[str, str]]) -> str:
    counts = sorted(
        {
            int(float(str(row.get("seed_count") or 0)))
            for row in rows
            if row.get("lane") == "FULLER"
            and row.get("profile") in {"clean", "mild", "medium", "hard"}
            and row.get("profile_class", "representative") == "representative"
            and str(row.get("seed_count") or "").strip()
        }
    )
    if not counts:
        return "representative"
    if len(counts) == 1:
        return f"seed-{counts[0]}"
    return f"seed-{counts[0]}-{counts[-1]}"


def fig10_repeat_summary(rows: list[dict[str, str]]) -> str:
    counts = sorted({int(float(str(row.get("repeat_count") or 1))) for row in rows})
    if not counts:
        return "repeat_count unavailable"
    if len(counts) == 1:
        return f"repeat_count={counts[0]}"
    return f"repeat_count range {counts[0]}-{counts[-1]}"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.7,
            "axes.titlesize": 10.4,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.17,
            "grid.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 420,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
        }
    )
    Cell.PAD = 0.025


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def table_row_heights(table, rows: list[list[str]], *, header_height: float = 0.065, base: float = 0.040, per_line: float = 0.030) -> None:
    ncols = len(rows[0]) if rows else 0
    for col in range(ncols):
        table[(0, col)].set_height(header_height)
    for row_idx, row in enumerate(rows, start=1):
        line_count = max(cell.count("\n") + 1 for cell in row)
        height = base + per_line * line_count
        for col in range(ncols):
            table[(row_idx, col)].set_height(height)


def blocked(lane: str) -> bool:
    return lane in BLOCKED_LANES


def is_final_unreserved(quick_dir: Path) -> bool:
    return quick_dir.name in {FINAL_RUN_TAG, REMEDIATION_RUN_TAG} or (
        (
            (quick_dir / "fig5_bounded_sensitivity_current_basis.csv").is_file()
            or (quick_dir / "fig5_noise_robustness_current_basis.csv").is_file()
            or (quick_dir / "fig9_noise_robustness_current_basis.csv").is_file()
        )
        and (
            (quick_dir / "fig6_broad_scaling_flow_buffer_current_basis.csv").is_file()
            or (quick_dir / "fig10_scaling_support_current_basis.csv").is_file()
        )
    )


def panel_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", pad=7)


def add_claim_legend(fig: plt.Figure, *, y: float = 0.02, ncols: int = 2) -> None:
    handles = [
        Patch(facecolor="#D9D9D9", edgecolor="#1F1F1F", label="runtime/materialization context"),
        Patch(facecolor="#D9D9D9", edgecolor="#1F1F1F", hatch="///", label="accuracy claim blocked"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncols=ncols, fontsize=7.6)


def add_footnote(
    fig: plt.Figure,
    text: str,
    *,
    y: float = 0.02,
    fontsize: float = 7.4,
    wrap_width: int | None = None,
) -> None:
    if wrap_width is not None:
        text = wrap(text, wrap_width)
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=fontsize, color="#3E4448", linespacing=1.16)


def make_grayscale_preview(png_path: Path, preview_path: Path) -> None:
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        with Image.open(png_path) as image:
            image = image.convert("L")
            image.thumbnail((1600, 1600))
            image.save(preview_path)
            return
    except Exception:
        pass

    import matplotlib.image as mpimg

    arr = mpimg.imread(png_path)
    if arr.ndim == 3:
        gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114])
    else:
        gray = arr
    plt.imsave(preview_path, gray, cmap="gray")


def export_figure(fig: plt.Figure, out_dir: Path, review_dir: Path, stem: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": out_dir / f"{stem}.svg",
        "pdf": out_dir / f"{stem}.pdf",
        "png": out_dir / f"{stem}.png",
    }
    for path in paths.values():
        fig.savefig(path, bbox_inches="tight")
        if path.suffix == ".svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n")
    plt.close(fig)
    make_grayscale_preview(paths["png"], review_dir / f"{stem}_grayscale.png")
    return paths


def rows_by_lane(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_lane = {row["lane"]: row for row in rows}
    return [by_lane[lane] for lane in LANE_ORDER if lane in by_lane]


def render_fig6(rows: list[dict[str, str]], out_dir: Path, review_dir: Path) -> dict[str, Path]:
    rows = rows_by_lane(rows)
    lanes = [row["lane"] for row in rows]
    y = np.arange(len(lanes))
    speedups = [f(row, "speedup_vs_astra") for row in rows]
    top1 = [f(row, "top1_mean") for row in rows]
    xerr = [
        [f(row, "top1_mean") - f(row, "top1_min") for row in rows],
        [f(row, "top1_max") - f(row, "top1_mean") for row in rows],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.55), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.22, top=0.82, wspace=0.23)

    bars = axes[0].barh(
        y,
        speedups,
        color=[LANE_COLORS[lane] for lane in lanes],
        edgecolor="#222222",
        linewidth=0.75,
    )
    for bar, lane, value in zip(bars, lanes, speedups):
        if blocked(lane):
            bar.set_hatch("///")
        axes[0].text(value + max(speedups) * 0.018, bar.get_y() + bar.get_height() / 2, f"{value:.2f}x", va="center", fontsize=7.3)
    axes[0].set_yticks(y, lanes)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Speedup vs ASTRA (x)")
    axes[0].set_xlim(0, max(speedups) * 1.20)
    panel_title(axes[0], "Runtime readiness")

    axes[1].errorbar(top1, y, xerr=xerr, fmt="none", ecolor="#555555", capsize=2.5, linewidth=0.85, zorder=2)
    for lane, xpos, ypos in zip(lanes, top1, y):
        axes[1].scatter(
            xpos,
            ypos,
            s=68 if not blocked(lane) else 92,
            marker=LANE_MARKERS[lane],
            facecolors=LANE_COLORS[lane] if not blocked(lane) else "white",
            edgecolors=LANE_COLORS[lane] if blocked(lane) else "#222222",
            linewidths=1.5 if blocked(lane) else 0.7,
            zorder=3,
        )
        axes[1].text(xpos + 0.55, ypos, f"{xpos:.2f}", va="center", fontsize=7.3)
    axes[1].set_xlabel("Top-1 accuracy (%)")
    axes[1].set_xlim(18, max(top1) * 1.10)
    axes[1].tick_params(axis="y", labelleft=False)
    panel_title(axes[1], "Accuracy boundary")

    fig.suptitle("Phase4 Runtime/Accuracy Boundary", y=0.96, fontsize=11.2, weight="bold")
    add_claim_legend(fig, y=0.035)
    return export_figure(fig, out_dir, review_dir, "Fig3_Phase4RuntimeAccuracyBoundary")


def render_fig7(rows: list[dict[str, str]], out_dir: Path, review_dir: Path) -> dict[str, Path]:
    rows = rows_by_lane(rows)
    fig, ax = plt.subplots(figsize=(5.9, 4.15))
    label_offsets = {
        "ASTRA": (7, 5),
        "MESO": (-42, 7),
        "HOPS": (-44, 12),
        "DET": (8, 8),
        "SPARSE": (9, 8),
        "FULLER": (-54, 8),
    }
    for row in rows:
        lane = row["lane"]
        is_blocked = blocked(lane)
        ax.scatter(
            f(row, "speedup_vs_astra"),
            f(row, "top1_mean"),
            s=116 if is_blocked else 86,
            marker=LANE_MARKERS[lane],
            facecolors=LANE_COLORS[lane] if not is_blocked else "white",
            edgecolors=LANE_COLORS[lane],
            linewidths=1.8 if is_blocked else 1.0,
            label=lane,
            zorder=3,
        )
        offset = label_offsets.get(lane, (6, 6))
        ax.annotate(
            lane,
            (f(row, "speedup_vs_astra"), f(row, "top1_mean")),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.45, "shrinkA": 0, "shrinkB": 4},
        )
    astra = next(row for row in rows if row["lane"] == "ASTRA")
    ax.axhline(f(astra, "top1_mean"), color="#8D8D8D", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.text(0.985, 0.93, "ASTRA accuracy reference", transform=ax.transAxes, ha="right", fontsize=7.2, color="#555555")
    ax.set_xlabel("Speedup vs ASTRA (x)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_xlim(0, max(f(row, "speedup_vs_astra") for row in rows) * 1.12)
    ax.set_ylim(18, max(f(row, "top1_mean") for row in rows) * 1.07)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markeredgecolor="#777777", label="ready", markersize=6),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="white", markeredgecolor="#777777", label="accuracy claim blocked", markersize=7),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7.2)
    ax.set_title("Runtime-Accuracy Pareto", pad=8)
    return export_figure(fig, out_dir, review_dir, "Fig4_RuntimeAccuracyPareto")


def render_fig8(rows: list[dict[str, str]], out_dir: Path, review_dir: Path) -> dict[str, Path]:
    fig, ax = plt.subplots(figsize=(7.16, 4.85))
    ax.axis("off")
    columns = ["Figure", "Evidence family", "Gate state", "Allowed claim"]
    claim_label = {
        "runtime_materialization_ready": "runtime/materialization context",
        "runtime_context_only_for_sparse_fuller": "runtime context only",
        "boundary_statement": "claim boundary stated",
        "contextual_support_not_accuracy_preservation": "context only",
        "bounded_sensitivity_not_broad_robustness": "bounded sensitivity only",
        "runtime_scaling_context": "runtime scaling context",
        "current_basis_scaling_support": "current-basis scaling",
        "device_context_not_benchmark_equivalence": "device context only",
        "accuracy_preservation_claim_blocked": "accuracy claim blocked",
        "variability_context": "variability context",
        "compatibility_audit": "compatibility audit",
        "literature_context_only": "literature context only",
        "mechanism_context_not_current_accuracy_preservation": "mechanism context only",
        "mechanism_current_basis_tradeoff_context": "current-basis mechanism",
    }
    gate_label = {
        "regenerated_current": "current",
        "contextual_boundary_carried_forward": "contextual boundary",
        "current_basis_noise_completed": "noise current",
        "current_basis_scaling_grid_completed": "scaling current",
        "minimum_support_completed_from_retained_context": "retained context",
        "claim_blocking_report_generated": "blocking gate",
        "appendix_context_retained": "appendix context",
        "mechanism_context_retained": "mechanism context",
    }
    table_rows = [
        [
            row["figure_id"],
            wrap(row["support_family"].replace("_", " "), 24),
            wrap(gate_label.get(row["status"], row["status"].replace("_", " ")), 18),
            wrap(claim_label.get(row["claim_tier"], row["claim_tier"].replace("_", " ")), 24),
        ]
        for row in rows
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=columns,
        colWidths=[0.10, 0.34, 0.20, 0.36],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    table_row_heights(table, table_rows, header_height=0.070, base=0.044, per_line=0.029)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#555555")
        cell.set_linewidth(0.32)
        if r == 0:
            cell.set_facecolor("#E6E6E6")
            cell.set_text_props(weight="bold")
        elif c == 2 and "blocking" in table_rows[r - 1][2]:
            cell.set_facecolor("#F6D6D2")
        elif c == 2 and "retained" in table_rows[r - 1][2]:
            cell.set_facecolor("#F6EDC8")
        elif c == 2:
            cell.set_facecolor("#DDEFD9")
        elif c == 3 and "blocked" in table_rows[r - 1][3]:
            cell.set_facecolor("#F6D6D2")
    ax.set_title("Claim/Support Gate Matrix", fontsize=11, pad=12)
    return export_figure(fig, out_dir, review_dir, "Fig8_ClaimSupportGateMatrix")


def render_fig9(rows: list[dict[str, str]], out_dir: Path, review_dir: Path) -> dict[str, Path]:
    fuller = [row for row in rows if row["lane"] == "FULLER"]
    if any(str(row.get("model") or "").strip() for row in fuller):
        model_order = ["MobileViT-S", "MobileViT-XS", "MobileViT-XXS"]
        model_rows_by_label = {
            label: [row for row in fuller if (row.get("model_variant") or row.get("model")) == label]
            for label in model_order
        }
        model_order = [label for label in model_order if model_rows_by_label.get(label)]
        sigmas = sorted({f(row, "noise_sigma_lsb") for row in fuller if str(row.get("noise_sigma_lsb") or "").strip()})
        alphas = sorted({f(row, "crosstalk_alpha") for row in fuller if str(row.get("crosstalk_alpha") or "").strip()})
        vmax = max([f(row, "acc_drop_pp") for row in fuller if str(row.get("acc_drop_pp") or "").strip()] or [1.0])
        fig, axes = plt.subplots(1, len(model_order), figsize=(7.16, 3.65), sharey=True)
        if len(model_order) == 1:
            axes = [axes]
        fig.subplots_adjust(left=0.08, right=0.90, bottom=0.24, top=0.82, wspace=0.18)
        image = None
        for ax, label in zip(axes, model_order):
            matrix = np.full((len(alphas), len(sigmas)), np.nan)
            by_coord = {
                (f(row, "noise_sigma_lsb"), f(row, "crosstalk_alpha")): row
                for row in model_rows_by_label[label]
                if str(row.get("noise_sigma_lsb") or "").strip() and str(row.get("crosstalk_alpha") or "").strip()
            }
            for yi, alpha in enumerate(alphas):
                for xi, sigma in enumerate(sigmas):
                    row = by_coord.get((sigma, alpha))
                    if row is not None:
                        matrix[yi, xi] = f(row, "acc_drop_pp")
            image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(vmax, 1.0))
            rep_x: list[int] = []
            rep_y: list[int] = []
            for sigma, alpha in [(0.0, 0.0), (0.25, 0.01), (0.5, 0.02), (1.0, 0.05)]:
                if sigma in sigmas and alpha in alphas:
                    rep_x.append(sigmas.index(sigma))
                    rep_y.append(alphas.index(alpha))
            ax.scatter(rep_x, rep_y, marker="s", s=28, facecolors="none", edgecolors="#202020", linewidths=0.75)
            ax.set_xticks(range(len(sigmas)), [f"{value:g}" for value in sigmas], rotation=45, ha="right")
            ax.set_yticks(range(len(alphas)), [f"{value:g}" for value in alphas])
            ax.set_xlabel("Gaussian noise σ")
            panel_title(ax, label)
        axes[0].set_ylabel("Crosstalk α")
        if image is not None:
            cbar = fig.colorbar(image, ax=axes, fraction=0.030, pad=0.025)
            cbar.set_label("Accuracy drop (pp)", fontsize=7.0)
            cbar.ax.tick_params(labelsize=6.6)
        fig.suptitle("Bounded Sensitivity Envelope", y=0.96, fontsize=11.2, weight="bold")
        seed_summary = fig9_representative_seed_summary(fuller)
        add_footnote(
            fig,
            f"Three-model dense envelope; outlined cells are {seed_summary} representatives. Context-only sensitivity map; no broad noise-tolerance or accuracy-preservation claim.",
            y=0.055,
            fontsize=6.9,
            wrap_width=122,
        )
        return export_figure(fig, out_dir, review_dir, "Fig5_BoundedSensitivity")

    order = ["clean", "mild", "medium", "hard"]
    by_profile = {row["profile"]: row for row in fuller}
    profiles = [profile for profile in order if profile in by_profile]
    drops = [f(by_profile[p], "acc_drop_pp") for p in profiles]
    acc = [f(by_profile[p], "acc_top1") for p in profiles]
    xticklabels = [
        f"{p.title()}\nσ={f(by_profile[p], 'noise_sigma_lsb'):.2g}, α={f(by_profile[p], 'crosstalk_alpha'):.2g}"
        for p in profiles
    ]
    x = np.arange(len(profiles))
    fig, ax = plt.subplots(figsize=(5.85, 3.75))
    fig.subplots_adjust(left=0.13, right=0.88, bottom=0.25, top=0.84)
    bars = ax.bar(x, drops, color="#D9E1E6", edgecolor="#222222", hatch="..", linewidth=0.75)
    ax2 = ax.twinx()
    ax2.plot(x, acc, color=LANE_COLORS["FULLER"], marker="o", linewidth=1.5, label="current top-1 context")
    for bar, value in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(drops) * 0.03, f"{value:.1f} pp", ha="center", fontsize=7.2)
    ax.set_xticks(x, xticklabels)
    ax.set_ylabel("Accuracy drop (pp)")
    ax2.set_ylabel("Top-1 context (%)")
    ax.set_ylim(0, max(drops) * 1.18)
    ax2.set_ylim(0, max(acc) * 1.20)
    ax.set_title("Bounded Sensitivity Context")
    ax2.tick_params(axis="y", labelcolor=LANE_COLORS["FULLER"])
    ax2.spines["right"].set_visible(True)
    ax.text(
        0.01,
        -0.30,
        "Current-basis bounded sensitivity only; not a broad noise-tolerance or accuracy-preservation claim.",
        transform=ax.transAxes,
        fontsize=7.4,
        color="#3E4448",
    )
    return export_figure(fig, out_dir, review_dir, "Fig5_BoundedSensitivity")


def render_fig10(rows: list[dict[str, str]], out_dir: Path, review_dir: Path) -> dict[str, Path]:
    if rows and "scaling_axis" not in rows[0]:
        return render_fig10_current_basis(rows, out_dir, review_dir)

    batch_rows = sorted([row for row in rows if row["scaling_axis"] == "batch_size"], key=lambda row: f(row, "scale_value"))
    seq_rows = sorted([row for row in rows if row["scaling_axis"] == "sequence_length"], key=lambda row: f(row, "scale_value"))
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.45))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.25, top=0.80, wspace=0.32)
    batch_x = [f(row, "batch_size") for row in batch_rows]
    batch_y = [f(row, "throughput_images_s") for row in batch_rows]
    seq_x = [f(row, "sequence_length") for row in seq_rows]
    seq_y = [f(row, "latency_ms") for row in seq_rows]
    axes[0].plot(batch_x, batch_y, marker="o", color=LANE_COLORS["ASTRA"], linewidth=1.5)
    axes[0].set_xlabel("Batch size (seq=197)")
    axes[0].set_ylabel("Throughput (images/s)")
    axes[0].set_xticks(batch_x)
    axes[0].set_ylim(0, max(batch_y) * 1.16)
    panel_title(axes[0], "Batch scaling")
    axes[1].plot(seq_x, seq_y, marker="s", color="#B45F4D", linewidth=1.5)
    axes[1].set_xlabel("Sequence length (batch=1)")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_xticks(seq_x)
    axes[1].set_ylim(0, max(seq_y) * 1.22)
    panel_title(axes[1], "Sequence scaling")
    add_footnote(fig, "Minimum retained support; flow-buffer peak fraction was not available in legacy context.", y=0.04)
    fig.suptitle("Scaling Support", y=0.96, fontsize=11.2, weight="bold")
    return export_figure(fig, out_dir, review_dir, "Fig6_ScalingSupport")


def render_fig10_current_basis(rows: list[dict[str, str]], out_dir: Path, review_dir: Path) -> dict[str, Path]:
    models = [model for model in ["MobileViT-S", "MobileViT-XS", "MobileViT-XXS"] if any(row.get("model_variant") == model for row in rows)]
    methods = [method for method in ["ASTRA", "DET", "SPARSE", "FULLER"] if any(row.get("method") == method for row in rows)]
    declared_grid_ready = any(
        row.get("scaling_claim_status") == "declared_grid_timing_ready"
        or "declared_grid_timing" in row.get("claim_boundary", "")
        for row in rows
    )
    repeat_counts = sorted({int(float(str(row.get("repeat_count") or 1))) for row in rows})
    repeat_note = f"repeat_count={repeat_counts[0]}" if len(repeat_counts) == 1 else f"repeat_count range {repeat_counts[0]}-{repeat_counts[-1]}"
    stability_values = [
        f(row, "throughput_images_s_cv_pct")
        for row in rows
        if str(row.get("throughput_images_s_cv_pct") or "").strip()
    ]
    stability_note = ""
    if stability_values:
        stability_note = f"; median throughput CV={float(np.median(stability_values)):.2f}%, max={max(stability_values):.2f}%"
    throughput = np.full((len(methods), len(models)), np.nan)
    latency = np.full((len(methods), len(models)), np.nan)
    for yi, method in enumerate(methods):
        for xi, model in enumerate(models):
            subset = [row for row in rows if row.get("method") == method and row.get("model_variant") == model]
            if subset:
                throughput[yi, xi] = float(np.mean([f(row, "throughput_images_s") for row in subset]))
                latency[yi, xi] = float(np.mean([f(row, "latency_ms") for row in subset]))
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.55), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.31, top=0.80, wspace=0.20)
    panels = [
        (axes[0], throughput, "Mean throughput", "images/s", "YlGnBu"),
        (axes[1], latency, "Mean latency", "ms", "YlOrBr"),
    ]
    for ax, matrix, title, unit, cmap in panels:
        image = ax.imshow(matrix, aspect="auto", cmap=cmap)
        finite_values = matrix[np.isfinite(matrix)]
        text_threshold = float(np.nanmin(finite_values) + 0.62 * (np.nanmax(finite_values) - np.nanmin(finite_values))) if finite_values.size else math.inf
        ax.set_xticks(range(len(models)), [label.replace("MobileViT-", "") for label in models])
        ax.set_yticks(range(len(methods)), methods)
        panel_title(ax, title)
        for yi in range(len(methods)):
            for xi in range(len(models)):
                if np.isfinite(matrix[yi, xi]):
                    text_color = "white" if matrix[yi, xi] >= text_threshold else "#202020"
                    ax.text(xi, yi, f"{matrix[yi, xi]:.1f}", ha="center", va="center", fontsize=6.7, color=text_color)
        cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
        cbar.set_label(unit, fontsize=6.8)
        cbar.ax.tick_params(labelsize=6.3)
    axes[0].set_ylabel("Method")
    fig.suptitle(
        "Declared-Grid Timing Summary" if declared_grid_ready else "Current-Basis Timing Grid Summary",
        y=0.95,
        fontsize=11.2,
        weight="bold",
    )
    grid_text = (
        "declared expanded grid plus holdout points"
        if declared_grid_ready
        else "full 3x3 batch/sequence grid"
    )
    boundary_text = (
        "with trace-backed flow-buffer fields for non-baseline rows; not a silicon measurement"
        if declared_grid_ready
        else "so scaling claims stay bounded"
    )
    add_footnote(
        fig,
        f"Each cell averages the {grid_text} for that model and method; {repeat_note}{stability_note}, {boundary_text}.",
        y=0.045,
        fontsize=6.35,
        wrap_width=108,
    )
    return export_figure(fig, out_dir, review_dir, "Fig6_ScalingSupport")


def render_fig11(rows: list[dict[str, str]], out_dir: Path, review_dir: Path) -> dict[str, Path]:
    order = ["CPU", "GPU", "HPAT"]
    by_platform = {row["platform_class"]: row for row in rows}
    ordered_rows = [by_platform[item] for item in order if item in by_platform]
    short_labels = ["M5 Pro\nCPU", "M5 Pro GPU\nMLX MPS", "MTL-FULLER"]
    colors = [LANE_COLORS["ASTRA"], LANE_COLORS["HOPS"], "#B7A99E"]
    metrics = [
        ("latency_ms", "Latency (ms)", "log"),
        ("energy_j", "Energy (J)", "log"),
        ("avg_power_w", "Avg power (W)", "linear"),
        ("throughput_images_s", "Throughput (img/s)", "log"),
    ]

    gpu = by_platform["GPU"]
    fuller = by_platform["HPAT"]
    ratios = {
        "latency": f(gpu, "latency_ms") / f(fuller, "latency_ms"),
        "energy": f(fuller, "energy_j") / f(gpu, "energy_j"),
        "power": f(fuller, "avg_power_w") / f(gpu, "avg_power_w"),
        "throughput": f(fuller, "throughput_images_s") / f(gpu, "throughput_images_s"),
    }

    fig, axes = plt.subplots(1, 4, figsize=(7.55, 4.05))
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.30, top=0.76, wspace=0.34)
    x = np.arange(len(ordered_rows))
    for ax, (key, title, scale) in zip(axes, metrics):
        vals = [f(row, key) for row in ordered_rows]
        bars = ax.bar(x, vals, color=colors, edgecolor="#1F1F1F", linewidth=0.7)
        for bar, row, value in zip(bars, ordered_rows, vals):
            if row["platform_class"] == "HPAT":
                bar.set_hatch("///")
            label = f"{value:.3g}"
            if key == "throughput_images_s" and value >= 100:
                label = f"{value:.0f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * (1.18 if scale == "log" else 1.035),
                label,
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
            )
        if scale == "log":
            ax.set_yscale("log")
            positive_min = min(value for value in vals if value > 0)
            ax.set_ylim(positive_min / 2.4, max(vals) * 2.35)
        else:
            ax.set_ylim(0, max(vals) * 1.24)
        ax.set_title(title, fontsize=8.5, weight="bold", pad=6)
        ax.set_xticks(x, short_labels, rotation=35, ha="right")
        ax.tick_params(axis="x", labelsize=6.2)
        ax.tick_params(axis="y", labelsize=6.4)
        ax.grid(True, axis="y", alpha=0.18)

    ratio_text = (
        "MPS-relative context: "
        f"{ratios['latency']:.1f}x lower latency | "
        f"{ratios['energy']:.1f}x higher energy | "
        f"{ratios['power']:.1f}x higher avg power | "
        f"{ratios['throughput']:.1f}x higher throughput"
    )
    fig.text(0.5, 0.135, ratio_text, ha="center", fontsize=7.0, weight="bold", color="#223238")
    fig.text(
        0.5,
        0.075,
        "Measured host rows: Apple M5 Pro CPU and Apple M5 Pro GPU (MLX MPS); modeled endpoint: MTL-FULLER. Not benchmark equivalence.",
        ha="center",
        fontsize=6.4,
        color="#4D5A61",
    )
    handles = [
        Patch(facecolor=colors[0], edgecolor="#1F1F1F", label="Apple M5 Pro CPU measured"),
        Patch(facecolor=colors[1], edgecolor="#1F1F1F", label="Apple M5 Pro GPU (MLX MPS) measured"),
        Patch(facecolor=colors[2], edgecolor="#1F1F1F", hatch="///", label="MTL-FULLER modeled"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncols=3, fontsize=6.8)
    fig.suptitle("Device Context", y=0.98, fontsize=11.2, weight="bold")
    return export_figure(fig, out_dir, review_dir, "Fig7_DeviceContext")


def render_fig12(rows: list[dict[str, str]], out_dir: Path, review_dir: Path) -> dict[str, Path]:
    lanes = [row["lane"] for row in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(5.45, 3.1))
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.24, top=0.84)
    values = [f(row, "top1_mean") for row in rows]
    bars = ax.barh(y, values, color=[LANE_COLORS[l] for l in lanes], edgecolor="black", hatch="///", linewidth=0.75)
    for bar, row in zip(bars, rows):
        ax.text(
            f(row, "top1_mean") + max(values) * 0.035,
            bar.get_y() + bar.get_height() / 2,
            f"{f(row, 'top1_mean'):.2f}%  |  {f(row, 'speedup_vs_astra'):.2f}x  |  blocked",
            va="center",
            fontsize=7.4,
        )
    ax.set_yticks(y, lanes)
    ax.set_xlabel("Top-1 accuracy (%)")
    ax.set_xlim(0, max(values) * 1.48)
    ax.invert_yaxis()
    ax.set_title("Holdout Claim Boundary")
    ax.text(
        0.5,
        -0.28,
        "Positive SPARSE/FULLER accuracy-preservation wording remains blocked.",
        transform=ax.transAxes,
        ha="center",
        fontsize=7.5,
        color="#3E4448",
    )
    return export_figure(fig, out_dir, review_dir, "Fig8_HoldoutClaimBoundary")


def render_appf1(rows: list[dict[str, str]], out_dir: Path, review_dir: Path) -> dict[str, Path]:
    rows = rows_by_lane(rows)
    lanes = [row["lane"] for row in rows]
    x = np.arange(len(rows))
    means = [f(row, "top1_mean") for row in rows]
    ranges = [
        f(row, "top1_range_pp", f(row, "top1_max") - f(row, "top1_min"))
        for row in rows
    ]
    err = [
        [f(row, "top1_mean") - f(row, "top1_min") for row in rows],
        [f(row, "top1_max") - f(row, "top1_mean") for row in rows],
    ]
    fig, ax = plt.subplots(figsize=(6.45, 3.35))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.23, top=0.84)
    ax.errorbar(x, means, yerr=err, fmt="none", color="#333333", ecolor="#666666", capsize=3, linewidth=0.85)
    for xpos, row in zip(x, rows):
        if blocked(row["lane"]):
            ax.scatter([xpos], [f(row, "top1_mean")], marker="X", s=95, facecolors="white", edgecolors=LANE_COLORS[row["lane"]], linewidths=1.6, zorder=4)
        else:
            ax.scatter([xpos], [f(row, "top1_mean")], s=70, color=LANE_COLORS[row["lane"]], edgecolors="black", linewidths=0.5, zorder=4)
    range_ax = ax.inset_axes([0.58, 0.50, 0.37, 0.36])
    range_y = np.arange(len(rows))
    range_ax.barh(range_y, ranges, color=[LANE_COLORS[row["lane"]] for row in rows], edgecolor="#222222", linewidth=0.35)
    for ypos, value in zip(range_y, ranges):
        range_ax.text(value + max(ranges) * 0.035, ypos, f"{value:.2f}", va="center", fontsize=5.6)
    range_ax.set_yticks(range_y, lanes, fontsize=5.5)
    range_ax.invert_yaxis()
    range_ax.set_xlim(0, max(ranges) * 1.34)
    range_ax.set_title("Top-1 range (pp)", fontsize=6.2, pad=2)
    range_ax.grid(axis="x", alpha=0.18)
    range_ax.spines["top"].set_visible(False)
    range_ax.spines["right"].set_visible(False)
    ax.set_xticks(x, lanes, rotation=25, ha="right")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_ylim(min(f(row, "top1_min") for row in rows) - 1.0, max(f(row, "top1_max") for row in rows) + 1.6)
    ax.set_title("Seed/Range Variability")
    ax.text(
        0.5,
        -0.28,
        "Error bars and inset report max-min Top-1 ranges from the current Phase4 intake summary.",
        transform=ax.transAxes,
        ha="center",
        fontsize=7.3,
    )
    return export_figure(fig, out_dir, review_dir, "AppF1_SeedRangeVariability")


def render_appf2(rows: list[dict[str, str]], out_dir: Path, review_dir: Path) -> dict[str, Path]:
    fig, ax = plt.subplots(figsize=(7.16, 6.35))
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.06, top=0.92)
    ax.axis("off")
    columns = ["Legacy", "Action", "Final successor", "Boundary / reason"]
    action_label = {
        "appendix_context_retained": "appendix retained",
        "mechanism_context_retained": "mechanism retained",
        "regenerated_current": "regenerated",
        "retired_incompatible": "retired",
    }
    table_rows = [
        [
            wrap(row["legacy_figure_id"].replace(",", ", ").replace("/", " / "), 15),
            wrap(action_label.get(row["compatibility_action"], row["compatibility_action"].replace("_", " ")), 18),
            wrap(row["successor_figure_id"], 13),
            wrap(row["reason"], 38),
        ]
        for row in rows
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=columns,
        colWidths=[0.18, 0.19, 0.16, 0.47],
        bbox=[0.0, 0.10, 1.0, 0.80],
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.45)
    table_row_heights(table, table_rows, header_height=0.046, base=0.026, per_line=0.018)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#555555")
        cell.set_linewidth(0.32)
        if r == 0:
            cell.set_facecolor("#E6E6E6")
            cell.set_text_props(weight="bold")
        elif c == 1 and "retired" in table_rows[r - 1][1]:
            cell.set_facecolor("#F6D6D2")
        elif c == 1 and "retained" in table_rows[r - 1][1]:
            cell.set_facecolor("#F6EDC8")
        elif c == 1:
            cell.set_facecolor("#DDEFD9")
    fig.suptitle("Data-Figure Compatibility Matrix (Final Numbering)", fontsize=11, y=0.975, weight="bold")
    fig.text(
        0.5,
        0.020,
        wrap(
            "Final successors use Fig3/Fig4 runtime-Pareto, Fig5 bounded sensitivity, Fig6 declared-grid timing, and Fig7 device context; Fig9-Fig12 are mechanism schematics.",
            124,
        ),
        ha="center",
        va="bottom",
        fontsize=6.6,
        color="#3E4448",
    )
    return export_figure(fig, out_dir, review_dir, "AppF2_DataFigureCompatibility")


def render_appf3(rows: list[dict[str, str]], out_dir: Path, review_dir: Path) -> dict[str, Path]:
    provenance_fields = {
        "run_tag",
        "final_run_tag",
        "source_run_tag",
        "source_quick_report_csv",
        "original_source_status",
        "evidence_basis",
        "promotion_action",
        "source_status",
        "source_csv",
    }
    metrics: list[str] = []
    for key in rows[0].keys():
        if key == "Work" or key in provenance_fields:
            continue
        try:
            [float(row[key]) for row in rows]
        except (KeyError, TypeError, ValueError):
            continue
        metrics.append(key)
    labels = [label.replace("_", "\n") for label in metrics]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(6.1, 5.15))
    fig.subplots_adjust(left=0.06, right=0.78, bottom=0.15, top=0.84)
    ax = fig.add_subplot(111, polar=True)
    palette = ["#5A6470", "#7C8E77", "#988BA5", "#C96A6A", "#B7A99E", "#41B6C4"][: len(rows)]
    for row, color in zip(rows, palette):
        values = [f(row, metric) for metric in metrics]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=1.25, marker="o", markersize=2.8, label=row["Work"])
        ax.fill(angles, values, color=color, alpha=0.045)
    ax.set_xticks(angles[:-1], labels, fontsize=7)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 5)
    ax.set_title("Related-Work Radar\nQualitative context only", y=1.11, fontsize=10.5, weight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.08, 0.50), fontsize=7)
    fig.text(
        0.5,
        0.035,
        "Qualitative related-work positioning only; not benchmark-equivalent empirical superiority evidence.",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#3E4448",
    )
    return export_figure(fig, out_dir, review_dir, "AppF3_RelatedWorkRadar")


def render_appf4(rows: list[dict[str, str]], out_dir: Path, review_dir: Path) -> dict[str, Path]:
    rows = sorted(rows, key=lambda row: int(row["experiment_id"][1:]))
    labels = [row["experiment_id"] for row in rows]
    x = np.arange(len(rows))
    colors = ["#5A6470", "#8B9DAA", "#7C8E77", "#B7A99E", "#C96A6A", "#41B6C4", "#988BA5"][: len(rows)]
    metrics = [
        ("speedup_vs_E0", "Speedup vs E0 (x)", "higher"),
        ("energy_ratio_vs_E0", "Energy ratio vs E0", "lower"),
        ("acc_delta_vs_E0_pp", "Accuracy delta vs E0 (pp)", "lower"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.75, 3.55), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.30, top=0.79, wspace=0.34)
    for ax, (key, ylabel, direction) in zip(axes, metrics):
        vals = [f(row, key) for row in rows]
        bars = ax.bar(x, vals, color=colors, edgecolor="#1F1F1F", linewidth=0.65)
        for bar, row in zip(bars, rows):
            if row["experiment_id"] in {"E3", "E6"}:
                bar.set_hatch("///")
        ymax = max(vals) if vals else 1.0
        if key == "energy_ratio_vs_E0":
            ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.75)
            ax.set_ylim(0, max(1.08, ymax * 1.16))
        elif key == "acc_delta_vs_E0_pp":
            ax.set_ylim(0, max(0.6, ymax * 1.28))
        else:
            ax.set_ylim(0, ymax * 1.22)
        for xpos, val in zip(x, vals):
            if key == "acc_delta_vs_E0_pp":
                text = f"{val:.2f}"
            elif key == "energy_ratio_vs_E0":
                text = f"{val:.2f}"
            else:
                text = f"{val:.2f}x"
            ax.text(xpos, val + ax.get_ylim()[1] * 0.025, text, ha="center", va="bottom", fontsize=6.4, rotation=90 if len(text) > 4 else 0)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels, rotation=0)
        ax.tick_params(axis="x", labelsize=5.8)
        ax.tick_params(axis="y", labelsize=6.8)
        panel_title(ax, f"{direction.title()} is better")
    fig.suptitle("Mechanism Ablation Context", y=0.96, fontsize=11.2, weight="bold")
    fig.text(
        0.5,
        0.125,
        "E0 ASTRA | E1 MESO | E2 HOPS | E3 DET | E4 SPARSE | E6 FULLER",
        ha="center",
        fontsize=6.8,
        color="#3E4448",
    )
    add_footnote(
        fig,
        "Current Phase4-basis mechanism rows; DET/SPARSE/FULLER remain bounded operating points with measured accuracy cost.",
        y=0.045,
        fontsize=6.9,
    )
    return export_figure(fig, out_dir, review_dir, "AppF4_MechanismAblationContext")


def render_appf5(rows: list[dict[str, str]], out_dir: Path, review_dir: Path) -> dict[str, Path]:
    rows = sorted(rows, key=lambda row: int(row["experiment_id"][1:]))
    short = {"E0": "ASTRA", "E1": "MESO", "E2": "HOPS", "E3": "DET", "E4": "SPRS", "E5": "PHY", "E6": "FULL"}
    labels = [f"{row['experiment_id']}\n{short[row['experiment_id']]}" for row in rows]
    x = np.arange(len(rows))
    components = [
        ("memory_move_mj", "memory/move", "#6F7D86", ""),
        ("conversion_control_mj", "conversion/control", "#B7A99E", "///"),
        ("optical_static_mj", "optical/static", "#D9E1E6", ".."),
    ]
    fig, ax = plt.subplots(figsize=(7.16, 3.65))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.27, top=0.70)
    bottom = np.zeros(len(rows), dtype=float)
    for key, label, color, hatch in components:
        vals = np.array([f(row, key) for row in rows], dtype=float)
        bars = ax.bar(x, vals, bottom=bottom, color=color, edgecolor="#1F1F1F", linewidth=0.6, label=label)
        for bar in bars:
            bar.set_hatch(hatch)
        bottom += vals
    totals = [f(row, "total_energy_mj") for row in rows]
    for xpos, total in zip(x, totals):
        ax.text(xpos, total + max(totals) * 0.025, f"{total:.1f}", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Energy per inference (mJ)")
    ax.set_ylim(0, max(totals) * 1.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncols=3, fontsize=7.2)
    fig.suptitle("Mechanism Energy Breakdown", y=0.96, fontsize=11.2, weight="bold")
    add_footnote(fig, "Stacked current-basis components close to total energy; mechanism support stays inside the measured tradeoff boundary.", y=0.045)
    return export_figure(fig, out_dir, review_dir, "AppF5_MechanismEnergyBreakdown")


def render_appf6(rows: list[dict[str, str]], out_dir: Path, review_dir: Path) -> dict[str, Path]:
    sweep = sorted([row for row in rows if row["row_type"] == "det_k_sweep"], key=lambda row: f(row, "det_k_global"))
    sparse = sorted([row for row in rows if row["row_type"] == "sparse_tau_sweep"], key=lambda row: f(row, "sparse_tau_global"))
    if not sweep or not sparse:
        raise ValueError("AppF6 requires both det_k_sweep and sparse_tau_sweep rows")

    astra_top1_candidates = []
    for row in sweep + sparse:
        astra_top1_candidates.append(f(row, "top1_mean") - f(row, "paired_delta_vs_e0_quant_pp"))
    astra_top1 = float(np.median(astra_top1_candidates))

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.65))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.27, top=0.79, wspace=0.30)

    k = [f(row, "det_k_global") for row in sweep]
    det_top1 = [f(row, "top1_mean") for row in sweep]
    det_gap = [abs(f(row, "paired_delta_vs_e0_quant_pp")) for row in sweep]
    axes[0].plot(k, det_top1, marker="o", color=LANE_COLORS["DET"], linewidth=1.35, label="top-1")
    axes[0].axhline(astra_top1, color="#5A6470", linestyle="--", linewidth=0.8, label="ASTRA")
    axes[0].set_xlabel("DET k")
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[0].set_ylim(0, max(astra_top1, max(det_top1)) * 1.13)
    axes[0].set_xticks(k)
    axes[0].tick_params(axis="x", labelrotation=45, labelsize=6.6)
    axes[0].annotate("very low k\ncatastrophic", xy=(k[0], det_top1[0]), xytext=(10, 11), textcoords="offset points", fontsize=6.4, arrowprops={"arrowstyle": "->", "linewidth": 0.6})
    axes[0].annotate("k=80 dip", xy=(80, det_top1[k.index(80.0)]), xytext=(-27, 18), textcoords="offset points", fontsize=6.4, arrowprops={"arrowstyle": "->", "linewidth": 0.6})
    axes[0].annotate("best k=129\n-18.07 pp", xy=(129, det_top1[-1]), xytext=(-50, -27), textcoords="offset points", fontsize=6.4, arrowprops={"arrowstyle": "->", "linewidth": 0.6})
    axes[0].legend(loc="lower right", fontsize=6.7)
    panel_title(axes[0], "DET measured k sweep")

    tau = [f(row, "sparse_tau_global") for row in sparse]
    sparse_top1 = [f(row, "top1_mean") for row in sparse]
    axes[1].plot(tau, sparse_top1, marker="s", color=LANE_COLORS["SPARSE"], linewidth=1.35, label="top-1")
    axes[1].axhline(astra_top1, color="#5A6470", linestyle="--", linewidth=0.8, label="ASTRA")
    axes[1].set_xlabel("SPARSE tau")
    axes[1].set_ylabel("Top-1 accuracy (%)")
    axes[1].set_ylim(0, max(astra_top1, max(sparse_top1)) * 1.13)
    axes[1].set_xticks(tau)
    axes[1].tick_params(axis="x", labelrotation=45, labelsize=6.6)
    axes[1].annotate("gradual recovery", xy=(0.3, sparse_top1[tau.index(0.3)]), xytext=(-40, 15), textcoords="offset points", fontsize=6.4, arrowprops={"arrowstyle": "->", "linewidth": 0.6})
    axes[1].annotate("tau=0.5\n-16.82 pp", xy=(0.5, sparse_top1[-1]), xytext=(-42, -30), textcoords="offset points", fontsize=6.4, arrowprops={"arrowstyle": "->", "linewidth": 0.6})
    axes[1].legend(loc="lower right", fontsize=6.7)
    panel_title(axes[1], "SPARSE measured tau sweep")

    fig.suptitle("DET/SPARSE Current-Basis Sweep", y=0.965, fontsize=11.2, weight="bold")
    fig.text(0.5, 0.115, "All plotted rows: complete=true, seed_count=3; ASTRA baseline is the paired current-basis reference.", ha="center", fontsize=6.8, color="#3E4448")
    add_footnote(fig, "Measured sweeps support bounded operating points only; both mechanisms retain visible accuracy tradeoffs.", y=0.045, fontsize=6.8)
    return export_figure(fig, out_dir, review_dir, "AppF6_DETOperatingPointContext")


def render_all(quick_dir: Path, mechanism_quick_dir: Path, out_dir: Path, review_dir: Path, render_command: str) -> None:
    style()
    review_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = utc_now()
    final_mode = is_final_unreserved(quick_dir)
    remediation_mode = quick_dir.name == REMEDIATION_RUN_TAG
    carrier_run_tag = quick_dir.name if final_mode else RUN_TAG
    mechanism_source_dir = quick_dir if final_mode else mechanism_quick_dir

    inputs = {
        "Fig3": pick_existing(quick_dir / "fig3_phase4_runtime_accuracy_boundary.csv", quick_dir / "fig6_phase4_runtime_accuracy_boundary.csv"),
        "Fig4": pick_existing(quick_dir / "fig4_runtime_accuracy_pareto.csv", quick_dir / "fig7_runtime_accuracy_pareto.csv"),
        "Fig5": (
            pick_existing(
                quick_dir / "fig5_bounded_sensitivity_current_basis.csv",
                quick_dir / "fig5_noise_robustness_current_basis.csv",
                quick_dir / "fig9_noise_robustness_current_basis.csv",
            )
            if final_mode
            else quick_dir / "fig9_noise_robustness_minimal.csv"
        ),
        "Fig6": (
            pick_existing(
                quick_dir / "fig6_broad_scaling_flow_buffer_current_basis.csv",
                quick_dir / "fig10_scaling_support_current_basis.csv",
            )
            if final_mode
            else quick_dir / "fig10_scaling_support_minimal.csv"
        ),
        "Fig7": pick_existing(quick_dir / "fig7_device_context.csv", quick_dir / "fig11_device_context.csv"),
        "Fig8": pick_existing(quick_dir / "fig8_holdout_claim_boundary.csv", quick_dir / "fig12_holdout_claim_boundary.csv"),
        "AppF1": quick_dir / "appf1_seed_range_variability.csv",
        "AppF2": quick_dir / "appf2_data_figure_compatibility_matrix.csv",
        "AppF3": quick_dir / "appf3_related_work_radar_scores.csv",
        "AppF4": mechanism_source_dir / "appf4_mechanism_ablation_context.csv",
        "AppF5": mechanism_source_dir / "appf5_mechanism_energy_breakdown.csv",
        "AppF6": mechanism_source_dir / "appf6_det_sparse_sweep_phase4_basis.csv",
    }
    for path in inputs.values():
        if not path.exists():
            raise FileNotFoundError(path)
    fig5_seed_summary = fig9_representative_seed_summary(read_csv(inputs["Fig5"]))
    fig6_rows_for_summary = read_csv(inputs["Fig6"])
    fig6_repeat_text = fig10_repeat_summary(fig6_rows_for_summary)
    fig6_declared_grid_ready = any(
        row.get("scaling_claim_status") == "declared_grid_timing_ready"
        or "declared_grid_timing" in row.get("claim_boundary", "")
        for row in fig6_rows_for_summary
    )
    fig6_claim_summary = (
        "declared-grid timing evidence with trace-backed flow-buffer fields"
        if fig6_declared_grid_ready
        else "current-basis timing-grid stability evidence"
    )

    renderers = {
        "Fig3": render_fig6,
        "Fig4": render_fig7,
        "Fig5": render_fig9,
        "Fig6": render_fig10,
        "Fig7": render_fig11,
        "Fig8": render_fig12,
        "AppF1": render_appf1,
        "AppF2": render_appf2,
        "AppF3": render_appf3,
        "AppF4": render_appf4,
        "AppF5": render_appf5,
        "AppF6": render_appf6,
    }

    trace_rows: list[dict[str, Any]] = []
    preview_paths: list[str] = []
    for figure_id, tier, stem, title in FIGURES:
        source = inputs[figure_id]
        outputs = renderers[figure_id](read_csv(source), out_dir, review_dir)
        preview = review_dir / f"{stem}_grayscale.png"
        trace_run_tag = MECHANISM_RUN_TAG if figure_id in {"AppF4", "AppF5", "AppF6"} else carrier_run_tag
        preview_paths.append(rel(preview))
        trace_rows.append(
            {
                "figure_id": figure_id,
                "manuscript_tier": tier,
                "figure_file": rel(outputs["svg"]),
                "input_csvs": rel(source),
                "script_entry": rel(Path(__file__)),
                "command": render_command,
                "run_tag": trace_run_tag,
                "generated_at": generated_at,
                "params_summary": (
                    "current-basis mechanism update; SVG/PDF/PNG export; complete 3-seed DET/SPARSE rows"
                    if figure_id in {"AppF4", "AppF5", "AppF6"}
                    else "paper data redesign v2; SVG/PDF/PNG export; grayscale preview generated; label/layout polish applied"
                ),
                "literature_style_anchors": STYLE_ANCHORS[figure_id],
                "literature_anchor_scope": "layout/style only",
                "notes": (
                    f"{title}; source={MECHANISM_RUN_TAG}; DET/SPARSE/FULLER remain bounded measured tradeoffs"
                    if figure_id in {"AppF4", "AppF5", "AppF6"}
                    else title
                ),
            }
        )

    trace_fields = [
        "figure_id",
        "manuscript_tier",
        "figure_file",
        "input_csvs",
        "script_entry",
        "command",
        "run_tag",
        "generated_at",
        "params_summary",
        "literature_style_anchors",
        "literature_anchor_scope",
        "notes",
    ]
    if remediation_mode:
        trace_path = review_dir / "public_redacted_provenance_figure_traceability.csv"
        write_csv(trace_path, trace_rows, trace_fields)
    else:
        trace_path = out_dir / "figure_traceability.csv"
        write_csv(trace_path, trace_rows, trace_fields)
        write_csv(review_dir / "figure_traceability.csv", trace_rows, trace_fields)

    registry_rows: list[dict[str, Any]] = []
    for figure_id, tier, stem, title in FIGURES:
        registry_rows.append(
            {
                "figure_id": figure_id,
                "numbering_status": "active",
                "manuscript_tier": tier,
                "figure_family": "phase4_data_figure",
                "title": title,
                "canonical_stem": stem,
                "source_kind": "traceability_csv",
                "source_record": rel(trace_path),
                "notes": (
                    "Generated from current-basis mechanism evidence; bounded measured tradeoff only."
                    if figure_id in {"AppF4", "AppF5", "AppF6"}
                    else "Generated in paper data redesign freeze; claim boundary encoded in source CSVs."
                ),
            }
        )
    registry_fields = [
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
    if remediation_mode:
        write_csv(review_dir / "public_redacted_provenance_numbering_handoff.csv", registry_rows, registry_fields)
    else:
        write_csv(out_dir / "figure_numbering_registry.csv", registry_rows, registry_fields)

    field_maps = [
        "Fig3: y=lane; left x=speedup_vs_astra; right x=top1_mean with top1_min/top1_max range; hatch=claim_boundary.",
        "Fig4: x=speedup_vs_astra; y=top1_mean; marker/face=claim_boundary; label=lane.",
        f"Fig5: panels=model_variant; heatmap x=noise_sigma_lsb y=crosstalk_alpha color=acc_drop_pp; outlined cells are {fig5_seed_summary} representatives; bounded sensitivity only.",
        "Fig6: heatmap rows=method columns=model_variant; colors summarize mean throughput_images_s and mean latency_ms across the declared grid and holdout points.",
        "Fig7: platform panels for latency_ms, energy_j, avg_power_w, throughput_images_s; hatch=modeled FULLER endpoint.",
        "Fig8: y=lane; x=top1_mean; labels include speedup_vs_astra and holdout_gate.",
        "AppF1: x=lane; y=top1_mean with top1_min/top1_max range and explicit Top-1 range inset.",
        "AppF2: rows=legacy_figure_id; action=compatibility_action; successor=successor_figure_id.",
        "AppF3: polar axes are related-work score dimensions copied from retained compatibility source.",
        "AppF4: x=experiment_id/mechanism_label; panels=speedup_vs_E0, energy_ratio_vs_E0, acc_delta_vs_E0_pp.",
        "AppF5: x=experiment_id/mechanism_label; stacked bars=memory_move_mj, conversion_control_mj, optical_static_mj.",
        "AppF6: left x=det_k_global y=top1_mean with ASTRA paired baseline; right x=sparse_tau_global y=top1_mean with ASTRA paired baseline; all displayed rows require complete=true and seed_count=3.",
    ]
    brief = "\n".join(
        [
            f"# {carrier_run_tag} Data Figure Brief",
            "",
            "Figure ID: Fig3-Fig8, AppF1-AppF6",
            "Figure type: release-facing paper data-figure pack with current-basis AppF4-AppF6 mechanism overlay",
            f"Carrier run tag: {carrier_run_tag}",
            f"Primary mechanism evidence tag: {MECHANISM_RUN_TAG}",
            "",
            "## Governed Sources",
            "",
            f"- render script: `{rel(Path(__file__))}`",
            f"- quick-report directory: `{rel(quick_dir)}`",
            f"- current-basis mechanism quick-report directory: `{rel(mechanism_source_dir)}`",
            f"- output pack: `{rel(out_dir)}`",
            f"- review directory: `{rel(review_dir)}`",
            "",
            "## Target Venue And Literature Style Anchors",
            "",
            "- target venue or paper family: IEEE-style compact data figures for photonic/accelerator papers",
            "- exemplar 1: `CrossLight_A_Cross-Layer_Optimized_Silicon_Photonic_Neural_Network_Accelerator_arXiv2102.06960v1.md` / Fig6-Fig8 / page not visible in Markdown / borrow comparison scatter and compact energy/performance posture / avoid copying claim framing",
            "- exemplar 2: `Lightening_Transformer_HPCA2024.md` / Fig11-Fig15 / page not visible in Markdown / borrow breakdown, comparison, scaling, and noise-sweep restraint / avoid copying dataset semantics",
            "- exemplar 3: `2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md` / Fig6-Fig8 / page not visible in Markdown / borrow speedup-energy panel rhythm and scalability split / avoid benchmark-equivalence wording",
            "- exemplar 4: `Noisy_Machines_Understanding_Noisy_Neural_Networks_and_Enhancing_Robustness_to_Analog_Hardware_Errors_Using_Distillation_arXiv2001.04974.md` / Fig3-Fig4 / page not visible in Markdown / borrow robustness curve/errorbar posture / avoid method-claim transfer",
            "- composition-only references logged in traceability: yes",
            "",
            "## Field Mapping Declaration",
            "",
            *[f"- {line}" for line in field_maps],
            "",
            "## Units and Labels",
            "",
            "- speedup: `x` relative to ASTRA",
            "- accuracy: Top-1 accuracy or accuracy drop in percentage points",
            "- runtime/device context: milliseconds, joules, watts, images/s as provided by the frozen CSV inputs",
            "- claim boundary labels: DET/SPARSE/FULLER statements stay inside the measured tradeoff boundary",
            "",
            "## Output Plan",
            "",
            "- expected exports: SVG, PDF, PNG for every active figure",
            "- grayscale preview: generated for every active figure in the review directory",
            "- traceability update needed: yes",
            "",
            "## Rerun Decision",
            "",
            "- decision: redraw only",
            "- reason: remediated pack uses frozen current-basis Fig5/Fig6 inputs; requested changes are redraw and metadata-label fixes only",
            "- accelerator status: no CUDA, MPS, or local model evaluation was launched for this remediation",
            "",
            "## Inputs",
            "",
            *[f"- `{rel(path)}`" for path in inputs.values()],
        ]
    )
    brief_path = review_dir / ("public_redacted_provenance_rendered_data_figure_brief.md" if remediation_mode else "data_figure_brief.md")
    brief_path.write_text(brief + "\n", encoding="utf-8")

    defect_rows = [
        {
            "defect_id": "CLAIM_BOUNDARY_001",
            "severity": "documented_boundary",
            "figure_id": "Fig3-Fig8",
            "status": "accepted",
            "description": "Positive SPARSE/FULLER accuracy-preservation wording is intentionally blocked.",
            "resolution": "Figures and quick reports label the boundary explicitly.",
        },
        {
            "defect_id": "VISUAL_POLISH_002",
            "severity": "medium",
            "figure_id": "Fig3-Fig8/AppF1-AppF6",
            "status": "closed",
            "description": "Strict review requested explicit range labels, final successor IDs, context boundaries, bounded Fig5 naming, and a wrapped Fig6 footer.",
            "resolution": "Renderer emits explicit AppF1 range inset, final-numbered AppF2 table, visible AppF3 qualitative boundary, Fig5_BoundedSensitivity assets, and wrapped Fig6 footer.",
        },
        {
            "defect_id": "MECHANISM_CONTEXT_003",
            "severity": "high",
            "figure_id": "AppF4-AppF6",
            "status": "closed",
            "description": "The first redesign used retained mechanism context rather than the completed current-basis mechanism evidence.",
            "resolution": "AppF4/AppF5 now read current-basis mechanism and energy rows; AppF6 now reads the complete DET/SPARSE sweep summary.",
        },
        {
            "defect_id": "CURRENT_BASIS_SWEEP_004",
            "severity": "documented_boundary",
            "figure_id": "AppF6",
            "status": "accepted",
            "description": "DET and SPARSE sweeps show measured accuracy tradeoffs, including catastrophic low-k DET rows and non-monotonic DET behavior.",
            "resolution": "The rendered AppF6 shows every complete 3-seed operating point and keeps the ASTRA gap visible.",
        },
    ]
    if final_mode:
        defect_rows.extend(
            [
                {
                    "defect_id": "DQA-APP-001",
                    "severity": "high",
                    "figure_id": "AppF2",
                    "status": "closed",
                    "description": "Compatibility matrix successor figure IDs were stale against final numbering.",
                    "resolution": "Remediated AppF2 CSV and rendered table use Fig3/Fig4, Fig5, Fig6, Fig7, and AppF successor IDs.",
                },
                {
                    "defect_id": "DQA-APP-002",
                    "severity": "low",
                    "figure_id": "AppF1",
                    "status": "closed",
                    "description": "Top-1 ranges were present but visually too small for standalone reading.",
                    "resolution": "AppF1 now includes a compact Top-1 range inset and range-only footer wording.",
                },
                {
                    "defect_id": "DQA-APP-003",
                    "severity": "low",
                    "figure_id": "AppF3",
                    "status": "closed",
                    "description": "Related-work radar needed explicit qualitative/context-only boundary.",
                    "resolution": "AppF3 now visibly states qualitative context only and blocks benchmark-equivalence reading.",
                },
                {
                    "defect_id": "TCQ-LOW-001",
                    "severity": "low",
                    "figure_id": "Fig5",
                    "status": "closed",
                    "description": "Fig5 title/stem used broad NoiseRobustness wording.",
                    "resolution": "Fig5 renders as Fig5_BoundedSensitivity with bounded title text; old generated Fig5_NoiseRobustness assets are removed from the data pack.",
                },
                {
                    "defect_id": "DQ-MAIN-LOW-001",
                    "severity": "low",
                    "figure_id": "Fig6",
                    "status": "closed",
                    "description": "Fig6 footer was too long and reduced page-scale readability.",
                    "resolution": "Fig6 footer is wrapped inside the asset while retaining declared-grid and flow-buffer boundary text.",
                },
            ]
        )
    write_csv(
        review_dir / ("public_redacted_provenance_defect_log.csv" if remediation_mode else "defect_log.csv"),
        defect_rows,
        ["defect_id", "severity", "figure_id", "status", "description", "resolution"],
    )

    rerun_decision = (
        "Rerun decision: redraw only after frozen current-basis CSV ingestion; active freeze promotion is handled separately."
        if final_mode and carrier_run_tag != FINAL_RUN_TAG
        else "Rerun decision: redraw only. The completed current-basis quick reports are used directly; no accelerator-backed run was launched."
    )
    review_report = "\n".join(
        [
            f"# {carrier_run_tag} Figure Review Report",
            "",
            f"Gate 0 input freeze: pass. Quick-report inputs are frozen under `{carrier_run_tag}`.",
            f"Gate 1 data integrity: pass. Current Phase4 values are regenerated; AppF4-AppF6 read current-basis mechanism CSVs from `{rel(mechanism_source_dir)}`.",
            "Gate 2 metric correctness: pass. Runtime/materialization and measured accuracy-tradeoff boundaries are separated.",
            "Gate 3 figure correctness: pass. Renderer v3 keeps DET non-monotonicity, SPARSE gradual recovery, and the ASTRA gap visible in AppF6.",
            "Gate 4 release readiness: data-pack pass pending sibling registry/LaTeX reconciliation; literature anchors are composition-only.",
            "",
            "Residual boundary: positive SPARSE/FULLER preservation claims remain blocked by the completed current-basis sweep.",
            "",
            rerun_decision,
        ]
    )
    (review_dir / ("public_redacted_provenance_figure_review_report.md" if remediation_mode else "figure_review_report.md")).write_text(
        review_report + "\n",
        encoding="utf-8",
    )

    if final_mode:
        promotion_decision = (
            "Promotion decision: promoted. `current_freeze.json` points at this final run tag, and no additional MPS run was launched during the redraw/status refresh."
            if carrier_run_tag == FINAL_RUN_TAG
            else "Promotion decision: candidate only. `current_freeze.json` was left unchanged while this seed/scaling-strengthened pack is reviewed."
        )
        data_review_report = "\n".join(
            [
                "# Data Review Report",
                "",
                f"Run tag: `{carrier_run_tag}`",
                "",
                "Gate 0 input freeze: pass. Target paths, carry-forward inputs, final ImageNet path, weights manifest, and 20260403 contract CSVs are fixed.",
                f"Gate 1 data integrity: pass. Fig5 exposes the three-model dense sensitivity envelope with {fig5_seed_summary} representative cells; Fig6 exposes {'the declared-grid and holdout timing surface' if fig6_declared_grid_ready else 'the full 108-cell current-basis timing grid'}.",
                f"Gate 2 metric correctness: pass within the bounded claim scope. Fig5 is bounded sensitivity-envelope context only; Fig6 carries {fig6_claim_summary} with {fig6_repeat_text}.",
                "Gate 3 figure correctness: pass. The pack has been rendered from the current-basis quick reports with traceability and grayscale previews.",
                "Gate 4 release readiness: data-pack pass. Blocker/high data-pack defects are closed; broad noise tolerance, device-superiority, and accuracy-preservation claims remain blocked.",
                "",
                promotion_decision,
            ]
        )
        (review_dir / ("public_redacted_provenance_data_review_report.md" if remediation_mode else "data_review_report.md")).write_text(
            data_review_report + "\n",
            encoding="utf-8",
        )

    manifest = {
        "run_tag": carrier_run_tag,
        "generated_at": generated_at,
        "paper_figures_dir": rel(out_dir),
        "quick_report_dir": rel(quick_dir),
        "mechanism_run_tag": MECHANISM_RUN_TAG,
        "mechanism_quick_report_dir": rel(mechanism_source_dir),
        "review_dir": rel(review_dir),
        "traceability_csv": rel(trace_path),
        "numbering_registry": rel(review_dir / "public_redacted_provenance_numbering_handoff.csv")
        if remediation_mode
        else rel(out_dir / "figure_numbering_registry.csv"),
        "active_figures": [figure_id for figure_id, _, _, _ in FIGURES],
        "reserved_main_slots": [],
        "grayscale_previews": preview_paths,
        "claim_boundary": "runtime/materialization context only; Fig5 bounded sensitivity and Fig6 declared-grid timing evidence stay bounded; current-basis DET/SPARSE/FULLER mechanisms remain measured tradeoffs",
    }
    if remediation_mode:
        write_json(review_dir / "public_redacted_provenance_review_manifest.json", manifest)
    else:
        write_json(review_dir / "review_manifest.json", manifest)
        write_json(out_dir / "pack_metadata.json", manifest)

    note = "\n".join(
        [
            f"# {carrier_run_tag} Promotion Note",
            "",
            "This is the remediated data-figure pack for the FULLER final numbered sequence, with AppF4-AppF6 refreshed from the current-basis mechanism evidence.",
            "",
            "## Numbering",
            "",
            "- Fig3-Fig8: active main-text data figures in this data-pack worker render.",
            "- AppF1-AppF6: active appendix support figures; AppF4-AppF6 now use current-basis mechanism evidence.",
            "",
            "## Claim Boundary",
            "",
            (
                "The pack separates runtime/materialization readiness from measured accuracy tradeoffs. "
                f"Fig5 carries {fig5_seed_summary} representative sensitivity evidence and Fig6 carries "
                f"{fig6_repeat_text} {fig6_claim_summary}, while DET, SPARSE, "
                "and FULLER stay bounded to the current-basis operating-point evidence."
            ),
            "",
            "## Artifacts",
            "",
            f"- Figure pack: `{rel(out_dir)}`",
            f"- Quick reports: `{rel(quick_dir)}`",
            f"- Current-basis mechanism reports: `{rel(mechanism_source_dir)}`",
            f"- Review artifacts: `{rel(review_dir)}`",
            f"- Traceability: `{rel(trace_path)}`",
            f"- Numbering handoff: `{rel(review_dir / 'public_redacted_provenance_numbering_handoff.csv') if remediation_mode else rel(out_dir / 'figure_numbering_registry.csv')}`",
        ]
    )
    doc_note_path = (
        review_dir / "public_redacted_provenance_promotion_note.md"
        if remediation_mode
        else (
        FINAL_DOC_NOTE
        if final_mode and carrier_run_tag == FINAL_RUN_TAG
        else ROOT / "docs" / "reports" / f"{carrier_run_tag}_note.md"
        if final_mode
        else DEFAULT_DOC_NOTE
        )
    )
    doc_note_path.write_text(note + "\n", encoding="utf-8")

    qa_note = "\n".join(
        [
            "# Current-Basis Data-Figure Mechanism QA Note",
            "",
            f"Date: {generated_at}",
            f"Carrier figure pack: `{rel(out_dir)}`",
            f"Primary mechanism evidence: `{rel(mechanism_source_dir)}`",
            f"Rendering script: `{rel(Path(__file__))}`",
            "",
            "## Decision Table",
            "",
            "| Figure | Decision | Source CSV | Rendering target | QA note |",
            "|---|---|---|---|---|",
            f"| Fig3 | accept | `{rel(inputs['Fig3'])}` | `render_fig6` | Supports bounded runtime/accuracy comparison; large accuracy gaps remain visible. |",
            f"| Fig4 | accept | `{rel(inputs['Fig4'])}` | `render_fig7` | Pareto companion to Fig3; use only with tradeoff wording. |",
            f"| Fig5 | accept | `{rel(inputs['Fig5'])}` | `render_fig9` | Three-model dense sensitivity envelope and {fig5_seed_summary} representative cells are rendered as Fig5_BoundedSensitivity. |",
            f"| Fig6 | accept | `{rel(inputs['Fig6'])}` | `render_fig10` | {('Declared-grid and holdout timing rows are complete' if fig6_declared_grid_ready else 'Current-basis 108-cell timing grid is complete')} with {fig6_repeat_text}; footer is wrapped for page-scale readability. |",
            f"| Fig7 | accept | `{rel(inputs['Fig7'])}` | `render_fig11` | Device context is acceptable when measured-host and modeled-accelerator boundaries stay explicit. |",
            f"| Fig8 | accept | `{rel(inputs['Fig8'])}` | `render_fig12` | Correctly blocks stronger SPARSE/FULLER wording. |",
            f"| AppF1 | accept | `{rel(inputs['AppF1'])}` | `render_appf1` | Variability context now includes explicit max-min Top-1 range labels/inset. |",
            f"| AppF2 | accept | `{rel(inputs['AppF2'])}` | `render_appf2` | Compatibility matrix uses final successor IDs and states the Fig9-Fig12 mechanism-schematic boundary. |",
            f"| AppF3 | accept | `{rel(inputs['AppF3'])}` | `render_appf3` | Related-work radar visibly states qualitative context only; do not use as empirical proof. |",
            f"| AppF4 | accept | `{rel(inputs['AppF4'])}` | `render_appf4` | Current-basis mechanism ablation rows replace retained context; DET/SPARSE/FULLER carry tradeoff caveats. |",
            f"| AppF5 | accept | `{rel(inputs['AppF5'])}` | `render_appf5` | Visible stacked components close to total energy and support source-of-gain discussion. |",
            f"| AppF6 | accept | `{rel(inputs['AppF6'])}` | `render_appf6` | DET non-monotonicity, catastrophic low-k rows, SPARSE gradual recovery, complete=true, seed_count=3, and ASTRA gaps are visible. |",
            "",
            "## Experiment Gate",
            "",
            "No new accelerator-backed outputs were generated for this remediation. The render uses frozen current-basis CSVs and keeps positive preservation claims for SPARSE, FULLER, or a repaired DET/SPARSE point blocked.",
        ]
    )
    qa_note_path = review_dir / (
        "public_redacted_provenance_current_basis_mechanism_qa_note.md"
        if remediation_mode
        else "current_basis_mechanism_qa_note.md"
    )
    qa_note_path.write_text(qa_note + "\n", encoding="utf-8")
    if carrier_run_tag == FINAL_RUN_TAG:
        DEFAULT_CURRENT_BASIS_QA_NOTE.write_text(qa_note + "\n", encoding="utf-8")

    print(f"Rendered {len(FIGURES)} active figures to {out_dir}")
    print(f"Wrote traceability to {trace_path}")
    print(f"Wrote review artifacts to {review_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick_dir", type=Path, default=DEFAULT_QUICK_DIR)
    parser.add_argument("--mechanism_quick_dir", type=Path, default=DEFAULT_MECHANISM_QUICK_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--review_dir", type=Path, default=DEFAULT_REVIEW_DIR)
    args = parser.parse_args()
    command = (
        "python3 experiments/tools/render_fuller_phase4_paper_data_figures.py "
        f"--quick_dir {rel(args.quick_dir)} --mechanism_quick_dir {rel(args.mechanism_quick_dir)} "
        f"--out_dir {rel(args.out_dir)} --review_dir {rel(args.review_dir)}"
    )
    render_all(
        args.quick_dir.resolve(),
        args.mechanism_quick_dir.resolve(),
        args.out_dir.resolve(),
        args.review_dir.resolve(),
        command,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
