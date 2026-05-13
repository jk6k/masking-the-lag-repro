#!/usr/bin/env python3
"""Render TETC pivot figures from governed CSV/JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260513_tetc_pivot"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"
OUT_DIR = REPO_ROOT / f"figures/suds_tetc_{TAG}"

SUMMARY_CSV = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}_summary.csv"
ARCH_JSON = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}.json"
CONSERVATIVE_JSON = REPORT_DATA / f"suds_tetc_conservative_pareto_{TAG}.json"
SCIENCE_JSON = REPORT_DATA / f"suds_tetc_science_gate_{TAG}.json"
BRIEF = REPO_ROOT / "docs/reports/task_briefs/20260513_suds_tetc_data_figures_brief.md"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8.5,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
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
    "lightening_dptc": "#4D4D4D",
    "l1": "#2C7BB6",
    "slack_only": "#7AA6C2",
    "signal_only": "#1B9E77",
    "suds_pareto": "#D95F02",
    "suds_signal": "#7570B3",
    "suds_l1": "#B07AA1",
    "hyatten_style": "#A6761D",
    "adc": "#6BAED6",
    "dac_mzm": "#9ECAE1",
    "detector_tia": "#C6DBEF",
    "laser": "#FDBF6F",
    "memory": "#74C476",
    "optical_link": "#FD8D3C",
    "control": "#9E9AC8",
    "digital": "#BDBDBD",
}

MARKERS = {
    "lightening_dptc": "s",
    "l1": "^",
    "slack_only": "v",
    "signal_only": "D",
    "suds_pareto": "*",
    "suds_signal": "P",
    "suds_l1": "X",
    "hyatten_style": "o",
}

LABELS = {
    "lightening_dptc": "Lightening DPTC",
    "l1": "L1",
    "slack_only": "Slack-only",
    "signal_only": "Signal-only",
    "suds_pareto": "SUDS Pareto",
    "suds_signal": "SUDS+signal",
    "suds_l1": "SUDS+L1",
    "hyatten_style": "HyAtten boundary",
}


@dataclass(frozen=True)
class FigureMeta:
    figure_id: str
    stem: str
    title: str
    source: str
    figure_type: str
    caption_scope_note: str


FIGURES = [
    FigureMeta(
        "Fig1",
        "Fig1_TETCArchitectureEvidenceFlow",
        "Architecture and evidence flow",
        "architecture_json;science_gate_json",
        "code_native_schematic",
        "Architecture schematic; modeled PPA and measured MPS accuracy linkage only.",
    ),
    FigureMeta(
        "Fig2",
        "Fig2_AccuracyEDPPareto",
        "Accuracy/EDP Pareto surface",
        "architecture_summary_csv",
        "data_figure",
        "Nominal architecture rows; old aggressive SUDS rows and stronger baselines remain visible.",
    ),
    FigureMeta(
        "Fig3",
        "Fig3_EnergyBreakdown",
        "Energy breakdown",
        "architecture_summary_csv",
        "data_figure",
        "Modeled energy components normalized to each workload's Lightening reference.",
    ),
    FigureMeta(
        "Fig4",
        "Fig4_ConservativeParetoAccuracy",
        "Conservative Pareto accuracy evidence",
        "conservative_pareto_json",
        "data_figure",
        "Measured MPS MobileViT-S per-seed Top-1 deltas; no CPU fallback.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--architecture-json", type=Path, default=ARCH_JSON)
    parser.add_argument("--conservative-json", type=Path, default=CONSERVATIVE_JSON)
    parser.add_argument("--science-json", type=Path, default=SCIENCE_JSON)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"missing required artifact: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def save_multi(fig: plt.Figure, out_dir: Path, stem: str) -> dict[str, str]:
    outputs: dict[str, str] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path)
        if ext == "svg":
            normalize_svg(path)
        outputs[ext] = rel(path)
    plt.close(fig)
    return outputs


def draw_box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str, width: float = 0.22) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        0.13,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(box)
    ax.text(x + width / 2.0, y + 0.065, text, ha="center", va="center", fontsize=8.0)


def draw_arrow(ax: plt.Axes, a: tuple[float, float], b: tuple[float, float], color: str = "#555555") -> None:
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=color,
            shrinkA=5,
            shrinkB=5,
        )
    )


def render_fig1(args: argparse.Namespace) -> dict[str, str]:
    arch = load_json(args.architecture_json)
    science = load_json(args.science_json)
    schedule = science.get("summary", {}).get("workload_grounding", {}).get("schedule_metadata", {})
    fig, ax = plt.subplots(figsize=(7.16, 2.7))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    draw_box(ax, (0.03, 0.64), "MPS accuracy\nBERT GLUE + MobileViT", "#2C7BB6")
    draw_box(ax, (0.29, 0.64), "Transformer\nGEMM extraction", "#4D4D4D")
    draw_box(ax, (0.55, 0.64), "DPTC schedule\nmetadata", "#D95F02")
    draw_box(ax, (0.78, 0.64), "Modeled PPA\nsummary", "#4D4D4D", width=0.19)

    draw_box(ax, (0.16, 0.22), "SUDS budget\nKEEP/DEGRADE/PRUNE", "#D95F02", width=0.24)
    draw_box(ax, (0.45, 0.22), "Local selector\nL1 or signal", "#1B9E77")
    draw_box(ax, (0.72, 0.22), "Calibration boundary\nADC RTL PHY", "#7570B3", width=0.22)

    draw_arrow(ax, (0.25, 0.705), (0.29, 0.705))
    draw_arrow(ax, (0.51, 0.705), (0.55, 0.705))
    draw_arrow(ax, (0.77, 0.705), (0.79, 0.705))
    draw_arrow(ax, (0.66, 0.64), (0.31, 0.35), "#D95F02")
    draw_arrow(ax, (0.40, 0.285), (0.45, 0.285), "#1B9E77")
    draw_arrow(ax, (0.67, 0.285), (0.72, 0.285), "#7570B3")
    draw_arrow(ax, (0.83, 0.35), (0.86, 0.64), "#7570B3")

    ax.text(
        0.03,
        0.04,
        (
            f"BERT schedule rows: {schedule.get('bert_kernel_rows', 'n/a')}; "
            f"GLUE schedule links: {schedule.get('glue_link_rows', 'n/a')}; "
            f"architecture rows: {len(arch.get('summary_rows', []))}. "
            "Evidence boundary: measured MPS accuracy plus architecture-modeled PPA."
        ),
        ha="left",
        va="bottom",
        fontsize=7.5,
    )
    return save_multi(fig, args.output_dir, "Fig1_TETCArchitectureEvidenceFlow")


def nominal_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("sensitivity_case") == "nominal"]


def render_fig2(args: argparse.Namespace, rows: list[dict[str, str]]) -> dict[str, str]:
    selected = [
        "lightening_dptc",
        "l1",
        "slack_only",
        "signal_only",
        "suds_pareto",
        "suds_signal",
        "suds_l1",
        "hyatten_style",
    ]
    workloads = [
        ("bert_base_glue_seq128", "BERT-base GLUE"),
        ("mobilevit_s_transformer_blocks_256", "MobileViT-S blocks"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9), sharey=True)
    for ax, (workload, title) in zip(axes, workloads):
        wr = [row for row in nominal_rows(rows) if row.get("workload") == workload and row.get("condition") in selected]
        for condition in selected:
            item = next((row for row in wr if row.get("condition") == condition), None)
            if not item:
                continue
            delta = fnum(item.get("delta_accuracy"))
            if math.isnan(delta):
                continue
            edp = fnum(item.get("edp_ratio_vs_lightening"))
            size = 110 if condition == "suds_pareto" else 52
            ax.scatter(
                delta,
                edp,
                s=size,
                marker=MARKERS.get(condition, "o"),
                color=COLORS.get(condition, "#333333"),
                edgecolor="black" if condition == "suds_pareto" else "white",
                linewidth=0.7,
                label=LABELS.get(condition, condition),
                zorder=4 if condition == "suds_pareto" else 3,
            )
            if condition in {"lightening_dptc", "suds_pareto"}:
                offsets = {
                    "lightening_dptc": (-58, -8),
                    "suds_pareto": (-76, 10) if workload.startswith("bert") else (6, -2),
                }
                ax.annotate(
                    LABELS.get(condition, condition).replace(" ", "\n")
                    if condition == "suds_pareto"
                    else LABELS.get(condition, condition),
                    (delta, edp),
                    xytext=offsets[condition],
                    textcoords="offset points",
                    fontsize=6.8,
                )
        ax.axvline(-1.0, color="#999999", linestyle="--", linewidth=0.8)
        ax.axhline(1.0, color="#999999", linestyle=":", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Accuracy delta vs dense (pp)")
        ax.set_xlim(-3.9, 0.25)
        ax.set_ylim(0.60, 1.04)
    axes[0].set_ylabel("EDP ratio vs Lightening")
    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_multi(fig, args.output_dir, "Fig2_AccuracyEDPPareto")


def render_fig3(args: argparse.Namespace, rows: list[dict[str, str]]) -> dict[str, str]:
    conditions = ["lightening_dptc", "l1", "signal_only", "suds_pareto"]
    components = [
        ("adc_energy_pj", "ADC", "adc"),
        ("dac_mzm_energy_pj", "DAC/MZM", "dac_mzm"),
        ("detector_tia_energy_pj", "Detector/TIA", "detector_tia"),
        ("laser_energy_pj", "Laser", "laser"),
        ("memory_energy_pj", "Memory", "memory"),
        ("optical_link_energy_pj", "Optical link", "optical_link"),
        ("control_energy_pj", "Control", "control"),
        ("digital_fallback_energy_pj", "Digital", "digital"),
    ]
    workloads = [
        ("bert_base_glue_seq128", "BERT-base GLUE"),
        ("mobilevit_s_transformer_blocks_256", "MobileViT-S blocks"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1), sharey=True)
    for ax, (workload, title) in zip(axes, workloads):
        wr = [row for row in nominal_rows(rows) if row.get("workload") == workload]
        baseline = next(row for row in wr if row.get("condition") == "lightening_dptc")
        baseline_energy = fnum(baseline.get("energy_pj"), 1.0)
        x = np.arange(len(conditions))
        bottom = np.zeros(len(conditions))
        for field, label, color_key in components:
            vals = []
            for condition in conditions:
                row = next(item for item in wr if item.get("condition") == condition)
                vals.append(fnum(row.get(field), 0.0) / baseline_energy)
            ax.bar(
                x,
                vals,
                bottom=bottom,
                label=label,
                color=COLORS[color_key],
                edgecolor="white",
                linewidth=0.4,
            )
            bottom += np.asarray(vals)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c].replace(" ", "\n") for c in conditions], rotation=0)
        ax.axhline(1.0, color="#555555", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Condition")
    axes[0].set_ylabel("Energy ratio vs Lightening")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.09))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save_multi(fig, args.output_dir, "Fig3_EnergyBreakdown")


def render_fig4(args: argparse.Namespace) -> dict[str, str]:
    payload = load_json(args.conservative_json)
    rows = [
        row for row in payload.get("rows", [])
        if row.get("row_type") == "per_seed"
    ]
    rows = sorted(rows, key=lambda row: int(row.get("seed", 0)))
    seeds = [int(row["seed"]) for row in rows]
    deltas = [fnum(row.get("delta_top1")) for row in rows]
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    ax.plot(seeds, deltas, color=COLORS["suds_pareto"], marker="o", linewidth=1.4, label="Measured seeds")
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.axhline(-1.0, color="#999999", linewidth=0.9, linestyle="--", label="-1 pp target")
    ax.fill_between([min(seeds), max(seeds)], [-1.0, -1.0], [0.0, 0.0], color="#FEE8C8", alpha=0.45, label="TETC target band")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Top-1 delta vs dense (pp)")
    ax.set_ylim(-1.1, 0.15)
    ax.set_xticks(seeds)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Conservative SUDS measured accuracy")
    fig.tight_layout()
    return save_multi(fig, args.output_dir, "Fig4_ConservativeParetoAccuracy")


def write_registry(output_dir: Path) -> None:
    path = output_dir / "figure_numbering_registry.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["figure_id", "manuscript_order", "canonical_stem", "title", "status"],
        )
        writer.writeheader()
        for idx, meta in enumerate(FIGURES, start=1):
            writer.writerow(
                {
                    "figure_id": meta.figure_id,
                    "manuscript_order": idx,
                    "canonical_stem": meta.stem,
                    "title": meta.title,
                    "status": "rendered",
                }
            )


def write_traceability(output_dir: Path, outputs: dict[str, dict[str, str]]) -> None:
    path = output_dir / "figure_traceability.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "figure_id",
                "canonical_stem",
                "figure_type",
                "run_tag",
                "primary_source",
                "brief",
                "pdf",
                "svg",
                "png",
                "caption_scope_note",
            ],
        )
        writer.writeheader()
        for meta in FIGURES:
            row_outputs = outputs[meta.stem]
            writer.writerow(
                {
                    "figure_id": meta.figure_id,
                    "canonical_stem": meta.stem,
                    "figure_type": meta.figure_type,
                    "run_tag": TAG,
                    "primary_source": meta.source,
                    "brief": rel(BRIEF),
                    "pdf": row_outputs.get("pdf", ""),
                    "svg": row_outputs.get("svg", ""),
                    "png": row_outputs.get("png", ""),
                    "caption_scope_note": meta.caption_scope_note,
                }
            )


def main() -> None:
    args = parse_args()
    rows = load_csv(args.summary_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "Fig1_TETCArchitectureEvidenceFlow": render_fig1(args),
        "Fig2_AccuracyEDPPareto": render_fig2(args, rows),
        "Fig3_EnergyBreakdown": render_fig3(args, rows),
        "Fig4_ConservativeParetoAccuracy": render_fig4(args),
    }
    write_registry(args.output_dir)
    write_traceability(args.output_dir, outputs)
    for meta in FIGURES:
        print(f"wrote {rel(args.output_dir / (meta.stem + '.pdf'))}")
        print(f"wrote {rel(args.output_dir / (meta.stem + '.svg'))}")
        print(f"wrote {rel(args.output_dir / (meta.stem + '.png'))}")
    print(f"wrote {rel(args.output_dir / 'figure_numbering_registry.csv')}")
    print(f"wrote {rel(args.output_dir / 'figure_traceability.csv')}")


if __name__ == "__main__":
    main()
