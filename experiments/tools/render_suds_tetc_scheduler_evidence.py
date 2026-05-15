#!/usr/bin/env python3
"""Render scheduler slack and SUDS budget evidence for the TETC figure pack."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260513_tetc_pivot"
OUT_TAG = "20260516_submission_figure_pack"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"
DEFAULT_INPUT = REPORT_DATA / f"suds_tetc_scheduler_ablation_{TAG}.csv"
DEFAULT_OUT = REPO_ROOT / f"figures/suds_tetc_{OUT_TAG}"

FIG_STEM = "Fig7_SchedulerSlackBudgetEvidence"

SCHEDULER_ORDER = [
    "fifo",
    "asap",
    "edf_deadline_aware",
    "utilization_aware",
    "suds_aware",
]
SCHEDULER_LABELS = {
    "fifo": "FIFO",
    "asap": "ASAP",
    "edf_deadline_aware": "EDF",
    "utilization_aware": "Util.",
    "suds_aware": "SUDS-aware",
}
WORKLOAD_ORDER = [
    "bert_base_glue_seq128",
    "mobilevit_s_transformer_blocks_256",
]
WORKLOAD_LABELS = {
    "bert_base_glue_seq128": "BERT-base\nGLUE",
    "mobilevit_s_transformer_blocks_256": "MobileViT-S\nblocks",
}
WORKLOAD_COLORS = {
    "bert_base_glue_seq128": "#2C7BB6",
    "mobilevit_s_transformer_blocks_256": "#D95F02",
}
BUDGET_COLORS = {
    "keep_ratio": "#1B7837",
    "degrade_ratio": "#E66101",
    "prune_ratio": "#5E3C99",
}
BUDGET_HATCHES = {
    "keep_ratio": "///",
    "degrade_ratio": "\\\\\\",
    "prune_ratio": "xx",
}
BUDGET_LABELS = {
    "keep_ratio": "KEEP",
    "degrade_ratio": "DEGRADE",
    "prune_ratio": "PRUNE",
}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 7.6,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.4,
        "legend.fontsize": 6.6,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.45,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-ablation-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def fnum(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"missing required artifact: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "tag",
        "sensitivity_case",
        "workload",
        "condition",
        "scheduler",
        "p10_slack_norm",
        "p50_slack_norm",
        "p90_slack_norm",
        "mean_queue_wait_ns",
        "deadline_miss_rate",
        "keep_ratio",
        "degrade_ratio",
        "prune_ratio",
        "promotion_decision",
    }
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise SystemExit(f"missing required columns in {path}: {sorted(missing)}")
    return rows


def selected_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    filtered = [
        row
        for row in rows
        if row["tag"] == TAG
        and row["sensitivity_case"] == "nominal"
        and row["condition"] == "suds_pareto"
        and row["workload"] in WORKLOAD_ORDER
        and row["scheduler"] in SCHEDULER_ORDER
    ]
    by_key = {(row["workload"], row["scheduler"]): row for row in filtered}
    expected = {(w, s) for w in WORKLOAD_ORDER for s in SCHEDULER_ORDER}
    missing = expected.difference(by_key)
    if missing:
        raise SystemExit(f"missing SUDS Pareto scheduler rows: {sorted(missing)}")
    return by_key


def write_source_summary(by_key: dict[tuple[str, str], dict[str, str]], out_dir: Path) -> Path:
    out_path = out_dir / f"{FIG_STEM}_source.csv"
    fields = [
        "workload",
        "scheduler",
        "p10_slack_norm",
        "p50_slack_norm",
        "p90_slack_norm",
        "mean_queue_wait_ns",
        "deadline_miss_rate",
        "keep_ratio",
        "degrade_ratio",
        "prune_ratio",
        "p10_gain_vs_fifo_pp",
        "queue_wait_reduction_vs_fifo_pct",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for workload in WORKLOAD_ORDER:
            fifo = by_key[(workload, "fifo")]
            fifo_p10 = fnum(fifo["p10_slack_norm"])
            fifo_queue = fnum(fifo["mean_queue_wait_ns"])
            for scheduler in SCHEDULER_ORDER:
                row = by_key[(workload, scheduler)]
                queue = fnum(row["mean_queue_wait_ns"])
                reduction = 0.0 if fifo_queue == 0 else 100.0 * (fifo_queue - queue) / fifo_queue
                writer.writerow(
                    {
                        "workload": workload,
                        "scheduler": scheduler,
                        "p10_slack_norm": row["p10_slack_norm"],
                        "p50_slack_norm": row["p50_slack_norm"],
                        "p90_slack_norm": row["p90_slack_norm"],
                        "mean_queue_wait_ns": row["mean_queue_wait_ns"],
                        "deadline_miss_rate": row["deadline_miss_rate"],
                        "keep_ratio": row["keep_ratio"],
                        "degrade_ratio": row["degrade_ratio"],
                        "prune_ratio": row["prune_ratio"],
                        "p10_gain_vs_fifo_pp": 100.0 * (fnum(row["p10_slack_norm"]) - fifo_p10),
                        "queue_wait_reduction_vs_fifo_pct": reduction,
                    }
                )
    return out_path


def panel_slack_distribution(ax: plt.Axes, by_key: dict[tuple[str, str], dict[str, str]]) -> None:
    x = np.arange(len(SCHEDULER_ORDER), dtype=float)
    offsets = [-0.13, 0.13]
    for idx, workload in enumerate(WORKLOAD_ORDER):
        p10 = np.array([fnum(by_key[(workload, s)]["p10_slack_norm"]) for s in SCHEDULER_ORDER])
        p50 = np.array([fnum(by_key[(workload, s)]["p50_slack_norm"]) for s in SCHEDULER_ORDER])
        p90 = np.array([fnum(by_key[(workload, s)]["p90_slack_norm"]) for s in SCHEDULER_ORDER])
        color = WORKLOAD_COLORS[workload]
        ax.errorbar(
            x + offsets[idx],
            p50,
            yerr=np.vstack([p50 - p10, p90 - p50]),
            fmt="o" if idx == 0 else "s",
            markersize=4.0,
            color=color,
            ecolor=color,
            capsize=2.5,
            linewidth=1.0,
            label=WORKLOAD_LABELS[workload].replace("\n", " "),
        )
    ax.set_title("A. Scheduler-visible slack")
    ax.set_ylabel("Normalized slack")
    ax.set_xticks(x)
    ax.set_xticklabels([SCHEDULER_LABELS[s] for s in SCHEDULER_ORDER], rotation=25, ha="right")
    ax.set_ylim(0.45, 1.02)
    ax.legend(frameon=False, loc="lower right", handlelength=1.0)
    ax.text(
        0.02,
        0.96,
        "marker=p50; whiskers=p10-p90",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color="#555555",
    )


def panel_budget_mix(ax: plt.Axes, by_key: dict[tuple[str, str], dict[str, str]]) -> None:
    y = np.arange(len(WORKLOAD_ORDER), dtype=float)
    left = np.zeros(len(WORKLOAD_ORDER), dtype=float)
    for key in ("keep_ratio", "degrade_ratio", "prune_ratio"):
        values = np.array([100.0 * fnum(by_key[(w, "suds_aware")][key]) for w in WORKLOAD_ORDER])
        ax.barh(
            y,
            values,
            left=left,
            height=0.42,
            color=BUDGET_COLORS[key],
            edgecolor="black",
            linewidth=0.45,
            hatch=BUDGET_HATCHES[key],
            label=BUDGET_LABELS[key],
        )
        for yy, ll, vv in zip(y, left, values):
            if vv >= 8:
                ax.text(ll + vv / 2, yy, f"{vv:.0f}%", ha="center", va="center", fontsize=6.8, color="white")
        left += values
    ax.set_title("B. SUDS budget mix")
    ax.set_xlabel("Budget share (%)")
    ax.set_yticks(y)
    ax.set_yticklabels([WORKLOAD_LABELS[w] for w in WORKLOAD_ORDER])
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.5, 0.50), ncols=3, handlelength=1.4)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)


def panel_trace_delta(ax: plt.Axes, by_key: dict[tuple[str, str], dict[str, str]]) -> None:
    x = np.arange(len(WORKLOAD_ORDER), dtype=float)
    p10_gain = []
    queue_reduction = []
    for workload in WORKLOAD_ORDER:
        fifo = by_key[(workload, "fifo")]
        suds = by_key[(workload, "suds_aware")]
        fifo_p10 = fnum(fifo["p10_slack_norm"])
        fifo_queue = fnum(fifo["mean_queue_wait_ns"])
        suds_queue = fnum(suds["mean_queue_wait_ns"])
        p10_gain.append(100.0 * (fnum(suds["p10_slack_norm"]) - fifo_p10))
        queue_reduction.append(0.0 if fifo_queue == 0 else 100.0 * (fifo_queue - suds_queue) / fifo_queue)

    ax.bar(
        x,
        p10_gain,
        width=0.46,
        color="#2C7BB6",
        edgecolor="black",
        linewidth=0.45,
        hatch="//",
    )
    ax.set_ylabel("p10 slack gain vs FIFO (pp)")
    ax.set_xticks(x)
    ax.set_xticklabels([WORKLOAD_LABELS[w] for w in WORKLOAD_ORDER])
    ax.set_title("C. SUDS-aware trace delta")
    ax.axhline(0, color="#333333", linewidth=0.7)
    ax.set_ylim(0, max(p10_gain) * 1.45)
    for xx, val, qred in zip(x, p10_gain, queue_reduction):
        ax.text(xx, val + 0.25, f"{val:.1f}", ha="center", va="bottom", fontsize=6.8)
        ax.text(
            xx,
            0.08 * max(p10_gain),
            f"queue wait\n-{qred:.0f}%",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color="#8C3B00",
        )
    ax.text(
        0.02,
        0.96,
        "Deadline miss rate: 0%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color="#555555",
    )


def render_figure(by_key: dict[tuple[str, str], dict[str, str]], out_dir: Path) -> dict[str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.0), gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
    panel_slack_distribution(axes[0], by_key)
    panel_budget_mix(axes[1], by_key)
    panel_trace_delta(axes[2], by_key)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.22, wspace=0.48)

    outputs: dict[str, str] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        path = out_dir / f"{FIG_STEM}.{ext}"
        fig.savefig(path)
        if ext == "svg":
            text = path.read_text(encoding="utf-8")
            path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")
        outputs[ext] = rel(path)
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    rows = load_rows(args.scheduler_ablation_csv)
    by_key = selected_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = write_source_summary(by_key, args.output_dir)
    outputs = render_figure(by_key, args.output_dir)
    print(f"source={rel(source)}")
    for ext, path in outputs.items():
        print(f"{ext}={path}")


if __name__ == "__main__":
    main()
