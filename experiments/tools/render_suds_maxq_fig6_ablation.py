"""Render the 2026-05-11 SUDS MAX-Q Fig. 6 ablation panel.

The figure intentionally separates measured accuracy deltas from modeled ADC
energy ratios. Accuracy comes from full MobileViT-S ImageNet validation runs on
MPS; energy is computed from the mapped tier model and is not a hardware
measurement.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "experiments/results/report_data/suds_ablation_matrix_20260511_maxq.csv"
DEFAULT_OUTPUT_DIR = ROOT / "figures/paper_figures_20260511_suds_maxq"
DEFAULT_STEM = "Fig6_FullImageNetAblation"

ORDER = ["e2_l1", "e3_slack", "e4_suds", "e5_random", "e6_signal", "e7_overlay"]
LABELS = {
    "e2_l1": "E2\nL1",
    "e3_slack": "E3\nslack",
    "e4_suds": "E4\nSUDS",
    "e5_random": "E5\nrand.",
    "e6_signal": "E6\nsignal",
    "e7_overlay": "E7\nSUDS+L1",
}
COLORS = {
    "e2_l1": "#9C3D54",
    "e3_slack": "#3B73A8",
    "e4_suds": "#2E8B57",
    "e5_random": "#8A8F98",
    "e6_signal": "#C17D10",
    "e7_overlay": "#6E4A9E",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8.0,
        "axes.titlesize": 8.8,
        "axes.labelsize": 8.0,
        "legend.fontsize": 7.4,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 7.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linewidth": 0.5,
    }
)


def read_aggregate_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["matrix"] != "main_ablation" or row["row_type"] != "aggregate":
                continue
            condition = row["condition"]
            if condition in ORDER:
                rows[condition] = row
    missing = [condition for condition in ORDER if condition not in rows]
    if missing:
        raise ValueError(f"Missing aggregate condition rows: {missing}")
    return rows


def f(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def render(input_csv: Path, output_dir: Path, stem: str) -> list[Path]:
    rows = read_aggregate_rows(input_csv)
    x = np.arange(len(ORDER))
    width = 0.64
    colors = [COLORS[key] for key in ORDER]

    delta_top1 = np.array([f(rows[key], "delta_top1") for key in ORDER])
    delta_low = np.array([f(rows[key], "delta_top1_ci95_low") for key in ORDER])
    delta_high = np.array([f(rows[key], "delta_top1_ci95_high") for key in ORDER])
    delta_err = np.vstack([delta_top1 - delta_low, delta_high - delta_top1])
    adc_ratio = np.array([f(rows[key], "adc_energy_ratio_vs_e0") for key in ORDER])
    prune_ratio = np.array([f(rows[key], "mapped_prune_ratio") for key in ORDER])

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.72), constrained_layout=True)

    ax = axes[0]
    bars = ax.bar(x, delta_top1, width=width, color=colors, edgecolor="#2F2F2F", linewidth=0.55)
    ax.errorbar(x, delta_top1, yerr=delta_err, fmt="none", ecolor="#1E1E1E", elinewidth=0.8, capsize=2.4)
    ax.axhline(0, color="#2A2A2A", linewidth=0.8)
    ax.set_title("Measured accuracy delta")
    ax.set_ylabel("Top-1 delta vs dense (pp)")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[key] for key in ORDER])
    ax.set_ylim(-6.2, 0.35)
    for bar, value in zip(bars, delta_top1):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.28,
            f"{value:.2f}",
            ha="center",
            va="top",
            fontsize=6.8,
            color="white" if value < -2.2 else "#222222",
        )

    ax = axes[1]
    bars = ax.bar(x, adc_ratio, width=width, color=colors, edgecolor="#2F2F2F", linewidth=0.55)
    ax.axhline(1.0, color="#2A2A2A", linewidth=0.8, linestyle="--")
    ax.set_title("Modeled ADC ratio")
    ax.set_ylabel("ADC energy ratio vs dense")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[key] for key in ORDER])
    ax.set_ylim(0, 1.08)
    for bar, value, prune in zip(bars, adc_ratio, prune_ratio):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.035,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#222222",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(value - 0.08, 0.04),
            f"P {prune:.2f}",
            ha="center",
            va="top",
            fontsize=6.6,
            color="white" if value > 0.28 else "#222222",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for ext in ("pdf", "svg", "png"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path)
        outputs.append(path)
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = render(args.input_csv, args.output_dir, args.stem)
    for path in outputs:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
