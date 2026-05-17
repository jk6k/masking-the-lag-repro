#!/usr/bin/env python3
"""Render the fuller report figures from governed report-data CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = ROOT / "experiments" / "results" / "report_data"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "results" / "report_figures"
DEFAULT_REVIEW_ROOT = ROOT / "experiments" / "results" / "review"
DEFAULT_RUN_TAG = "20260319_fullerexp_v1"

DEVICE_COLOR_MAP = {
    "CPU": "#4C78A8",
    "GPU (MPS)": "#F58518",
    "ASTRA": "#54A24B",
    "FULLER": "#E45756",
}
KIND_HATCH_MAP = {
    "Real device": "",
    "Accelerator model": "//",
}
STAGE_ORDER = [
    "fetch_map",
    "btos",
    "serialize_drive",
    "oag_compute",
    "pca_adc",
    "electronic_compute",
    "writeback",
    "bubble",
]
STAGE_LABELS = {
    "fetch_map": "Fetch Map",
    "btos": "BtoS",
    "serialize_drive": "Serialize / Drive",
    "oag_compute": "OAG Compute",
    "pca_adc": "PCA / ADC",
    "electronic_compute": "Electronic Compute",
    "writeback": "Writeback",
    "bubble": "Bubble",
}
ENERGY_COMPONENT_ORDER = [
    "conversion_control",
    "memory_move",
    "oe",
    "adc_pca",
    "laser_optical",
    "other_static",
    "hidden_system_cost",
]
ENERGY_COMPONENT_LABELS = {
    "conversion_control": "Conversion + Control",
    "memory_move": "Memory Move",
    "oe": "OE",
    "adc_pca": "ADC + PCA",
    "laser_optical": "Laser Optical",
    "other_static": "Other Static",
    "hidden_system_cost": "Hidden System Cost",
}
ABLATION_VARIANT_ORDER = ["ASTRA", "HOPS", "FLOW_MESO", "FLOW_PHY", "FULLER"]
ABLATION_VARIANT_LABELS = {
    "ASTRA": "ASTRA",
    "HOPS": "HOPS",
    "FLOW_MESO": "HOPS+MESO",
    "FLOW_PHY": "HOPS+PHY",
    "FULLER": "FULLER",
}
ABLATION_COLOR_SEQ = {
    "ASTRA": "#4C78A8",
    "HOPS": "#72B7B2",
    "FLOW_MESO": "#54A24B",
    "FLOW_PHY": "#F58518",
    "FULLER": "#E45756",
}
SUPPORT_PROFILE_ORDER = ["clean", "mild", "medium", "hard"]
SUPPORT_PROFILE_LABELS = {
    "clean": "Clean",
    "mild": "Mild",
    "medium": "Medium",
    "hard": "Hard",
}


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 400,
        }
    )


def _export(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        out_dir / f"{stem}.svg",
        out_dir / f"{stem}.pdf",
        out_dir / f"{stem}.png",
    ]
    for path in outputs:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return outputs


def _device_rows(input_root: Path) -> list[dict[str, Any]]:
    cpu = _load_csv(input_root / "fuller_cpu_device_metrics_20260319_fullerexp_v1.csv")[0]
    gpu = _load_csv(input_root / "fuller_gpu_device_metrics_20260319_fullerexp_v1.csv")[0]
    astra = _load_csv(input_root / "fuller_astra_substrate_model_summary_20260319_fullerexp_v1.csv")[0]
    fuller = _load_csv(input_root / "fuller_slice_model_summary_20260319_fullerexp_v1.csv")[0]
    return [
        {
            "label": "CPU",
            "kind": "Real device",
            "latency_ms": _float(cpu["latency_ms"]),
            "avg_power_w": _float(cpu["avg_power_w"]),
            "energy_j": _float(cpu["energy_j"]),
            "host_name": str(cpu["host_name"]),
            "device_model": str(cpu["device_model"]),
            "model": str(cpu["model"]),
            "batch_size": str(cpu["batch_size"]),
            "sequence_length": str(cpu["sequence_length"]),
        },
        {
            "label": "GPU (MPS)",
            "kind": "Real device",
            "latency_ms": _float(gpu["latency_ms"]),
            "avg_power_w": _float(gpu["avg_power_w"]),
            "energy_j": _float(gpu["energy_j"]),
            "host_name": str(gpu["host_name"]),
            "device_model": str(gpu["device_model"]),
            "model": str(gpu["model"]),
            "batch_size": str(gpu["batch_size"]),
            "sequence_length": str(gpu["sequence_length"]),
        },
        {
            "label": "ASTRA",
            "kind": "Accelerator model",
            "latency_ms": _float(astra["latency_ms"]),
            "avg_power_w": _float(astra["avg_power_w"]),
            "energy_j": _float(astra["energy_j"]),
        },
        {
            "label": "FULLER",
            "kind": "Accelerator model",
            "latency_ms": _float(fuller["latency_ms"]),
            "avg_power_w": _float(fuller["avg_power_w"]),
            "energy_j": _float(fuller["energy_j"]),
        },
    ]


def _render_device_comparison(input_root: Path, out_dir: Path) -> tuple[list[Path], list[Path]]:
    rows = _device_rows(input_root)
    labels = [row["label"] for row in rows]
    kinds = [row["kind"] for row in rows]
    metrics = [
        ("latency_ms", "Latency (ms)"),
        ("avg_power_w", "Power (W)"),
        ("energy_j", "Energy per Inference (J)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.1), constrained_layout=True)
    x = np.arange(len(labels))
    for ax, (field, title) in zip(axes, metrics):
        values = [row[field] for row in rows]
        bars = ax.bar(
            x,
            values,
            color=[DEVICE_COLOR_MAP[label] for label in labels],
            edgecolor="black",
            linewidth=0.8,
        )
        for bar, kind, value in zip(bars, kinds, values):
            bar.set_hatch(KIND_HATCH_MAP[kind])
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.set_ylabel(title)
    legend_handles = [
        Patch(facecolor="white", edgecolor="black", label="Real device"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Accelerator model"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", fontsize=8)
    host_note = (
        f"Host: {rows[0]['host_name']} | CPU: {rows[0]['device_model']} | "
        f"GPU: {rows[1]['device_model']} | Workload: {rows[0]['model']} bs={rows[0]['batch_size']} "
        f"seq={rows[0]['sequence_length']}"
    )
    fig.text(0.5, 0.01, host_note, ha="center", va="bottom", fontsize=7)
    fig.suptitle("FigFuller_DeviceComparison", fontsize=11)

    outputs = _export(fig, out_dir, "FigFuller_DeviceComparison")
    inputs = [
        input_root / "fuller_cpu_device_metrics_20260319_fullerexp_v1.csv",
        input_root / "fuller_gpu_device_metrics_20260319_fullerexp_v1.csv",
        input_root / "fuller_astra_substrate_model_summary_20260319_fullerexp_v1.csv",
        input_root / "fuller_slice_model_summary_20260319_fullerexp_v1.csv",
    ]
    return outputs, inputs


def _render_latency_breakdown(input_root: Path, out_dir: Path) -> tuple[list[Path], list[Path]]:
    rows = _load_csv(input_root / "fuller_stage_latency_breakdown_20260319_fullerexp_v1.csv")
    stages = {str(row["stage"]) for row in rows}
    stage_order = [stage for stage in STAGE_ORDER if stage in stages]
    astra = {row["stage"]: _float(row["latency_ms"]) for row in rows if row["baseline_variant"] == "ASTRA"}
    fuller = {row["stage"]: _float(row["latency_ms"]) for row in rows if row["baseline_variant"] == "FULLER"}

    y = np.arange(len(stage_order))
    h = 0.36
    fig, ax = plt.subplots(figsize=(7.16, 3.2), constrained_layout=True)
    ax.barh(
        y - h / 2,
        [astra.get(stage, 0.0) for stage in stage_order],
        h,
        label="ASTRA",
        color=DEVICE_COLOR_MAP["ASTRA"],
        edgecolor="black",
    )
    ax.barh(
        y + h / 2,
        [fuller.get(stage, 0.0) for stage in stage_order],
        h,
        label="FULLER",
        color=DEVICE_COLOR_MAP["FULLER"],
        edgecolor="black",
        hatch="//",
    )
    ax.set_yticks(y, [STAGE_LABELS[stage] for stage in stage_order])
    ax.invert_yaxis()
    ax.set_xlabel("Latency Contribution (ms)")
    ax.set_title("FigFuller_LatencyBreakdown")
    ax.legend(ncols=2, loc="lower right")

    outputs = _export(fig, out_dir, "FigFuller_LatencyBreakdown")
    return outputs, [input_root / "fuller_stage_latency_breakdown_20260319_fullerexp_v1.csv"]


def _render_energy_breakdown(input_root: Path, out_dir: Path) -> tuple[list[Path], list[Path]]:
    rows = _load_csv(input_root / "fuller_stage_energy_breakdown_20260319_fullerexp_v1.csv")
    components = [component for component in ENERGY_COMPONENT_ORDER if any(row["component"] == component for row in rows)]
    astra = {row["component"]: _float(row["energy_j"]) for row in rows if row["baseline_variant"] == "ASTRA"}
    fuller = {row["component"]: _float(row["energy_j"]) for row in rows if row["baseline_variant"] == "FULLER"}

    y = np.arange(len(components))
    h = 0.36
    fig, ax = plt.subplots(figsize=(7.16, 3.2), constrained_layout=True)
    ax.barh(
        y - h / 2,
        [astra.get(item, 0.0) for item in components],
        h,
        label="ASTRA",
        color=DEVICE_COLOR_MAP["ASTRA"],
        edgecolor="black",
    )
    ax.barh(
        y + h / 2,
        [fuller.get(item, 0.0) for item in components],
        h,
        label="FULLER",
        color=DEVICE_COLOR_MAP["FULLER"],
        edgecolor="black",
        hatch="//",
    )
    ax.set_yticks(y, [ENERGY_COMPONENT_LABELS[item] for item in components])
    ax.invert_yaxis()
    ax.set_xlabel("Energy Contribution (J)")
    ax.set_title("FigFuller_EnergyBreakdown")
    ax.legend(ncols=2, loc="lower right")

    outputs = _export(fig, out_dir, "FigFuller_EnergyBreakdown")
    return outputs, [input_root / "fuller_stage_energy_breakdown_20260319_fullerexp_v1.csv"]


def _render_noise_accuracy(input_root: Path, out_dir: Path) -> tuple[list[Path], list[Path]]:
    dense_rows = _load_csv(input_root / "fuller_noise_accuracy_summary_s_dense_20260319_fullerexp_v1.csv")
    xs_rows = _load_csv(input_root / "fuller_noise_accuracy_summary_xs_sparse_20260319_fullerexp_v1.csv")
    xxs_rows = _load_csv(input_root / "fuller_noise_accuracy_summary_xxs_sparse_20260319_fullerexp_v1.csv")

    primary_crosstalk_rows = sorted(
        [
            row
            for row in dense_rows
            if abs(_float(row["gaussian_noise_std"])) < 1e-9
        ],
        key=lambda row: _float(row["crosstalk_alpha"]),
    )
    gaussian_anchor_rows = {
        alpha: sorted(
            [
                row
                for row in dense_rows
                if abs(_float(row["crosstalk_alpha"]) - alpha) < 1e-9
            ],
            key=lambda row: _float(row["gaussian_noise_std"]),
        )
        for alpha in (0.02, 0.05)
    }

    fig = plt.figure(figsize=(7.16, 4.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    crosstalk_ax = fig.add_subplot(gs[0, 0])
    gaussian_ax = fig.add_subplot(gs[0, 1])
    support_ax = fig.add_subplot(gs[1, :])

    crosstalk_ax.plot(
        [_float(row["crosstalk_alpha"]) for row in primary_crosstalk_rows],
        [_float(row["acc_drop_pp"]) for row in primary_crosstalk_rows],
        color=DEVICE_COLOR_MAP["FULLER"],
        marker="o",
        linewidth=2,
    )
    crosstalk_ax.set_title("Primary Crosstalk Trend\n(gaussian=0)")
    crosstalk_ax.set_xlabel("Crosstalk Alpha")
    crosstalk_ax.set_ylabel("Accuracy Drop (pp)")
    crosstalk_ax.set_ylim(bottom=0.0)

    for alpha, color, marker, linestyle in (
        (0.02, "#4C78A8", "o", "-"),
        (0.05, "#F58518", "s", "--"),
    ):
        rows = gaussian_anchor_rows[alpha]
        gaussian_ax.plot(
            [_float(row["gaussian_noise_std"]) for row in rows],
            [_float(row["acc_drop_pp"]) for row in rows],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            label=f"alpha={alpha:g}",
        )
    gaussian_ax.set_title("Gaussian Noise Support\n(fixed alpha)")
    gaussian_ax.set_xlabel("Gaussian Noise Std")
    gaussian_ax.set_ylabel("Accuracy Drop (pp)")
    gaussian_ax.set_ylim(bottom=0.0)
    gaussian_ax.legend(loc="upper left")

    xs_map = {row["profile"]: _float(row["acc_drop_pp"]) for row in xs_rows}
    xxs_map = {row["profile"]: _float(row["acc_drop_pp"]) for row in xxs_rows}
    x = np.arange(len(SUPPORT_PROFILE_ORDER))
    support_ax.plot(
        x,
        [xs_map[item] for item in SUPPORT_PROFILE_ORDER],
        marker="o",
        color="#4C78A8",
        linewidth=2,
        label="MobileViT-XS",
    )
    support_ax.plot(
        x,
        [xxs_map[item] for item in SUPPORT_PROFILE_ORDER],
        marker="s",
        color="#E45756",
        linestyle="--",
        linewidth=2,
        label="MobileViT-XXS",
    )
    support_ax.set_xticks(x, [SUPPORT_PROFILE_LABELS[item] for item in SUPPORT_PROFILE_ORDER])
    support_ax.set_ylabel("Accuracy Drop (pp)")
    support_ax.set_xlabel("Bounded Support Profile")
    support_ax.set_title("Secondary Model-Scale Support\n(mixed bounded profiles)")
    support_ax.set_ylim(bottom=0.0)
    support_ax.legend(ncols=2, loc="upper left")

    fig.suptitle("FigFuller_NoiseAccuracy", fontsize=11)
    outputs = _export(fig, out_dir, "FigFuller_NoiseAccuracy")
    inputs = [
        input_root / "fuller_noise_accuracy_summary_s_dense_20260319_fullerexp_v1.csv",
        input_root / "fuller_noise_accuracy_summary_xs_sparse_20260319_fullerexp_v1.csv",
        input_root / "fuller_noise_accuracy_summary_xxs_sparse_20260319_fullerexp_v1.csv",
    ]
    return outputs, inputs


def _render_ablation(input_root: Path, out_dir: Path) -> tuple[list[Path], list[Path]]:
    row_map = {
        row["mechanism_variant"]: row
        for row in _load_csv(input_root / "fuller_ablation_summary_20260319_fullerexp_v1.csv")
    }
    variants = [variant for variant in ABLATION_VARIANT_ORDER if variant in row_map]
    rows = [row_map[variant] for variant in variants]
    x = np.arange(len(variants))
    color_seq = [ABLATION_COLOR_SEQ[variant] for variant in variants]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.9), constrained_layout=True)
    panels = [
        ("latency_ms", "Latency (ms)"),
        ("energy_j", "Energy (J)"),
        ("acc_drop_pp", "Accuracy Drop (pp)"),
    ]
    for ax, (field, ylabel) in zip(axes, panels):
        values = [_float(row[field]) for row in rows]
        bars = ax.bar(x, values, color=color_seq, edgecolor="black", linewidth=0.8)
        for idx, bar in enumerate(bars):
            if idx == len(bars) - 1:
                bar.set_hatch("//")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{values[idx]:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
        ax.set_xticks(x, [ABLATION_VARIANT_LABELS[variant] for variant in variants], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
    fig.suptitle("FigFuller_Ablation", fontsize=11)

    outputs = _export(fig, out_dir, "FigFuller_Ablation")
    inputs = [
        input_root / "fuller_ablation_summary_20260319_fullerexp_v1.csv",
        input_root / "fuller_ablation_appendix_table_20260319_fullerexp_v1.csv",
    ]
    return outputs, inputs


def _render_scaling(input_root: Path, out_dir: Path) -> tuple[list[Path], list[Path]]:
    rows = _load_csv(input_root / "fuller_scaling_summary_20260319_fullerexp_v1.csv")
    variants = ("ASTRA", "FULLER")
    batches = sorted({_float(row["batch_size"]) for row in rows})
    seqs = sorted({_float(row["sequence_length"]) for row in rows})

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), constrained_layout=True)
    vmin = min(_float(row["throughput_tokens_s"]) for row in rows)
    vmax = max(_float(row["throughput_tokens_s"]) for row in rows)
    for ax, variant in zip(axes, variants):
        grid = np.zeros((len(batches), len(seqs)))
        for row in rows:
            if row["baseline_variant"] != variant:
                continue
            i = batches.index(_float(row["batch_size"]))
            j = seqs.index(_float(row["sequence_length"]))
            grid[i, j] = _float(row["throughput_tokens_s"])
        im = ax.imshow(grid, cmap="YlGnBu", origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(variant)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Batch Size")
        ax.set_xticks(np.arange(len(seqs)), [f"{int(item)}" for item in seqs])
        ax.set_yticks(np.arange(len(batches)), [f"{int(item)}" for item in batches])
        for i in range(len(batches)):
            for j in range(len(seqs)):
                ax.text(j, i, f"{grid[i, j]/1000:.1f}k", ha="center", va="center", fontsize=7, color="black")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Throughput (tokens/s)")
    fig.suptitle("FigFuller_Scaling", fontsize=11)

    outputs = _export(fig, out_dir, "FigFuller_Scaling")
    return outputs, [input_root / "fuller_scaling_summary_20260319_fullerexp_v1.csv"]


def _write_traceability(review_dir: Path, rows: list[dict[str, str]]) -> Path:
    review_dir.mkdir(parents=True, exist_ok=True)
    path = review_dir / "figure_traceability.csv"
    fieldnames = [
        "figure_id",
        "run_tag",
        "render_script",
        "render_command",
        "input_paths",
        "output_paths",
        "generated_at_utc",
        "key_render_params_summary",
        "literature_style_anchors",
        "literature_anchor_scope",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_review_manifest(
    review_dir: Path,
    run_tag: str,
    render_script: str,
    generated_at: str,
    figure_ids: list[str],
) -> Path:
    review_dir.mkdir(parents=True, exist_ok=True)
    path = review_dir / "review_manifest.json"
    payload = {
        "run_tag": run_tag,
        "artifact_layer": "report_figures",
        "render_script": render_script,
        "generated_at_utc": generated_at,
        "figure_ids": figure_ids,
        "figure_brief": str(review_dir / "data_figure_briefs.md"),
        "review_spec": str(ROOT / "experiments" / "DATA_FIGURE_REVIEW_SPEC.md"),
        "figure_spec": str(ROOT / "experiments" / "FIGURE_SPEC_IEEE.md"),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render fuller report figures.")
    parser.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--review_root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--run_tag", default=DEFAULT_RUN_TAG)
    args = parser.parse_args()

    _style()
    out_dir = args.output_root / args.run_tag
    review_dir = args.review_root / args.run_tag
    render_script = str(Path(__file__).resolve())
    render_command = " ".join(
        [
            "python3",
            str(Path(__file__).resolve()),
            "--input_root",
            str(args.input_root),
            "--output_root",
            str(args.output_root),
            "--review_root",
            str(args.review_root),
            "--run_tag",
            args.run_tag,
        ]
    )
    generated_at = datetime.now(timezone.utc).isoformat()

    trace_rows = []
    renderers = [
        ("FigFuller_DeviceComparison", _render_device_comparison),
        ("FigFuller_LatencyBreakdown", _render_latency_breakdown),
        ("FigFuller_EnergyBreakdown", _render_energy_breakdown),
        ("FigFuller_NoiseAccuracy", _render_noise_accuracy),
        ("FigFuller_Ablation", _render_ablation),
        ("FigFuller_Scaling", _render_scaling),
    ]
    for figure_id, renderer in renderers:
        outputs, inputs = renderer(args.input_root, out_dir)
        trace_rows.append(
            {
                "figure_id": figure_id,
                "run_tag": args.run_tag,
                "render_script": render_script,
                "render_command": render_command,
                "input_paths": "; ".join(str(path) for path in inputs),
                "output_paths": "; ".join(str(path) for path in outputs),
                "generated_at_utc": generated_at,
                "key_render_params_summary": "",
                "literature_style_anchors": "",
                "literature_anchor_scope": "",
                "notes": "",
            }
        )
        print(f"[fuller-figures] wrote {figure_id}")

    manifest_path = _write_review_manifest(
        review_dir=review_dir,
        run_tag=args.run_tag,
        render_script=render_script,
        generated_at=generated_at,
        figure_ids=[figure_id for figure_id, _ in renderers],
    )
    traceability_path = _write_traceability(review_dir, trace_rows)
    print(f"[fuller-figures] wrote {manifest_path}")
    print(f"[fuller-figures] wrote {traceability_path}")


if __name__ == "__main__":
    main()
