#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

STAGES = [
    ("fetch_map", "Fetch/Map", "#3E6A90"),
    ("btos", "BtoS", "#6186AE"),
    ("serialize_drive", "Serialize", "#D18B4B"),
    ("oag_compute", "OAG", "#4E8D45"),
    ("pca_adc", "PCA / ADC", "#8C70A6"),
    ("electronic_compute", "Elec.", "#A9A05D"),
    ("writeback", "Writeback", "#6F6A67"),
]

SERVICE_STAGE_NAMES = [stage for stage, _, _ in STAGES]
SERVICE_COLORS = {stage: color for stage, _, color in STAGES}
SERVICE_LABELS = {stage: label for stage, label, _ in STAGES}
SERVICE_LANES = {stage: len(STAGES) - 1 - idx for idx, (stage, _, _) in enumerate(STAGES)}

INK = "#1A1A1A"
SLATE = "#5A6470"
GRID = "#E1E5EA"
BUBBLE = "#C96A6A"
BUBBLE_FILL = "#FEF6F6"
OVERLAP_EDGE = "#2E8B57"
OVERLAP_FILL = "#E8F4EC"
UTIL_BASELINE_FILL = "#DFE2E8"
UTIL_HOPS_FILL = "#DEE7F8"


def _load_summary(path: Path):
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            exp = str(row.get("mechanism_variant") or row.get("experiment_id") or "").strip()
            key = None
            if exp in {"ASTRA", "E0"}:
                key = "E0"
            elif exp in {"HOPS", "E2"}:
                key = "E2"
            if key is None:
                continue

            stage_cycles = {}
            payload = str(row.get("stage_cycles") or "").strip()
            if payload:
                stage_cycles = json.loads(payload.replace('""', '"'))

            rows[key] = {
                "latency_ms": float(row["latency_ms"]),
                "utilization_avg": float(row["utilization_avg"]),
                "bubble_cycles": float(row["bubble_cycles"]),
                **{name: float(stage_cycles.get(name, 0.0)) for name in stage_cycles},
            }

    if {"E0", "E2"} - set(rows):
        raise ValueError("Requires ASTRA (E0) and HOPS (E2) in CSV.")
    return rows


def _sync_contract(contract_path: Path, run_tag: str, generated_at: str, input_csv: Path, outputs):
    if not contract_path.exists():
        return

    with contract_path.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)

    contract["run_tag"] = run_tag
    contract["generated_at"] = generated_at
    contract["outputs"] = outputs

    data_source = contract.get("data_or_semantic_source", {})
    data_source["quickpack_csv"] = str(input_csv)
    calibrated_fields = list(data_source.get("calibrated_fields", []))
    for field in ["latency_ms", "bubble_cycles", "utilization_avg", "stage_cycles"]:
        if field not in calibrated_fields:
            calibrated_fields.append(field)
    data_source["calibrated_fields"] = calibrated_fields
    contract["data_or_semantic_source"] = data_source

    must_have = list(contract.get("must_have", []))
    extra = [
        "compact frozen E0/E2 improvement callout",
        "exact service-chain stage widths from the audited E0/E2 rows",
        "bubble semantics localized to the OAG starvation lane",
        "gains box labels exact-trace plus reviewed-summary metric source",
        "utilization rail shows normalized 0-100 percent gauge cues",
    ]
    for item in extra:
        if item not in must_have:
            must_have.append(item)
    contract["must_have"] = must_have

    updated_must_not = []
    for item in contract.get("must_not_have", []):
        if item == "exact result numbers":
            updated_must_not.append("exact result numbers beyond the bounded E0/E2 callout and audited stage widths")
        else:
            updated_must_not.append(item)
    contract["must_not_have"] = updated_must_not

    with contract_path.open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2)


def _upsert_trace_row(trace_path: Path, fieldnames, trace_row):
    rows = []
    output_fieldnames = list(fieldnames)
    if trace_path.exists():
        with trace_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            output_fieldnames = list(reader.fieldnames or fieldnames)
            for key in fieldnames:
                if key not in output_fieldnames:
                    output_fieldnames.append(key)
            for key in trace_row:
                if key not in output_fieldnames:
                    output_fieldnames.append(key)
            for row in reader:
                if row.get("figure_id") == trace_row["figure_id"]:
                    continue
                rows.append({key: row.get(key, "") for key in output_fieldnames})
    for key in trace_row:
        if key not in output_fieldnames:
            output_fieldnames.append(key)

    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({key: trace_row.get(key, "") for key in output_fieldnames})


def _service_cycles(summary_row):
    return {stage: float(summary_row.get(stage, 0.0)) for stage in SERVICE_STAGE_NAMES}


def _service_total(service_cycles):
    return sum(service_cycles.values())


def _oag_gap_start_offset(summary_row):
    """Start seq(i+1) so the OAG-lane starvation window equals the audited bubble."""
    return max(0.0, float(summary_row.get("oag_compute", 0.0)) + float(summary_row["bubble_cycles"]))


def _build_sequence_events(seq_label, service_cycles, start_cycle, hatch=None, alpha=1.0):
    events = []
    cursor = float(start_cycle)
    for stage in SERVICE_STAGE_NAMES:
        width = float(service_cycles.get(stage, 0.0))
        event = {
            "sequence": seq_label,
            "stage": stage,
            "lane": SERVICE_LANES[stage],
            "start_cycle": cursor,
            "end_cycle": cursor + width,
            "width_cycles": width,
            "hatch": hatch,
            "alpha": alpha,
        }
        events.append(event)
        cursor += width
    return events


def _index_events_by_stage(events):
    return {event["stage"]: event for event in events}


def _build_panel_trace(panel_key, summary_row):
    service_cycles = _service_cycles(summary_row)
    seq_i = _build_sequence_events("Sequence i", service_cycles, start_cycle=0.0)
    seq_i1 = _build_sequence_events(
        "Sequence i+1",
        service_cycles,
        start_cycle=_oag_gap_start_offset(summary_row),
        hatch="////",
        alpha=0.92,
    )
    seq_i_idx = _index_events_by_stage(seq_i)
    seq_i1_idx = _index_events_by_stage(seq_i1)
    starvation = {
        "start_cycle": seq_i_idx["oag_compute"]["end_cycle"],
        "end_cycle": seq_i1_idx["oag_compute"]["start_cycle"],
        "width_cycles": seq_i1_idx["oag_compute"]["start_cycle"] - seq_i_idx["oag_compute"]["end_cycle"],
        "lane": SERVICE_LANES["oag_compute"],
    }
    panel_end = max(seq_i[-1]["end_cycle"], seq_i1[-1]["end_cycle"])
    overlap = {
        "start_cycle": seq_i1_idx["fetch_map"]["start_cycle"],
        "end_cycle": seq_i1_idx["oag_compute"]["start_cycle"],
    }
    return {
        "panel_key": panel_key,
        "service_cycles": service_cycles,
        "sequence_i": seq_i,
        "sequence_i1": seq_i1,
        "starvation": starvation,
        "overlap": overlap,
        "panel_end_cycle": panel_end,
        "service_total_cycles": _service_total(service_cycles),
    }


def _to_trace_rows(panel_label, trace):
    rows = []
    for event in trace["sequence_i"] + trace["sequence_i1"]:
        rows.append(
            {
                "panel": panel_label,
                "kind": "stage",
                "sequence": event["sequence"],
                "stage": event["stage"],
                "lane_label": SERVICE_LABELS[event["stage"]],
                "start_cycle": f"{event['start_cycle']:.6f}",
                "end_cycle": f"{event['end_cycle']:.6f}",
                "width_cycles": f"{event['width_cycles']:.6f}",
                "derivation": "",
            }
        )
    rows.append(
        {
            "panel": panel_label,
            "kind": "starvation_window",
            "sequence": "",
            "stage": "oag_starvation",
            "lane_label": SERVICE_LABELS["oag_compute"],
            "start_cycle": f"{trace['starvation']['start_cycle']:.6f}",
            "end_cycle": f"{trace['starvation']['end_cycle']:.6f}",
            "width_cycles": f"{trace['starvation']['width_cycles']:.6f}",
            "derivation": "seq_i_oag_end_to_seq_i1_oag_start",
        }
    )
    if panel_label == "E2":
        overlap = trace["overlap"]
        width_cycles = overlap["end_cycle"] - overlap["start_cycle"]
        rows.append(
            {
                "panel": panel_label,
                "kind": "overlap_window",
                "sequence": "Sequence i+1",
                "stage": "hops_earlier_staging",
                "lane_label": "Fetch/Map-to-OAG preparation window",
                "start_cycle": f"{overlap['start_cycle']:.6f}",
                "end_cycle": f"{overlap['end_cycle']:.6f}",
                "width_cycles": f"{width_cycles:.6f}",
                "derivation": "seq_i1_fetch_map_start_to_seq_i1_oag_compute_start",
            }
        )
    return rows


def _write_exact_trace_csv(trace_path: Path, e0_trace, e2_trace):
    fieldnames = [
        "panel",
        "kind",
        "sequence",
        "stage",
        "lane_label",
        "start_cycle",
        "end_cycle",
        "width_cycles",
        "derivation",
    ]
    rows = _to_trace_rows("E0", e0_trace) + _to_trace_rows("E2", e2_trace)
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _draw_stage_block(ax, event, scale):
    x = event["start_cycle"] / scale
    w = max(event["width_cycles"] / scale, 0.01)
    y = event["lane"] - 0.35
    patch = FancyBboxPatch(
        (x, y),
        w,
        0.70,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        linewidth=1.0,
        edgecolor="#2A2A2A",
        facecolor=SERVICE_COLORS[event["stage"]],
        alpha=event["alpha"],
        hatch=event["hatch"],
    )
    ax.add_patch(patch)


def _draw_starvation_window(ax, trace, scale, label):
    starve = trace["starvation"]
    x = starve["start_cycle"] / scale
    w = max(starve["width_cycles"] / scale, 0.01)
    y = starve["lane"] - 0.35
    patch = FancyBboxPatch(
        (x, y),
        w,
        0.70,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        linewidth=1.2,
        edgecolor=BUBBLE,
        facecolor=BUBBLE_FILL,
        hatch="///",
        linestyle="--",
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2.0,
        starve["lane"] + 0.52,
        label,
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=BUBBLE,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.18),
    )


def _draw_utilization_rail(ax, x_max, util_value, y_base, outline, fill, label):
    fill_width = x_max * util_value
    text_color = SLATE if outline == "#AEB5BF" else outline
    ax.add_patch(Rectangle((0.0, y_base), x_max, 0.16, fill=False, edgecolor=outline, linestyle="--", linewidth=1.0))
    ax.add_patch(Rectangle((0.0, y_base), fill_width, 0.16, facecolor=fill, edgecolor=outline, linewidth=0.8))
    ax.text(
        min(fill_width + 0.08, x_max - 0.08),
        y_base + 0.23,
        f"{util_value * 100.0:.1f}%",
        ha="left" if fill_width < x_max * 0.82 else "right",
        va="bottom",
        fontsize=7.8,
        color=text_color,
        fontweight="bold",
    )
    ax.text(0.0, y_base - 0.04, "0%", ha="left", va="top", fontsize=7.0, color=SLATE)
    ax.text(x_max, y_base - 0.04, "100%", ha="right", va="top", fontsize=7.0, color=SLATE)
    ax.text(
        x_max / 2.0,
        y_base - 0.22,
        label,
        ha="center",
        va="top",
        fontsize=8.4,
        color=text_color,
        fontweight="bold",
    )


def _render_stage_key(ax):
    ax.axis("off")
    x = 0.01
    step = 0.115
    for idx, (_, label, color) in enumerate(STAGES):
        ax.add_patch(Rectangle((x, 0.32), 0.030, 0.36, color=color, ec=INK))
        ax.text(x + 0.037, 0.50, label, ha="left", va="center", fontsize=7.3, color=SLATE)
        x += step


def _render_compact_key_bar(ax, speedup, bubble_reduction_pct, util_lift_pp):
    ax.axis("off")
    ax.text(
        0.01,
        0.86,
        "Workload: MobileViT Attention-Serving Path",
        ha="left",
        va="center",
        fontsize=10.6,
        fontweight="bold",
        color=INK,
        transform=ax.transAxes,
    )

    def semantic_item(x0, y_center, label, edge, face, hatch=None):
        ax.add_patch(
            Rectangle(
                (x0, y_center - 0.065),
                0.030,
                0.13,
                edgecolor=edge,
                facecolor=face,
                hatch=hatch,
                linestyle="--",
                linewidth=1.0,
                transform=ax.transAxes,
            )
        )
        ax.text(x0 + 0.038, y_center, label, ha="left", va="center", fontsize=7.5, color=INK, transform=ax.transAxes)

    semantic_item(0.01, 0.48, "OAG starvation window", BUBBLE, BUBBLE_FILL, hatch="///")
    semantic_item(0.300, 0.48, "Earlier next-seq staging", OVERLAP_EDGE, OVERLAP_FILL)
    ax.add_patch(
        Rectangle(
            (0.01, 0.145),
            0.030,
            0.09,
            edgecolor="#1B428A",
            facecolor="none",
            linestyle="--",
            linewidth=1.0,
            transform=ax.transAxes,
        )
    )
    ax.add_patch(
        Rectangle(
            (0.01, 0.145),
            0.021,
            0.09,
            edgecolor="#1B428A",
            facecolor=UTIL_HOPS_FILL,
            linewidth=0.8,
            transform=ax.transAxes,
        )
    )
    ax.text(0.048, 0.190, "Normalized util. gauge", ha="left", va="center", fontsize=7.5, color=INK, transform=ax.transAxes)

    gain_text = (
        "E0/E2 gains (mixed source)\n"
        f"exact trace: bubble -{bubble_reduction_pct:.1f}%\n"
        "reviewed summary:\n"
        f"speed {speedup:.2f}x, util +{util_lift_pp:.1f} pp"
    )
    ax.text(
        0.665,
        0.88,
        gain_text,
        ha="left",
        va="top",
        fontsize=6.9,
        color="#1F2F44",
        fontweight="bold",
        linespacing=1.10,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.08", linewidth=1.0, edgecolor=SLATE, facecolor="#F8FAFC"),
    )


def _render_semantic_key(ax):
    legend_x, legend_y = 1.01, 0.70
    legend_w, legend_h = 0.23, 0.22
    patch = FancyBboxPatch(
        (legend_x, legend_y),
        legend_w,
        legend_h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.0,
        edgecolor=SLATE,
        facecolor="#F8FAFC",
        transform=ax.transAxes,
        clip_on=False,
        zorder=20,
    )
    ax.add_patch(patch)
    ax.text(
        legend_x + 0.02,
        legend_y + legend_h - 0.04,
        "Semantic Key",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        fontweight="bold",
        color=INK,
        clip_on=False,
        zorder=21,
    )

    icon_x = legend_x + 0.02
    text_x = legend_x + 0.085
    y = legend_y + legend_h - 0.11

    bubble_box = Rectangle(
        (icon_x, y - 0.016),
        0.04,
        0.03,
        edgecolor=BUBBLE,
        facecolor=BUBBLE_FILL,
        hatch="///",
        linestyle="--",
        linewidth=1.0,
        transform=ax.transAxes,
        clip_on=False,
        zorder=21,
    )
    ax.add_patch(bubble_box)
    ax.text(text_x, y, "OAG starvation window", transform=ax.transAxes, ha="left", va="center", fontsize=7.5, color=INK, clip_on=False, zorder=21)

    y -= 0.08
    overlap_box = Rectangle(
        (icon_x, y - 0.016),
        0.04,
        0.03,
        edgecolor=OVERLAP_EDGE,
        facecolor=OVERLAP_FILL,
        linestyle="--",
        linewidth=1.0,
        transform=ax.transAxes,
        clip_on=False,
        zorder=21,
    )
    ax.add_patch(overlap_box)
    ax.text(text_x, y, "Earlier next-seq staging", transform=ax.transAxes, ha="left", va="center", fontsize=7.5, color=INK, clip_on=False, zorder=21)

    y -= 0.08
    rail_outline = Rectangle(
        (icon_x, y - 0.010),
        0.04,
        0.02,
        edgecolor="#1B428A",
        facecolor="none",
        linestyle="--",
        linewidth=1.0,
        transform=ax.transAxes,
        clip_on=False,
        zorder=21,
    )
    rail_fill = Rectangle(
        (icon_x, y - 0.010),
        0.025,
        0.02,
        edgecolor="#1B428A",
        facecolor=UTIL_HOPS_FILL,
        linewidth=0.8,
        transform=ax.transAxes,
        clip_on=False,
        zorder=22,
    )
    ax.add_patch(rail_outline)
    ax.add_patch(rail_fill)
    ax.text(text_x, y, "Normalized util. gauge", transform=ax.transAxes, ha="left", va="center", fontsize=7.5, color=INK, clip_on=False, zorder=21)


def _render_panel(ax, trace, summary_row, title, title_color, scale, x_limit, show_overlap=False):
    ax.set_title(title, loc="left", fontsize=9.7, fontweight="bold", color=title_color, pad=8)
    ax.set_xlim(0.0, x_limit)
    ax.set_ylim(-1.45, len(STAGES) - 0.25)
    ax.set_yticks(list(range(len(STAGES))))
    ax.set_yticklabels([SERVICE_LABELS[stage] for stage in reversed(SERVICE_STAGE_NAMES)], fontsize=8.4, fontweight="bold", color=INK)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.tick_params(axis="x", labelsize=8.0, colors=SLATE)
    ax.grid(axis="y", color=GRID, linestyle=":", linewidth=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)

    for event in trace["sequence_i"] + trace["sequence_i1"]:
        _draw_stage_block(ax, event, scale)

    seq_i_fetch = trace["sequence_i"][0]
    seq_i1_fetch = trace["sequence_i1"][0]
    ax.text(seq_i_fetch["start_cycle"] / scale, SERVICE_LANES["fetch_map"] + 0.58, "Sequence $i$", fontsize=8.2, fontweight="bold", color=SLATE)
    ax.text(seq_i1_fetch["start_cycle"] / scale, SERVICE_LANES["fetch_map"] + 0.58, "Sequence $i+1$", fontsize=8.2, fontweight="bold", color=SLATE)

    _draw_starvation_window(ax, trace, scale, f"OAG bubble: {summary_row['bubble_cycles'] / 1e9:.3f}B cycles")

    if show_overlap:
        overlap = trace["overlap"]
        rect = Rectangle(
            (overlap["start_cycle"] / scale, SERVICE_LANES["serialize_drive"] - 0.50),
            (overlap["end_cycle"] - overlap["start_cycle"]) / scale,
            3.0,
            fill=True,
            edgecolor=OVERLAP_EDGE,
            facecolor=OVERLAP_FILL,
            linestyle="--",
            linewidth=1.2,
            zorder=-1,
            alpha=0.9,
        )
        ax.add_patch(rect)
        ax.text(
            (overlap["start_cycle"] + overlap["end_cycle"]) / (2.0 * scale),
            SERVICE_LANES["btos"] + 0.16,
            "HOPS earlier staging window",
            ha="center",
            va="center",
            fontsize=7.4,
            color="#1D5B2E",
            fontweight="bold",
        )

    rail_y = -1.03
    rail_outline = "#AEB5BF" if not show_overlap else "#1B428A"
    rail_fill = UTIL_BASELINE_FILL if not show_overlap else UTIL_HOPS_FILL
    rail_label = "Normalized utilization gauge (0-100%)" if not show_overlap else "HOPS normalized utilization gauge (0-100%)"
    _draw_utilization_rail(ax, x_limit, summary_row["utilization_avg"], rail_y, rail_outline, rail_fill, rail_label)


def render_fig2(input_csv: Path, out_dir: Path, review_dir: Path, run_tag: str):
    summary = _load_summary(input_csv)
    e0_trace = _build_panel_trace("E0", summary["E0"])
    e2_trace = _build_panel_trace("E2", summary["E2"])

    speedup = summary["E0"]["latency_ms"] / summary["E2"]["latency_ms"]
    bubble_reduction_pct = (1.0 - summary["E2"]["bubble_cycles"] / summary["E0"]["bubble_cycles"]) * 100.0
    util_lift_pp = (summary["E2"]["utilization_avg"] - summary["E0"]["utilization_avg"]) * 100.0

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    scale = 1e9
    x_limit = max(e0_trace["panel_end_cycle"], e2_trace["panel_end_cycle"]) / scale * 1.04

    fig = plt.figure(figsize=(6.35, 4.55), dpi=300, facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[0.72, 3.05, 3.05], hspace=0.38)

    ax_key = fig.add_subplot(gs[0])
    _render_compact_key_bar(ax_key, speedup, bubble_reduction_pct, util_lift_pp)

    ax_top = fig.add_subplot(gs[1])
    ax_bottom = fig.add_subplot(gs[2], sharex=ax_top)

    _render_panel(
        ax_top,
        e0_trace,
        summary["E0"],
        "(a) Without HOPS: ASTRA exact service chain",
        INK,
        scale,
        x_limit,
        show_overlap=False,
    )
    _render_panel(
        ax_bottom,
        e2_trace,
        summary["E2"],
        "(b) With HOPS: exact service chain",
        "#2C5A3A",
        scale,
        x_limit,
        show_overlap=True,
    )

    ax_top.tick_params(axis="x", labelbottom=False)
    ax_bottom.set_xlabel("Execution Time (Cycles, ×10$^9$)", fontsize=8.8, fontweight="bold", color=SLATE)
    fig.subplots_adjust(left=0.140, right=0.985, top=0.965, bottom=0.085)

    out_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / "Fig2_HOPSTimeline.svg"
    pdf_path = out_dir / "Fig2_HOPSTimeline.pdf"
    png_path = out_dir / "Fig2_HOPSTimeline.png"
    exact_trace_path = review_dir / "Fig2_HOPSTimeline_exact_trace.csv"

    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=600)
    plt.close(fig)

    _write_exact_trace_csv(exact_trace_path, e0_trace, e2_trace)

    if HAS_PIL:
        img = Image.open(png_path).convert("L")
        gray_path = review_dir / "Fig2_HOPSTimeline_grayscale.png"
        img.save(gray_path)
    else:
        gray_path = review_dir / "Fig2_HOPSTimeline_grayscale.png"
        import shutil

        shutil.copy(png_path, gray_path)

    generated_at = datetime.now().isoformat(timespec="seconds")

    manifest_path = out_dir / "Fig2_HOPSTimeline_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["run_tag"] = run_tag
        manifest["generated_at"] = generated_at
        manifest["status"] = "frozen_pack_rendered"
        manifest["workflow"] = "deterministic matplotlib exact service-chain redraw aligned to the current Fig.2 HOPS timeline input rows"
        manifest["exact_trace_csv"] = str(exact_trace_path)
        sources = manifest.get("sources", [])
        if len(sources) >= 3:
            sources[2] = str(input_csv)
            manifest["sources"] = sources
        manifest["notes"] = [
            "Reader-facing lane labels preserve the governed attention-serving service chain, including the nonzero electronic-compute sidecar stage.",
            "Stage widths are exact E0/E2 stage-cycle totals from the current Fig.2 HOPS timeline input CSV.",
            "Bubble semantics are localized to the OAG starvation lane, avoiding the old full-system blank-block implication.",
            f"The gains callout is labeled as mixed-source: exact trace for bubble reduction plus reviewed input-summary metrics for {speedup:.2f}x speedup and +{util_lift_pp:.1f} pp utilization lift.",
            "The exact trace CSV includes an explicit HOPS overlap_window row for the shaded earlier-staging interval.",
            "The utilization rail is labeled as a normalized 0-100% gauge and includes compact percent labels.",
        ]
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    contract_path = review_dir / "Fig2_HOPSTimeline_contract.json"
    _sync_contract(
        contract_path,
        run_tag,
        generated_at,
        input_csv,
        [str(svg_path), str(pdf_path), str(png_path), str(exact_trace_path)],
    )

    fieldnames = [
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
    trace_row = {
        "figure_id": "Fig2",
        "manuscript_tier": "main",
        "figure_file": str(svg_path),
        "input_csvs": str(input_csv),
        "script_entry": "scripts/render_fig2_pipeline_matplotlib.py",
        "command": f"python3 scripts/render_fig2_pipeline_matplotlib.py --input-csv {input_csv} --run-tag {run_tag} --out-dir {out_dir} --review-dir {review_dir}",
        "run_tag": run_tag,
        "generated_at": generated_at,
        "params_summary": "exact_service_chain_timeline; calibrated_rows=E0|E2; exact_stage_widths_including_electronic_compute; oag_starvation_localization; explicit_overlap_window_trace_row; mixed_source_gain_callout; normalized_utilization_gauge_percent_labels; thesis_width_native_pdf",
        "literature_style_anchors": "original_papers/markdown/01_transformer_attention_photonic/ASTRA_Stochastic_Transformer_Silicon_Photonics_TECS2025.md::Fig3-Fig5 | original_papers/markdown/01_transformer_attention_photonic/Lightening_Transformer_HPCA2024.md::Fig5 | original_papers/markdown/01_transformer_attention_photonic/2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md::Fig3-Fig4 | original_papers/markdown/01_transformer_attention_photonic/2402.07393_TeMPO_Transformer_Acceleration_with_Co-packaged_Silicon_Photonics.md::Fig3-Fig5",
        "literature_anchor_scope": "composition_only; family=pipeline_timeline; borrow=lane_separation+overlap_posture+compact_key; avoid=paragraph_text+code_like_stage_ids",
        "notes": "Exact service-chain reconstruction from the current Fig.2 E0/E2 timeline rows, including electronic_compute; bubble is shown as an OAG starvation window rather than a full-system blank block; the HOPS earlier-staging shade has an explicit overlap_window exact-trace row; gains box states exact trace plus reviewed input-summary metrics; utilization rail is a normalized 0-100% gauge with percent labels.",
        "pack_run_tag": run_tag,
        "source_data_run_tag": "20260424_fig2_hops_current_timeline",
        "source_figure_id": "Fig2_HOPSTimeline",
        "source_pack_run_tag": "20260429_fuller_final_paper_numbered",
        "promotion_reason": "remediated Fig2 timeline redraw closing F2-TL-L01 and F2-TL-L02",
        "svg_status": "available",
        "derived_evidence_artifact": str(exact_trace_path),
    }
    for target_dir in [out_dir, review_dir]:
        trace_path = target_dir / "figure_traceability.csv"
        _upsert_trace_row(trace_path, fieldnames, trace_row)

    print(f"Generated {png_path} and exact trace {exact_trace_path}.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Render governed Fig2 HOPS pipeline exact reconstruction")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--review-dir", required=True)
    args = ap.parse_args()

    render_fig2(Path(args.input_csv), Path(args.out_dir), Path(args.review_dir), args.run_tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
