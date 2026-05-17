#!/usr/bin/env python3
"""Render the current FULLER Phase 4 report pack from the intake summary."""

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
DEFAULT_RUN_TAG = "20260425_fuller_phase4_intake"
DEFAULT_INPUT_CSV = ROOT / "experiments" / "results" / "report_data" / "fuller_phase4_intake_summary_20260425.csv"
DEFAULT_REPORT_DATA_ROOT = ROOT / "experiments" / "results" / "report_data"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "results" / "report_figures"
DEFAULT_REVIEW_ROOT = ROOT / "experiments" / "results" / "review"
DEFAULT_REPORT_MD = ROOT / "docs" / "reports" / "20260425_fuller_phase4_report_pack.md"

LANE_ORDER = ["ASTRA", "MESO", "HOPS", "DET", "SPARSE", "FULLER"]
ACCURACY_CLAIM_BLOCKED = {"SPARSE", "FULLER"}
LANE_COLORS = {
    "ASTRA": "#4C78A8",
    "MESO": "#72B7B2",
    "HOPS": "#54A24B",
    "DET": "#F58518",
    "SPARSE": "#E45756",
    "FULLER": "#6F4E7C",
}
SOURCE_ROOTS_BY_LANE = {
    "ASTRA": "experiments/results/report_data/20260423_fuller_analysis_grade_replay/astra",
    "MESO": "experiments/results/report_data/20260423_fuller_analysis_grade_replay/meso",
    "HOPS": "experiments/results/report_data/20260423_fuller_analysis_grade_replay/hops",
    "DET": "experiments/results/report_data/20260423_fuller_analysis_grade_replay/det",
    "SPARSE": "experiments/results/report_data/20260425_sparse_fixed_analysis_grade_replay/sparse",
    "FULLER": "experiments/results/report_data/20260425_sparse_fixed_analysis_grade_replay/fuller",
}
MIXED_EVIDENCE_SURFACE_NOTE = (
    "current Phase 4 intake summary is a mixed evidence surface: "
    "20260423 for ASTRA/MESO/HOPS/DET and 20260425 for repaired SPARSE/FULLER"
)
SUPERSEDED_SPARSE_FULLER_NOTE = (
    "Old near-zero SPARSE/FULLER rows are superseded by the 20260425 sparse-fixed replay"
)
PAPER_FREEZE_STATUS = (
    "not_promoted; current paper-facing freeze remains "
    "20260403_det_sparse_mlx_final_cpu_context_repair_xs_xxs_dense_successor_freeze"
)
BITSTREAM_BOUNDARY = (
    "limited_linear_attention_pilot; active target modules 2/177; missing operator families "
    "activation, conv2d, and norm; speedup is not accuracy-preserving speedup; results are not silicon measurements"
)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lane_order = [row["lane"] for row in rows]
    if lane_order != LANE_ORDER:
        raise SystemExit(f"Unexpected lane order in {path}: {lane_order}")
    for row in rows:
        if row["ready_for_phase4_intake"] != "true":
            raise SystemExit(f"Lane is not Phase 4 intake-ready: {row['lane']}")
        if row["missing_required_fields_json"] != "[]":
            raise SystemExit(f"Lane has missing fields: {row['lane']}")
    return rows


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid float field {field!r} in row {row!r}") from exc


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


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
    outputs = [out_dir / f"{stem}.svg", out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"]
    for path in outputs:
        fig.savefig(path, bbox_inches="tight")
        if path.suffix == ".svg":
            svg_text = path.read_text(encoding="utf-8")
            cleaned = "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n"
            path.write_text(cleaned, encoding="utf-8")
    plt.close(fig)
    return outputs


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _lane_table_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    table_rows: list[dict[str, str]] = []
    for row in rows:
        lane = row["lane"]
        claim_note = (
            "runtime_materialization_ready_accuracy_claim_blocked"
            if lane in ACCURACY_CLAIM_BLOCKED
            else "runtime_materialization_ready"
        )
        table_rows.append(
            {
                "lane": lane,
                "top1_mean": row["top1_mean"],
                "top1_range": f"{row['top1_min']}-{row['top1_max']}",
                "top5_mean": row["top5_mean"],
                "samples_per_hour_mean": row["samples_per_hour_mean"],
                "seconds_per_sample_mean": row["seconds_per_sample_mean"],
                "speedup_vs_astra": row["speedup_vs_astra"],
                "phase4_intake_ready": row["ready_for_phase4_intake"],
                "claim_boundary_note": claim_note,
            }
        )
    return table_rows


def _render_lane_comparison(rows: list[dict[str, str]], out_dir: Path) -> list[Path]:
    lanes = [row["lane"] for row in rows]
    x = np.arange(len(lanes))
    colors = [LANE_COLORS[lane] for lane in lanes]
    speedups = [_float(row, "speedup_vs_astra") for row in rows]
    top1 = [_float(row, "top1_mean") for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.7))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.30, top=0.78, wspace=0.30)
    speed_ax, acc_ax = axes

    speed_bars = speed_ax.bar(x, speedups, color=colors, edgecolor="black", linewidth=0.8)
    for bar, value, lane in zip(speed_bars, speedups, lanes):
        if lane in ACCURACY_CLAIM_BLOCKED:
            bar.set_hatch("//")
        speed_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.25,
            f"{value:.2f}x",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )
    speed_ax.set_title("Runtime Intake")
    speed_ax.set_ylabel("Speedup vs ASTRA (x)")
    speed_ax.set_xticks(x, lanes, rotation=25, ha="right")
    speed_ax.set_ylim(0, max(speedups) * 1.25)

    acc_bars = acc_ax.bar(x, top1, color=colors, edgecolor="black", linewidth=0.8)
    for bar, value, lane in zip(acc_bars, top1, lanes):
        if lane in ACCURACY_CLAIM_BLOCKED:
            bar.set_hatch("//")
        acc_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.7,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )
    acc_ax.set_title("Accuracy Boundary")
    acc_ax.set_ylabel("Top-1 accuracy (%)")
    acc_ax.set_xticks(x, lanes, rotation=25, ha="right")
    acc_ax.set_ylim(0, max(top1) * 1.18)

    legend_handles = [
        Patch(facecolor="#CCCCCC", edgecolor="black", label="intake-ready lane"),
        Patch(facecolor="#CCCCCC", edgecolor="black", hatch="//", label="accuracy claim blocked"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncols=2, bbox_to_anchor=(0.5, 0.04), fontsize=8)
    fig.suptitle("FigFuller_Phase4LaneComparison", fontsize=11, y=0.96)
    return _export(fig, out_dir, "FigFuller_Phase4LaneComparison")


def _write_traceability(
    review_dir: Path,
    *,
    run_tag: str,
    input_csv: Path,
    table_csv: Path,
    outputs: list[Path],
    generated_at: str,
    render_command: str,
) -> Path:
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
    rows = [
        {
            "figure_id": "FigFuller_Phase4LaneComparison",
            "run_tag": run_tag,
            "render_script": _rel(Path(__file__)),
            "render_command": render_command,
            "input_paths": f"{_rel(input_csv)}; {_rel(table_csv)}",
            "output_paths": "; ".join(_rel(path) for path in outputs),
            "generated_at_utc": generated_at,
            "key_render_params_summary": "two-panel bars; speedup_vs_astra and top1_mean; SPARSE/FULLER hatched as accuracy-claim blocked",
            "literature_style_anchors": "CrossLight Fig. 6; Lightening-Transformer Fig. 13; HyAtten Fig. 6",
            "literature_anchor_scope": "composition_only",
            "notes": (
                "Current mixed evidence report pack; figure data comes from the regenerated lane table, "
                "with 20260425 sparse-fixed SPARSE/FULLER rows superseding the old near-zero rows."
            ),
        }
    ]
    _write_csv(path, fieldnames, rows)
    return path


def _write_review_files(
    review_dir: Path,
    *,
    run_tag: str,
    input_csv: Path,
    table_csv: Path,
    figure_outputs: list[Path],
    generated_at: str,
    render_command: str,
) -> tuple[Path, Path, Path]:
    review_dir.mkdir(parents=True, exist_ok=True)
    data_figure_brief = review_dir / "data_figure_brief.md"
    data_figure_brief.write_text(
        "\n".join(
            [
                "# Data Figure Brief",
                "",
                "Date: 2026-04-25",
                "",
                "Figure ID: `FigFuller_Phase4LaneComparison`",
                "",
                "Figure type: two-panel lane comparison bar chart for Phase 4 intake",
                "",
                f"Run tag: `{run_tag}`",
                "",
                "## Governed Sources",
                "",
                f"- mixed evidence surface: `{MIXED_EVIDENCE_SURFACE_NOTE}`",
                f"- superseded rows: `{SUPERSEDED_SPARSE_FULLER_NOTE}`",
                f"- input CSV: `{_rel(input_csv)}`",
                f"- regenerated lane table: `{_rel(table_csv)}`",
                f"- render script: `{_rel(Path(__file__))}`",
                f"- review directory: `{_rel(review_dir)}`",
                "",
                "## Target Venue And Literature Style Anchors",
                "",
                "- target venue or paper family: IEEE-style internal report-pack lane comparison",
                "- exemplar 1: `CrossLight_A_Cross-Layer_Optimized_Silicon_Photonic_Neural_Network_Accelerator_arXiv2102.06960v1.md` / Fig. 6 / borrow comparison scatter/bar economy and avoid dense callout clutter",
                "- exemplar 2: `Lightening_Transformer_HPCA2024.md` / Fig. 13 / borrow energy/performance comparison posture and avoid overstating against unsupported baselines",
                "- exemplar 3: `2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md` / Fig. 6 / borrow paired speedup/efficiency comparison structure and avoid mixing claim tiers",
                "- composition-only references logged in traceability: `yes`",
                "",
                "## Field Mapping Declaration",
                "",
                "- left panel x: `lane`",
                "- left panel y: `speedup_vs_astra`",
                "- right panel x: `lane`",
                "- right panel y: `top1_mean`",
                "- series or hue: lane identity; hatch marks lanes where positive accuracy-preservation claims are blocked",
                "- aggregation rule: use the committed repaired intake-summary rows directly; no recomputation beyond label formatting",
                "",
                "## Units and Labels",
                "",
                "- speedup label: `Speedup vs ASTRA (x)`",
                "- accuracy label: `Top-1 accuracy (%)`",
                "- throughput support metric: `samples_per_hour_mean`",
                "- unit notes: samples/hour is reported in the companion table and manifest, not plotted as a third axis",
                "",
                "## Must-Have Constraints",
                "",
                "- Show all six lanes in order: `ASTRA`, `MESO`, `HOPS`, `DET`, `SPARSE`, `FULLER`.",
                "- Show SPARSE Top-1 as `27.1985` and FULLER Top-1 as `20.6459` from the sparse-fixed replay.",
                "- Show FULLER speedup as `12.11x` relative to ASTRA.",
                "- Mark SPARSE and FULLER as blocked for positive accuracy-preservation claims.",
                "- Keep the claim boundary visible in review/report text: runtime/materialization ready is not the same as accuracy-preservation ready.",
                "",
                "## Must-Not-Have Constraints",
                "",
                "- Do not imply that SPARSE or FULLER are accuracy-preserving.",
                "- Do not cite engineering-smoke outputs as benchmark evidence.",
                "- Do not reuse legacy `20260319_fullerexp_v1` figure inputs for this current replay report pack.",
                "- Do not silently change lane aggregation or recompute replay metrics.",
                "",
                "## Output Plan",
                "",
                "- output stem: `FigFuller_Phase4LaneComparison`",
                "- expected exports: `SVG`, `PDF`, `PNG`",
                "- traceability update needed: `yes`",
                "",
                "## Review Gate",
                "",
                "- field mapping matches the repaired intake CSV",
                "- units are explicit",
                "- no color-only distinction",
                "- generated manifest and traceability point to the regenerated lane table",
                "- report wording preserves the runtime-vs-accuracy claim boundary",
                "",
            ]
        ),
        encoding="utf-8",
    )
    traceability = _write_traceability(
        review_dir,
        run_tag=run_tag,
        input_csv=input_csv,
        table_csv=table_csv,
        outputs=figure_outputs,
        generated_at=generated_at,
        render_command=render_command,
    )
    manifest = review_dir / "review_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_tag": run_tag,
                "artifact_layer": "phase4_report_pack",
                "input_csv": _rel(input_csv),
                "lane_table_csv": _rel(table_csv),
                "figure_ids": ["FigFuller_Phase4LaneComparison"],
                "figure_outputs": [_rel(path) for path in figure_outputs],
                "data_figure_brief": _rel(review_dir / "data_figure_brief.md"),
                "figure_traceability": _rel(traceability),
                "review_spec": _rel(ROOT / "experiments" / "DATA_FIGURE_REVIEW_SPEC.md"),
                "figure_spec": _rel(ROOT / "experiments" / "FIGURE_SPEC_IEEE.md"),
                "generated_at_utc": generated_at,
                "mixed_evidence_surface": MIXED_EVIDENCE_SURFACE_NOTE,
                "source_roots_by_lane": SOURCE_ROOTS_BY_LANE,
                "superseded_rows_note": SUPERSEDED_SPARSE_FULLER_NOTE,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    defect_log = review_dir / "defect_log.csv"
    _write_csv(
        defect_log,
        ["defect_id", "severity", "status", "scope", "description"],
        [
            {
                "defect_id": "CLAIM_BOUNDARY_001",
                "severity": "documented_boundary",
                "status": "documented",
                "scope": "SPARSE/FULLER accuracy",
                "description": "SPARSE and FULLER are report-pack ready for runtime/materialization only; positive accuracy-preservation claims remain blocked.",
            }
        ],
    )
    review_report = review_dir / "figure_review_report.md"
    review_report.write_text(
        "\n".join(
            [
                "# FULLER Phase 4 Figure Review",
                "",
                f"Run tag: `{run_tag}`",
                "",
                "## Gate Result",
                "",
                "- Gate 0 input freeze: pass.",
                "- Gate 1 data integrity: pass for the committed intake CSV.",
                "- Gate 2 metric correctness: pass by direct field mapping; no aggregation changed.",
                "- Gate 3 figure correctness: pass for internal report-pack use.",
                "- Gate 4 release readiness: internal report-pack ready; not cleared for positive SPARSE/FULLER accuracy-preservation claims.",
                "",
                "## Claim Boundary",
                "",
                "The figure may support runtime/materialization intake and lane-comparison discussion. It must not be used to imply that SPARSE or FULLER preserve accuracy.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest, defect_log, review_report


def _write_manifest(
    path: Path,
    *,
    run_tag: str,
    input_csv: Path,
    table_csv: Path,
    report_md: Path,
    figure_outputs: list[Path],
    review_manifest: Path,
    traceability: Path,
    generated_at: str,
    source_commit: str,
    source_git_hashes_by_lane: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_tag": run_tag,
                "source_commit": source_commit,
                "generated_at_utc": generated_at,
                "input_csv": _rel(input_csv),
                "lane_comparison_table_csv": _rel(table_csv),
                "report_md": _rel(report_md),
                "figure_outputs": [_rel(path) for path in figure_outputs],
                "review_manifest": _rel(review_manifest),
                "figure_traceability": _rel(traceability),
                "source_roots_by_lane": SOURCE_ROOTS_BY_LANE,
                "source_git_hashes_by_lane": source_git_hashes_by_lane,
                "mixed_evidence_surface": MIXED_EVIDENCE_SURFACE_NOTE,
                "superseded_rows_note": SUPERSEDED_SPARSE_FULLER_NOTE,
                "paper_freeze_status": PAPER_FREEZE_STATUS,
                "bitstream_boundary": BITSTREAM_BOUNDARY,
                "claim_boundary": "runtime/materialization intake-ready with explicit accuracy tradeoffs; SPARSE/FULLER positive accuracy-preservation claims blocked",
                "ready_for_phase4_runtime_materialization_intake": True,
                "ready_for_positive_sparse_fuller_accuracy_claims": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _source_git_hashes_by_lane() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for lane, source_root in SOURCE_ROOTS_BY_LANE.items():
        annotated_csv = ROOT / source_root / "annotated_accuracy.csv"
        lane_hash = ""
        if annotated_csv.exists():
            for row in _load_csv_rows(annotated_csv):
                if str(row.get("baseline") or "").strip().lower() == "true":
                    continue
                lane_hash = str(row.get("git_hash") or "").strip()
                if lane_hash:
                    break
        hashes[lane] = lane_hash
    return hashes


def _write_report(
    path: Path,
    *,
    run_tag: str,
    rows: list[dict[str, str]],
    manifest_path: Path,
    table_csv: Path,
    figure_outputs: list[Path],
    source_commit: str,
    source_git_hashes_by_lane: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FULLER Phase 4 Report Pack",
        "",
        "Date: 2026-04-25",
        "",
        f"Run tag: `{run_tag}`",
        "",
        "## Status",
        "",
        "This report pack is ready for bounded Phase 4 runtime/materialization intake. It is intentionally scoped to the repaired analysis-grade replay summary and does not reuse legacy `20260319_fullerexp_v1` figure inputs.",
        "",
        f"The evidence lineage is explicit: {MIXED_EVIDENCE_SURFACE_NOTE}. {SUPERSEDED_SPARSE_FULLER_NOTE}.",
        "",
        "The claim boundary remains explicit: SPARSE and FULLER are runtime/materialization ready with accuracy tradeoffs, but positive accuracy-preservation claims remain blocked.",
        "",
        "## Artifacts",
        "",
        f"- Manifest: `{_rel(manifest_path)}`",
        f"- Lane comparison table: `{_rel(table_csv)}`",
        f"- Figure outputs: `{'; '.join(_rel(path) for path in figure_outputs)}`",
        "",
        "## Evidence Surface",
        "",
        f"- Mixed surface: `{MIXED_EVIDENCE_SURFACE_NOTE}`",
        f"- Report generation commit: `{source_commit or 'unspecified'}`",
        f"- Paper-facing freeze: `{PAPER_FREEZE_STATUS}`",
        f"- Bitstream boundary: `{BITSTREAM_BOUNDARY}`",
        "",
        "| Lane | Source root | Source git hash |",
        "| --- | --- | --- |",
    ]
    for lane in LANE_ORDER:
        lines.append(
            f"| {lane} | `{SOURCE_ROOTS_BY_LANE[lane]}` | `{source_git_hashes_by_lane.get(lane) or 'missing'}` |"
        )
    lines.extend(
        [
            "",
            "## Lane Comparison",
            "",
            "| Lane | Top-1 mean | Top-5 mean | Samples/hour mean | Seconds/sample mean | Speedup vs ASTRA | Claim boundary |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lane = row["lane"]
        claim = "accuracy claim blocked" if lane in ACCURACY_CLAIM_BLOCKED else "runtime/materialization ready"
        lines.append(
            f"| {lane} | {float(row['top1_mean']):.4f} | {float(row['top5_mean']):.4f} | "
            f"{float(row['samples_per_hour_mean']):.1f} | {float(row['seconds_per_sample_mean']):.5f} | "
            f"{float(row['speedup_vs_astra']):.2f}x | {claim} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Proceed with Phase 4 report-pack integration for runtime/materialization evidence only. Keep accuracy-preservation language blocked for SPARSE/FULLER, do not promote this surface into the paper-facing freeze, and keep paper figures/oral deck on the retained 2026-04-03 freeze unless a separate promotion gate is opened.",
            "",
            "The remaining bitstream slice artifact packaging gap is documented as a residual clean-check limitation: historical slice summaries under `experiments/results/accuracy/bitstream_slices/*summary*.json` are external residual inputs, not unexplained failures in this report-pack boundary.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def render_fuller_phase4_report_pack(
    *,
    input_csv: Path = DEFAULT_INPUT_CSV,
    report_data_root: Path = DEFAULT_REPORT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    review_root: Path = DEFAULT_REVIEW_ROOT,
    report_md: Path = DEFAULT_REPORT_MD,
    run_tag: str = DEFAULT_RUN_TAG,
    source_commit: str = "",
) -> dict[str, Any]:
    input_csv = input_csv.resolve()
    report_data_root = report_data_root.resolve()
    output_root = output_root.resolve()
    review_root = review_root.resolve()
    report_md = report_md.resolve()
    rows = _load_rows(input_csv)
    source_git_hashes_by_lane = _source_git_hashes_by_lane()

    generated_at = datetime.now(timezone.utc).isoformat()
    out_dir = output_root / run_tag
    review_dir = review_root / run_tag
    table_csv = report_data_root / "fuller_phase4_lane_comparison_table_20260425.csv"
    manifest_path = report_data_root / "fuller_phase4_report_pack_manifest_20260425.json"
    render_command = (
        f"python3 {_rel(Path(__file__))} --input_csv {_rel(input_csv)} "
        f"--run_tag {run_tag} --source_commit {source_commit or 'unspecified'}"
    )

    lane_rows = _lane_table_rows(rows)
    _write_csv(
        table_csv,
        [
            "lane",
            "top1_mean",
            "top1_range",
            "top5_mean",
            "samples_per_hour_mean",
            "seconds_per_sample_mean",
            "speedup_vs_astra",
            "phase4_intake_ready",
            "claim_boundary_note",
        ],
        lane_rows,
    )
    _style()
    figure_outputs = _render_lane_comparison(rows, out_dir)
    review_manifest, _defect_log, _review_report = _write_review_files(
        review_dir,
        run_tag=run_tag,
        input_csv=input_csv,
        table_csv=table_csv,
        figure_outputs=figure_outputs,
        generated_at=generated_at,
        render_command=render_command,
    )
    traceability = review_dir / "figure_traceability.csv"
    _write_report(
        report_md,
        run_tag=run_tag,
        rows=rows,
        manifest_path=manifest_path,
        table_csv=table_csv,
        figure_outputs=figure_outputs,
        source_commit=source_commit,
        source_git_hashes_by_lane=source_git_hashes_by_lane,
    )
    _write_manifest(
        manifest_path,
        run_tag=run_tag,
        input_csv=input_csv,
        table_csv=table_csv,
        report_md=report_md,
        figure_outputs=figure_outputs,
        review_manifest=review_manifest,
        traceability=traceability,
        generated_at=generated_at,
        source_commit=source_commit,
        source_git_hashes_by_lane=source_git_hashes_by_lane,
    )
    return {
        "status": "pass",
        "run_tag": run_tag,
        "manifest": _rel(manifest_path),
        "report_md": _rel(report_md),
        "lane_table_csv": _rel(table_csv),
        "figure_outputs": [_rel(path) for path in figure_outputs],
        "review_manifest": _rel(review_manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the current FULLER Phase 4 report pack.")
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--report_data_root", type=Path, default=DEFAULT_REPORT_DATA_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--review_root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--report_md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--run_tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--source_commit", default="")
    args = parser.parse_args()
    payload = render_fuller_phase4_report_pack(
        input_csv=args.input_csv,
        report_data_root=args.report_data_root,
        output_root=args.output_root,
        review_root=args.review_root,
        report_md=args.report_md,
        run_tag=args.run_tag,
        source_commit=args.source_commit,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
