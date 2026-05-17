#!/usr/bin/env python3
"""Validate the current FULLER Phase 4 report pack."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_TAG = "20260425_fuller_phase4_intake"
DEFAULT_MANIFEST = ROOT / "experiments" / "results" / "report_data" / "fuller_phase4_report_pack_manifest_20260425.json"
LANE_ORDER = ["ASTRA", "MESO", "HOPS", "DET", "SPARSE", "FULLER"]
EXPECTED_TOP1 = {
    "SPARSE": 27.1985,
    "FULLER": 20.6459,
}
EXPECTED_SOURCE_ROOTS = {
    "ASTRA": "experiments/results/report_data/20260423_fuller_analysis_grade_replay/astra",
    "MESO": "experiments/results/report_data/20260423_fuller_analysis_grade_replay/meso",
    "HOPS": "experiments/results/report_data/20260423_fuller_analysis_grade_replay/hops",
    "DET": "experiments/results/report_data/20260423_fuller_analysis_grade_replay/det",
    "SPARSE": "experiments/results/report_data/20260425_sparse_fixed_analysis_grade_replay/sparse",
    "FULLER": "experiments/results/report_data/20260425_sparse_fixed_analysis_grade_replay/fuller",
}


def _resolve(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return ROOT / path


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_fuller_phase4_report_pack(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    if not manifest_path.exists():
        raise SystemExit(f"Missing report-pack manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_tag") != DEFAULT_RUN_TAG:
        raise SystemExit("Report-pack run_tag drifted")
    if manifest.get("ready_for_phase4_runtime_materialization_intake") is not True:
        raise SystemExit("Report pack must be runtime/materialization intake-ready")
    if manifest.get("ready_for_positive_sparse_fuller_accuracy_claims") is not False:
        raise SystemExit("SPARSE/FULLER positive accuracy claims must remain blocked")
    if "mixed evidence surface" not in str(manifest.get("mixed_evidence_surface") or ""):
        raise SystemExit("Manifest must record the mixed evidence surface")
    if manifest.get("source_roots_by_lane") != EXPECTED_SOURCE_ROOTS:
        raise SystemExit("Manifest must record source roots per lane")
    if "not_promoted" not in str(manifest.get("paper_freeze_status") or ""):
        raise SystemExit("Manifest must retain the paper-facing freeze unless explicitly promoted")
    if "limited_linear_attention_pilot" not in str(manifest.get("bitstream_boundary") or ""):
        raise SystemExit("Manifest must record the limited bitstream boundary")

    input_csv = _resolve(manifest["input_csv"])
    table_csv = _resolve(manifest["lane_comparison_table_csv"])
    report_md = _resolve(manifest["report_md"])
    review_manifest = _resolve(manifest["review_manifest"])
    traceability = _resolve(manifest["figure_traceability"])
    figure_outputs = [_resolve(path) for path in manifest["figure_outputs"]]
    for path in [input_csv, table_csv, report_md, review_manifest, traceability, *figure_outputs]:
        if not path.exists():
            raise SystemExit(f"Missing report-pack artifact: {path}")
        if path.stat().st_size <= 0:
            raise SystemExit(f"Empty report-pack artifact: {path}")

    input_rows = _load_csv(input_csv)
    table_rows = _load_csv(table_csv)
    if [row["lane"] for row in input_rows] != LANE_ORDER:
        raise SystemExit("Input lane order drifted")
    if [row["lane"] for row in table_rows] != LANE_ORDER:
        raise SystemExit("Lane table order drifted")
    if any(row["phase4_intake_ready"] != "true" for row in table_rows):
        raise SystemExit("Lane table contains a non-ready lane")
    boundary_by_lane = {row["lane"]: row["claim_boundary_note"] for row in table_rows}
    for lane in ("SPARSE", "FULLER"):
        if boundary_by_lane[lane] != "runtime_materialization_ready_accuracy_claim_blocked":
            raise SystemExit(f"{lane} must remain accuracy-claim blocked")
        actual_top1 = float(next(row["top1_mean"] for row in table_rows if row["lane"] == lane))
        if abs(actual_top1 - EXPECTED_TOP1[lane]) > 0.0001:
            raise SystemExit(f"{lane} Top-1 must come from the sparse-fixed replay")
    if float(table_rows[-1]["speedup_vs_astra"]) <= 10.0:
        raise SystemExit("FULLER speedup must remain above 10x in this report pack")

    review_payload = json.loads(review_manifest.read_text(encoding="utf-8"))
    if review_payload.get("data_figure_brief") != "experiments/results/review/20260425_fuller_phase4_intake/data_figure_brief.md":
        raise SystemExit("Review manifest must point to the data figure brief")
    trace_rows = _load_csv(traceability)
    if len(trace_rows) != 1 or trace_rows[0]["figure_id"] != "FigFuller_Phase4LaneComparison":
        raise SystemExit("Traceability must contain the lane-comparison figure")
    if "composition_only" not in trace_rows[0]["literature_anchor_scope"]:
        raise SystemExit("Traceability must mark literature anchors as composition-only")
    if "regenerated lane table" not in trace_rows[0]["notes"]:
        raise SystemExit("Traceability must point to the regenerated lane table")

    report_text = report_md.read_text(encoding="utf-8")
    for forbidden in ("near `0.10%`", "| SPARSE | 0.1030", "| FULLER | 0.0993"):
        if forbidden in report_text:
            raise SystemExit(f"Report markdown still contains stale SPARSE/FULLER text: {forbidden}")
    for needle in (
        "runtime/materialization intake",
        "mixed evidence surface",
        "near-zero SPARSE/FULLER rows are superseded",
        "positive accuracy-preservation claims remain blocked",
        "does not reuse legacy `20260319_fullerexp_v1` figure inputs",
        "current paper-facing freeze remains",
        "limited_linear_attention_pilot",
    ):
        if needle not in report_text:
            raise SystemExit(f"Report markdown missing required wording: {needle}")
    return {
        "status": "pass",
        "manifest": str(manifest_path),
        "figure_count": len(figure_outputs),
        "lane_count": len(table_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the current FULLER Phase 4 report pack.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    payload = check_fuller_phase4_report_pack(args.manifest)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
