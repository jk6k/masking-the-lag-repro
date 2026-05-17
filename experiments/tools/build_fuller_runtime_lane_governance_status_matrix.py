#!/usr/bin/env python3
"""Build the current FULLER runtime lane governance/status matrix."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .fuller_experiment_program_common import (
        DEFAULT_CONTRACT,
        ROOT,
        _write_csv,
        _write_json,
        _write_text,
    )
except ImportError:
    from fuller_experiment_program_common import (  # type: ignore
        DEFAULT_CONTRACT,
        ROOT,
        _write_csv,
        _write_json,
        _write_text,
    )


LANE_ORDER = ["ASTRA", "MESO", "HOPS", "DET", "SPARSE", "PHY", "FULLER"]
STATUS_FIELDS = [
    "lane_id",
    "experiment_family_id",
    "tier",
    "runtime_smoke_status",
    "result_surface_status",
    "ready_for_next_lane",
    "top1",
    "top5",
    "analysis_grade_ready",
    "analysis_grade_blockers_json",
    "claim_tier",
    "phase4_eligible",
    "phase4_intake_status",
    "claim_boundary",
    "next_step",
    "annotated_csv",
    "checker_json",
]

DEFAULT_RUNTIME_ROOT = (
    ROOT / "experiments" / "results" / "report_data" / "20260422_fuller_experiment_program_v2_runtime_smoke"
)
DEFAULT_CHECKER_ROOT = (
    ROOT / "experiments" / "results" / "report_data" / "fuller_v2_runtime_smoke_surface_checks_20260423"
)
DEFAULT_PHASE4_CONTRACT = (
    ROOT / "experiments" / "results" / "report_data" / "fuller_phase4_intake_contract_20260422.csv"
)
DEFAULT_MATRIX_CSV = (
    ROOT / "experiments" / "results" / "report_data" / "fuller_runtime_lane_governance_status_matrix_20260423.csv"
)
DEFAULT_MATRIX_JSON = (
    ROOT / "experiments" / "results" / "report_data" / "fuller_runtime_lane_governance_status_matrix_20260423.json"
)
DEFAULT_MATRIX_MD = (
    ROOT / "docs" / "reports" / "20260423_fuller_runtime_lane_governance_status_matrix.md"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_eval_row(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in reversed(rows):
        if str(row.get("baseline") or "").strip().lower() != "true":
            return row
    return rows[-1] if rows else {}


def _phase4_lookup(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    return {str(row["experiment_family_id"]).strip(): row for row in rows}


def _family_for_lane(lane_id: str) -> str:
    if lane_id == "ASTRA":
        return "anchor_validation"
    if lane_id == "PHY":
        return "realism_calibration_support"
    return "lane_isolation_runtime_smoke"


def _tier_for_family(family_id: str) -> str:
    return "runtime_smoke"


def _next_step(
    *,
    lane_id: str,
    family_id: str,
    ready_for_next_lane: bool,
    analysis_grade_ready: bool,
    phase4_eligible: bool,
) -> str:
    if not ready_for_next_lane:
        return "repair_result_surface"
    if lane_id == "ASTRA":
        return "continue_analysis_grade_canonical_baseline"
    if family_id == "realism_calibration_support":
        return "route_to_realism_calibration_support"
    if phase4_eligible and analysis_grade_ready:
        return "phase4_intake_ready"
    return "await_astra_baseline_then_analysis_grade_quantized_only_replay"


def build_runtime_lane_governance_status_matrix(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    checker_root: Path = DEFAULT_CHECKER_ROOT,
    phase4_contract_csv: Path = DEFAULT_PHASE4_CONTRACT,
    output_csv: Path = DEFAULT_MATRIX_CSV,
    output_json: Path = DEFAULT_MATRIX_JSON,
    output_md: Path = DEFAULT_MATRIX_MD,
) -> dict[str, Any]:
    phase4_lookup = _phase4_lookup(phase4_contract_csv)
    rows: list[dict[str, Any]] = []

    for lane_id in LANE_ORDER:
        family_id = _family_for_lane(lane_id)
        family_row = phase4_lookup[family_id]
        lane_dir = runtime_root / lane_id.lower()
        annotated_csv = lane_dir / "annotated_accuracy.csv"
        manifest_json = lane_dir / "progress" / "manifest.json"
        checker_json = checker_root / f"{lane_id.lower()}.json"

        annotated_rows = _read_csv(annotated_csv)
        annotated_row = _resolve_eval_row(annotated_rows)
        manifest = _load_json(manifest_json)
        checker = _load_json(checker_json)

        analysis_grade_blockers = manifest.get("analysis_grade_blockers") or []
        analysis_grade_ready = bool(manifest.get("analysis_grade_ready"))
        ready_for_next_lane = bool(checker.get("ready_for_next_lane"))
        phase4_eligible = str(family_row["phase4_eligible"]).strip().lower() == "true"
        result_surface_status = str(checker.get("result_surface_status") or "")

        rows.append(
            {
                "lane_id": lane_id,
                "experiment_family_id": family_id,
                "tier": _tier_for_family(family_id),
                "runtime_smoke_status": (
                    "complete_clean_pass" if ready_for_next_lane else "repair_required"
                ),
                "result_surface_status": result_surface_status,
                "ready_for_next_lane": str(ready_for_next_lane).lower(),
                "top1": annotated_row.get("top1", ""),
                "top5": annotated_row.get("top5", ""),
                "analysis_grade_ready": str(analysis_grade_ready).lower(),
                "analysis_grade_blockers_json": json.dumps(
                    analysis_grade_blockers,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "claim_tier": family_row["claim_tier"],
                "phase4_eligible": family_row["phase4_eligible"],
                "phase4_intake_status": family_row["intake_status"],
                "claim_boundary": family_row["claim_boundary"],
                "next_step": _next_step(
                    lane_id=lane_id,
                    family_id=family_id,
                    ready_for_next_lane=ready_for_next_lane,
                    analysis_grade_ready=analysis_grade_ready,
                    phase4_eligible=phase4_eligible,
                ),
                "annotated_csv": str(annotated_csv.resolve()),
                "checker_json": str(checker_json.resolve()),
            }
        )

    _write_csv(output_csv, STATUS_FIELDS, rows)
    _write_json(
        output_json,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows": rows,
        },
    )
    _write_text(output_md, _status_note(rows))
    return {
        "status": "pass",
        "row_count": len(rows),
        "status_matrix_csv": str(output_csv.resolve()),
    }


def _status_note(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# FULLER Runtime Lane Governance Status Matrix",
        "",
        "Date: `2026-04-23`",
        "Status: `runtime_smoke_current_outputs_complete`",
        "",
        "## Runtime-Smoke Lanes",
        "",
    ]
    lines.extend(
        f"- `{row['lane_id']}` runtime_smoke=`{row['runtime_smoke_status']}` result_surface=`{row['result_surface_status']}` phase4_eligible=`{row['phase4_eligible']}` next=`{row['next_step']}`"
        for row in rows
    )
    lines.extend(
        [
            "",
        "## Boundary",
        "",
        "All seven runtime-smoke lanes now have complete current result surfaces, but they remain",
        "engineering-validation outputs. They are not phase4-eligible claim-tier evidence and may",
        "not be promoted in place of `analysis_grade_replay`.",
        "`PHY` is now routed to `realism_calibration_support`, while `MESO/HOPS/DET/SPARSE/FULLER`",
        "wait for the redesigned analysis-grade queue behind the active ASTRA canonical baseline.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the FULLER runtime lane governance/status matrix."
    )
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--checker-root", type=Path, default=DEFAULT_CHECKER_ROOT)
    parser.add_argument("--phase4-contract-csv", type=Path, default=DEFAULT_PHASE4_CONTRACT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_MATRIX_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_MATRIX_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MATRIX_MD)
    args = parser.parse_args()
    payload = build_runtime_lane_governance_status_matrix(
        runtime_root=args.runtime_root,
        checker_root=args.checker_root,
        phase4_contract_csv=args.phase4_contract_csv,
        output_csv=args.output_csv,
        output_json=args.output_json,
        output_md=args.output_md,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
