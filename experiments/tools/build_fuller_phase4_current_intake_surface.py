#!/usr/bin/env python3
"""Build the current FULLER phase4 intake/evidence surface."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .fuller_experiment_program_common import (
        CLAIM_TIER_ENGINEERING,
        CLAIM_TIER_SUPPORT,
        DEFAULT_CONTRACT,
        EXPERIMENT_FAMILY_ORDER,
        HOST_TUNING_PROVENANCE_FIELDS,
        ROOT,
        _resolve_path,
        _write_csv,
        _write_json,
        _write_text,
        build_phase4_intake_rows,
        load_program_context,
    )
except ImportError:
    from fuller_experiment_program_common import (  # type: ignore
        CLAIM_TIER_ENGINEERING,
        CLAIM_TIER_SUPPORT,
        DEFAULT_CONTRACT,
        EXPERIMENT_FAMILY_ORDER,
        HOST_TUNING_PROVENANCE_FIELDS,
        ROOT,
        _resolve_path,
        _write_csv,
        _write_json,
        _write_text,
        build_phase4_intake_rows,
        load_program_context,
    )


PHASE4_CURRENT_INTAKE_FIELDS = [
    "experiment_family_id",
    "claim_tier",
    "phase4_eligible",
    "claim_boundary",
    "current_surface_status",
    "evidence_capture_status",
    "current_evidence_surface",
    "intake_gate",
    "next_action",
    "blocking_condition",
    "baseline_dependency",
    *HOST_TUNING_PROVENANCE_FIELDS,
    "support_boundary",
]


def _load_status_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return payload


def _family_surface_path(root_dir: Path, outputs: dict[str, Any], family_id: str) -> str:
    if family_id == "analysis_grade_replay":
        return str(_resolve_path(root_dir, outputs["analysis_grade_replay_status_json"]))
    if family_id == "anchor_validation":
        return str(_resolve_path(root_dir, outputs["runtime_smoke_slice_manifest_512_csv"]))
    if family_id == "lane_isolation_runtime_smoke":
        return str(_resolve_path(root_dir, outputs["runtime_smoke_slice_manifest_512_csv"]))
    if family_id == "realism_calibration_support":
        return str(_resolve_path(root_dir, outputs["phase4_intake_contract_csv"]))
    if family_id == "report_pack":
        return str(_resolve_path(root_dir, outputs["report_contract_csv"]))
    return str(_resolve_path(root_dir, outputs["phase4_intake_contract_csv"]))


def _phase4_current_rows(
    intake_rows: list[dict[str, Any]],
    *,
    root_dir: Path,
    outputs: dict[str, Any],
    status_json: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    intake_by_family = {str(row["experiment_family_id"]): row for row in intake_rows}
    queue_state = str(status_json.get("queue_state") or "").strip()
    current_lane = str(status_json.get("current_lane") or "").strip()
    if queue_state in {
        "analysis_grade_active_astra_canonical_baseline",
        "analysis_grade_active_astra_canonical_baseline_resume_repair",
    } and current_lane == "ASTRA":
        analysis_surface_status = (
            "active_astra_canonical_baseline_resume_repair"
            if queue_state == "analysis_grade_active_astra_canonical_baseline_resume_repair"
            else "active_astra_canonical_baseline"
        )
        analysis_evidence_status = "in_progress_not_phase4_complete"
        analysis_blocker = "ASTRA canonical baseline incomplete"
    elif queue_state in {
        "analysis_grade_paused_resume_ready",
        "analysis_grade_paused_resume_repair_complete",
    } and current_lane == "ASTRA":
        analysis_surface_status = (
            "paused_astra_canonical_baseline_resume_repair_complete"
            if queue_state == "analysis_grade_paused_resume_repair_complete"
            else "paused_astra_canonical_baseline_resume_ready"
        )
        analysis_evidence_status = "paused_not_phase4_complete"
        analysis_blocker = "ASTRA canonical baseline paused at a resume-safe checkpoint"
    else:
        analysis_surface_status = "blocked_pending_astra_canonical_baseline"
        analysis_evidence_status = "not_started_or_status_unavailable"
        analysis_blocker = "ASTRA canonical baseline reference missing"

    for family_id in EXPERIMENT_FAMILY_ORDER:
        intake = intake_by_family[family_id]
        claim_tier = str(intake["claim_tier"])
        phase4_eligible = bool(intake["phase4_eligible"])
        claim_boundary = str(intake["claim_boundary"])
        baseline_dependency = str(intake["baseline_dependency"])
        current_surface = _family_surface_path(root_dir, outputs, family_id)
        current_surface_status = "contract_current"
        evidence_status = "pending_family_gate"
        intake_gate = str(intake["intake_status"])
        next_action = "keep_contract_current"
        blocking_condition = ""
        support_boundary = "not_support_family"

        if claim_tier == CLAIM_TIER_ENGINEERING:
            current_surface_status = "complete_current_engineering_smoke"
            evidence_status = "captured_not_claim_tier"
            intake_gate = "closed_to_claim_tier"
            next_action = "use_for_governance_status_only"
            blocking_condition = "engineering smoke cannot substitute for analysis-grade evidence"
            support_boundary = "engineering_validation_only"
        elif family_id == "analysis_grade_replay":
            current_surface_status = analysis_surface_status
            evidence_status = analysis_evidence_status
            intake_gate = "await_astra_success_then_mainline_quantized_only"
            if analysis_surface_status.startswith("paused_"):
                next_action = "resume ASTRA paired baseline with --resume, then launch MESO/HOPS/DET/SPARSE/FULLER quantized_only replay after success"
            else:
                next_action = "complete ASTRA paired baseline, then launch MESO/HOPS/DET/SPARSE/FULLER quantized_only replay"
            blocking_condition = analysis_blocker
            support_boundary = "claim_tier_mainline_lanes_only"
        elif family_id == "realism_calibration_support":
            current_surface_status = "support_contract_current_not_started"
            evidence_status = "support_family_pending"
            intake_gate = "not_main_claim_tier"
            next_action = "scope PHY realism/calibration audit after mainline analysis-grade admission"
            blocking_condition = "PHY support audit is not required for mainline claim-lane admission"
            support_boundary = "PHY_support_calibration_not_main_claim_tier"
        elif claim_tier == CLAIM_TIER_SUPPORT:
            current_surface_status = "support_contract_current"
            evidence_status = "support_family_pending"
            intake_gate = "support_family_gate_required"
            next_action = "defer until analysis-grade claim surface is anchored"
            blocking_condition = "support family is outside current claim-tier intake"
            support_boundary = "support_only"
        else:
            current_surface_status = "claim_audit_contract_current"
            evidence_status = "holdout_audit_pending"
            intake_gate = "holdout_family_gate_required"
            next_action = "defer until claim surface is available for audit"
            blocking_condition = "holdout audit cannot run before claim surface exists"
            support_boundary = "claim_audit_only"

        rows.append(
            {
                "experiment_family_id": family_id,
                "claim_tier": claim_tier,
                "phase4_eligible": phase4_eligible,
                "claim_boundary": claim_boundary,
                "current_surface_status": current_surface_status,
                "evidence_capture_status": evidence_status,
                "current_evidence_surface": current_surface,
                "intake_gate": intake_gate,
                "next_action": next_action,
                "blocking_condition": blocking_condition,
                "baseline_dependency": baseline_dependency,
                **{
                    field: intake.get(field, "" if field in {"host_profile_id", "calibration_artifact_path"} else {})
                    for field in HOST_TUNING_PROVENANCE_FIELDS
                },
                "support_boundary": support_boundary,
            }
        )
    return rows


def _phase4_current_note(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# FULLER Phase4 Current Intake Surface",
        "",
        "Date: `2026-04-23`",
        "Status: `phase4_current_intake_pending_astra_completion`",
        "",
        "## Decision",
        "",
        "Phase4 has a current intake/evidence surface, but claim-tier evidence capture is not complete until ASTRA finishes the paired canonical baseline and the redesigned mainline quantized-only replay finishes.",
        "`MESO/HOPS/DET/SPARSE/FULLER` remain paper-mainline claim lanes under `analysis_grade_replay`; `PHY` remains support-only under `realism_calibration_support`.",
        "",
        "## Families",
        "",
    ]
    lines.extend(
        f"- `{row['experiment_family_id']}` surface=`{row['current_surface_status']}` evidence=`{row['evidence_capture_status']}` gate=`{row['intake_gate']}`"
        for row in rows
    )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Engineering-smoke outputs are captured as governance/status evidence only.",
            "The only phase4-eligible claim family is `analysis_grade_replay`.",
            "`PHY` cannot substitute for a paper-mainline mechanism lane and remains calibration/support evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_fuller_phase4_current_intake_surface(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
) -> dict[str, Any]:
    ctx = load_program_context(contract_path, root_dir=root_dir)
    outputs = dict(ctx.contract.get("outputs") or {})
    status_json = _load_status_json(_resolve_path(root_dir, outputs["analysis_grade_replay_status_json"]))
    rows = _phase4_current_rows(
        build_phase4_intake_rows(ctx),
        root_dir=root_dir,
        outputs=outputs,
        status_json=status_json,
    )
    csv_path = _resolve_path(root_dir, outputs["phase4_current_intake_surface_csv"])
    json_path = _resolve_path(root_dir, outputs["phase4_current_intake_surface_json"])
    md_path = _resolve_path(root_dir, outputs["phase4_current_intake_surface_md"])
    _write_csv(csv_path, PHASE4_CURRENT_INTAKE_FIELDS, rows)
    _write_json(
        json_path,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "analysis_grade_queue_state": str(status_json.get("queue_state") or ""),
            "rows": rows,
        },
    )
    _write_text(md_path, _phase4_current_note(rows))
    return {
        "status": "pass",
        "row_count": len(rows),
        "phase4_current_intake_surface_csv": str(csv_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FULLER phase4 current intake surface.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    payload = build_fuller_phase4_current_intake_surface(args.contract)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
