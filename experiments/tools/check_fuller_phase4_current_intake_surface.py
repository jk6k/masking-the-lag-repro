#!/usr/bin/env python3
"""Validate the current FULLER phase4 intake/evidence surface."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from .build_fuller_phase4_current_intake_surface import PHASE4_CURRENT_INTAKE_FIELDS
    from .fuller_experiment_program_common import (
        CLAIM_TIER_ANALYSIS,
        CLAIM_TIER_ENGINEERING,
        CLAIM_TIER_SUPPORT,
        DEFAULT_CONTRACT,
        EXPERIMENT_FAMILY_ORDER,
        HOST_TUNING_PROVENANCE_FIELDS,
        ROOT,
        _resolve_path,
        build_phase4_intake_rows,
        load_program_context,
    )
except ImportError:
    from build_fuller_phase4_current_intake_surface import PHASE4_CURRENT_INTAKE_FIELDS  # type: ignore
    from fuller_experiment_program_common import (  # type: ignore
        CLAIM_TIER_ANALYSIS,
        CLAIM_TIER_ENGINEERING,
        CLAIM_TIER_SUPPORT,
        DEFAULT_CONTRACT,
        EXPERIMENT_FAMILY_ORDER,
        HOST_TUNING_PROVENANCE_FIELDS,
        ROOT,
        _resolve_path,
        build_phase4_intake_rows,
        load_program_context,
    )


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parsed_provenance_value(row: dict[str, str], field: str) -> object:
    raw = str(row.get(field) or "").strip()
    if field in {"host_profile_id", "calibration_artifact_path"}:
        return raw
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object in phase4 current intake field: {field}")
    return payload


def check_fuller_phase4_current_intake_surface(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
) -> dict[str, str]:
    ctx = load_program_context(contract_path, root_dir=root_dir)
    outputs = ctx.contract.get("outputs") or {}
    csv_path = _resolve_path(root_dir, outputs["phase4_current_intake_surface_csv"])
    json_path = _resolve_path(root_dir, outputs["phase4_current_intake_surface_json"])
    md_path = _resolve_path(root_dir, outputs["phase4_current_intake_surface_md"])
    for path in (csv_path, json_path, md_path):
        if not path.exists():
            raise SystemExit(f"Missing phase4 current intake output: {path}")

    rows = _load_csv_rows(csv_path)
    json_rows = json.loads(json_path.read_text(encoding="utf-8")).get("rows")
    if not isinstance(json_rows, list):
        raise SystemExit("Phase4 current intake JSON must expose rows")
    if len(rows) != len(EXPERIMENT_FAMILY_ORDER):
        raise SystemExit("Phase4 current intake surface must contain one row per family")
    if [row["experiment_family_id"] for row in rows] != EXPERIMENT_FAMILY_ORDER:
        raise SystemExit("Phase4 current intake family order drifted")
    for field in PHASE4_CURRENT_INTAKE_FIELDS:
        if field not in rows[0]:
            raise SystemExit(f"Phase4 current intake surface is missing field: {field}")

    by_family = {row["experiment_family_id"]: row for row in rows}
    intake_by_family = {
        str(row["experiment_family_id"]): row for row in build_phase4_intake_rows(ctx)
    }
    if by_family["anchor_validation"]["claim_tier"] != CLAIM_TIER_ENGINEERING:
        raise SystemExit("anchor_validation must remain engineering-only")
    if by_family["lane_isolation_runtime_smoke"]["evidence_capture_status"] != "captured_not_claim_tier":
        raise SystemExit("lane_isolation_runtime_smoke must be captured only as non-claim evidence")
    if by_family["analysis_grade_replay"]["claim_tier"] != CLAIM_TIER_ANALYSIS:
        raise SystemExit("analysis_grade_replay must remain the claim-tier family")
    if by_family["analysis_grade_replay"]["phase4_eligible"] != "true":
        raise SystemExit("analysis_grade_replay must remain phase4 eligible")
    if "quantized_only" not in by_family["analysis_grade_replay"]["intake_gate"]:
        raise SystemExit("analysis_grade_replay intake gate must name the quantized-only downstream model")
    if by_family["analysis_grade_replay"]["evidence_capture_status"] == "complete":
        raise SystemExit("analysis_grade_replay cannot be complete while only ASTRA is active")
    if by_family["realism_calibration_support"]["claim_tier"] != CLAIM_TIER_SUPPORT:
        raise SystemExit("realism_calibration_support must remain support-tier")
    if by_family["realism_calibration_support"]["phase4_eligible"] != "false":
        raise SystemExit("realism_calibration_support must remain phase4-ineligible")
    if "PHY_support_calibration" not in by_family["realism_calibration_support"]["support_boundary"]:
        raise SystemExit("realism_calibration_support must explicitly carry PHY support boundary")
    for family_id, row in by_family.items():
        intake_row = intake_by_family[family_id]
        for field in HOST_TUNING_PROVENANCE_FIELDS:
            expected = intake_row[field]
            actual = _parsed_provenance_value(row, field)
            if actual != expected:
                raise SystemExit(f"{family_id} must preserve phase4 host-tuning provenance for {field}")
    analysis_provenance = _parsed_provenance_value(
        by_family["analysis_grade_replay"],
        "pass_kind_profile",
    )
    if "ASTRA" not in analysis_provenance or "MESO" not in analysis_provenance:
        raise SystemExit("analysis_grade_replay must surface per-lane pass_kind_profile provenance")
    if not str(by_family["analysis_grade_replay"]["calibration_artifact_path"] or "").strip():
        raise SystemExit("analysis_grade_replay must surface calibration_artifact_path provenance")
    for family_id in ("noise_robustness", "scaling_support", "device_compare"):
        if by_family[family_id]["claim_tier"] != CLAIM_TIER_SUPPORT:
            raise SystemExit(f"{family_id} must remain a support family")
    note = md_path.read_text(encoding="utf-8")
    for needle in (
        "claim-tier evidence capture is not complete until ASTRA finishes",
        "`MESO/HOPS/DET/SPARSE/FULLER` remain paper-mainline claim lanes",
        "`PHY` remains support-only",
        "Engineering-smoke outputs are captured as governance/status evidence only",
    ):
        if needle not in note:
            raise SystemExit(f"Phase4 current intake note must contain: {needle}")
    return {"status": "pass", "phase4_current_intake_surface_csv": str(csv_path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the FULLER phase4 current intake surface.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    payload = check_fuller_phase4_current_intake_surface(args.contract)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
