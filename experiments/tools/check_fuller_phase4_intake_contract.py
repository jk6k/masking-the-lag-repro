#!/usr/bin/env python3
"""Validate the unified FULLER phase4 intake contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from .fuller_experiment_program_common import (
        DEFAULT_CONTRACT,
        EXPERIMENT_FAMILY_ORDER,
        PHASE4_INTAKE_FIELDS,
        ROOT,
        _resolve_path,
        load_program_context,
    )
except ImportError:
    from fuller_experiment_program_common import (  # type: ignore
        DEFAULT_CONTRACT,
        EXPERIMENT_FAMILY_ORDER,
        PHASE4_INTAKE_FIELDS,
        ROOT,
        _resolve_path,
        load_program_context,
    )


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_json_list(raw: str, field_name: str) -> list[str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {field_name}: {raw!r}") from exc
    if not isinstance(payload, list):
        raise SystemExit(f"Expected JSON list in {field_name}")
    return [str(item) for item in payload]


def check_fuller_phase4_intake_contract(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
) -> dict[str, str]:
    ctx = load_program_context(contract_path, root_dir=root_dir)
    outputs = ctx.contract.get("outputs") or {}
    path_csv = _resolve_path(root_dir, outputs["phase4_intake_contract_csv"])
    path_json = _resolve_path(root_dir, outputs["phase4_intake_contract_json"])
    path_md = _resolve_path(root_dir, outputs["phase4_intake_contract_md"])
    for path in (path_csv, path_json, path_md):
        if not path.exists():
            raise SystemExit(f"Missing phase4-intake output: {path}")
    rows = _load_csv_rows(path_csv)
    json_rows = json.loads(path_json.read_text(encoding="utf-8")).get("rows")
    if not isinstance(json_rows, list):
        raise SystemExit("Phase4 intake JSON must expose rows")
    if len(rows) != len(EXPERIMENT_FAMILY_ORDER):
        raise SystemExit("Phase4 intake contract must contain one row per experiment family")
    if [row["experiment_family_id"] for row in rows] != EXPERIMENT_FAMILY_ORDER:
        raise SystemExit("Phase4 intake family order drifted")
    for field in PHASE4_INTAKE_FIELDS:
        if field not in rows[0]:
            raise SystemExit(f"Phase4 intake contract is missing field: {field}")
    row_by_family = {row["experiment_family_id"]: row for row in rows}
    if row_by_family["anchor_validation"]["phase4_eligible"] != "false":
        raise SystemExit("anchor_validation must remain phase4-ineligible")
    if row_by_family["lane_isolation_runtime_smoke"]["phase4_eligible"] != "false":
        raise SystemExit("lane_isolation_runtime_smoke must remain phase4-ineligible")
    if row_by_family["analysis_grade_replay"]["phase4_eligible"] != "true":
        raise SystemExit("analysis_grade_replay must remain the only phase4-eligible lane family")
    if row_by_family["realism_calibration_support"]["phase4_eligible"] != "false":
        raise SystemExit("realism_calibration_support must remain phase4-ineligible")
    if row_by_family["lane_isolation_runtime_smoke"]["claim_tier"] != "engineering_validation_only":
        raise SystemExit("lane_isolation_runtime_smoke must be stamped as engineering_validation_only")
    if row_by_family["analysis_grade_replay"]["claim_tier"] != "analysis_grade":
        raise SystemExit("analysis_grade_replay must be stamped as analysis_grade")
    if row_by_family["realism_calibration_support"]["claim_tier"] != "support_family":
        raise SystemExit("realism_calibration_support must be stamped as support_family")
    if row_by_family["analysis_grade_replay"]["full_dataset_required"] != "true":
        raise SystemExit("analysis_grade_replay must require full-dataset replay")
    if row_by_family["realism_calibration_support"]["full_dataset_required"] != "true":
        raise SystemExit("realism_calibration_support must keep full-manifest audit semantics")
    runtime_smoke_fields = _parse_json_list(
        row_by_family["lane_isolation_runtime_smoke"]["required_summary_fields_json"],
        "lane_isolation_runtime_smoke.required_summary_fields_json",
    )
    for needle in ("hops_scheduler_mode", "det_policy", "sparse_active_fraction", "phy_link_budget_status", "integrated_system_cost_evidence"):
        if needle not in runtime_smoke_fields:
            raise SystemExit(f"lane_isolation_runtime_smoke intake must include {needle}")
    realism_fields = _parse_json_list(
        row_by_family["realism_calibration_support"]["required_summary_fields_json"],
        "realism_calibration_support.required_summary_fields_json",
    )
    for needle in ("phy_link_budget_status", "gaussian_noise_std_ref", "crosstalk_alpha_ref"):
        if needle not in realism_fields:
            raise SystemExit(f"realism_calibration_support intake must include {needle}")
    device_fields = _parse_json_list(row_by_family["device_compare"]["required_summary_fields_json"], "device_compare.required_summary_fields_json")
    for needle in ("host_name", "device_model", "latency_ms", "avg_power_w", "energy_j", "comparison_boundary"):
        if needle not in device_fields:
            raise SystemExit(f"device_compare intake must include {needle}")
    holdout_outputs = _parse_json_list(row_by_family["holdout_audit"]["required_outputs_json"], "holdout_audit.required_outputs_json")
    if "holdout_claim_report.md" not in holdout_outputs or "holdout_claim_summary.csv" not in holdout_outputs:
        raise SystemExit("holdout_audit intake must explicitly require claim report + claim summary")
    holdout_forbidden = _parse_json_list(row_by_family["holdout_audit"]["forbidden_promotions_json"], "holdout_audit.forbidden_promotions_json")
    if "smoke_substitution_for_holdout" not in holdout_forbidden:
        raise SystemExit("holdout_audit intake must forbid smoke substitution")
    note_text = path_md.read_text(encoding="utf-8")
    for needle in (
        "family-driven, not lane-generic",
        "Engineering smoke families remain explicitly",
        "analysis-grade replay remains the only lane family",
        "PHY is now tracked under `realism_calibration_support`",
    ):
        if needle not in note_text:
            raise SystemExit(f"Phase4 intake note must contain: {needle}")
    return {"status": "pass", "phase4_intake_contract_csv": str(path_csv.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the FULLER phase4 intake contract.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    payload = check_fuller_phase4_intake_contract(args.contract)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
