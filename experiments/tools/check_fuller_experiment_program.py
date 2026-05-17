#!/usr/bin/env python3
"""Validate the active FULLER experiment program matrix and data contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from .fuller_experiment_program_common import (
        ANALYSIS_GRADE_LANES,
        DATA_CONTRACT_FIELDS,
        DEFAULT_CONTRACT,
        ENGINEERING_SMOKE_LANES,
        EXPERIMENT_FAMILY_ORDER,
        EXPERIMENT_MATRIX_FIELDS,
        LANE_ORDER,
        ROOT,
        _load_json,
        _load_yaml,
        _resolve_path,
        load_program_context,
    )
except ImportError:
    from fuller_experiment_program_common import (  # type: ignore
        ANALYSIS_GRADE_LANES,
        DATA_CONTRACT_FIELDS,
        DEFAULT_CONTRACT,
        ENGINEERING_SMOKE_LANES,
        EXPERIMENT_FAMILY_ORDER,
        EXPERIMENT_MATRIX_FIELDS,
        LANE_ORDER,
        ROOT,
        _load_json,
        _load_yaml,
        _resolve_path,
        load_program_context,
    )


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_json_field(raw: str, field_name: str) -> list[str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {field_name}: {raw!r}") from exc
    if not isinstance(payload, list):
        raise SystemExit(f"Expected JSON list in {field_name}")
    return [str(item) for item in payload]


def check_fuller_experiment_program(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
) -> dict[str, Any]:
    ctx = load_program_context(contract_path, root_dir=root_dir)
    outputs = ctx.contract.get("outputs") or {}
    coordination = ctx.contract.get("coordination") or {}

    experiment_matrix_csv = _resolve_path(root_dir, outputs["experiment_matrix_csv"])
    experiment_matrix_json = _resolve_path(root_dir, outputs["experiment_matrix_json"])
    data_contract_csv = _resolve_path(root_dir, outputs["data_contract_csv"])
    data_contract_json = _resolve_path(root_dir, outputs["data_contract_json"])
    program_manifest_json = _resolve_path(root_dir, outputs["program_manifest_json"])
    program_refactor_note_md = _resolve_path(root_dir, outputs["program_refactor_note_md"])
    phase35_status_md = _resolve_path(root_dir, coordination["phase35_status_md"])

    for path in (
        experiment_matrix_csv,
        experiment_matrix_json,
        data_contract_csv,
        data_contract_json,
        program_manifest_json,
        program_refactor_note_md,
        phase35_status_md,
    ):
        if not path.exists():
            raise SystemExit(f"Missing experiment-program output: {path}")

    experiment_rows = _load_csv_rows(experiment_matrix_csv)
    experiment_json = _load_json(experiment_matrix_json)
    if list(experiment_json.get("rows") or []) == []:
        raise SystemExit("Experiment matrix JSON must expose rows")
    if len(experiment_rows) != len(EXPERIMENT_FAMILY_ORDER):
        raise SystemExit("Experiment matrix must contain exactly one row per declared family")
    if [row["experiment_family_id"] for row in experiment_rows] != EXPERIMENT_FAMILY_ORDER:
        raise SystemExit("Experiment matrix family order drifted")

    for field in EXPERIMENT_MATRIX_FIELDS:
        if field not in (experiment_rows[0].keys() if experiment_rows else []):
            raise SystemExit(f"Experiment matrix is missing field: {field}")

    runtime_smoke_row = next(row for row in experiment_rows if row["experiment_family_id"] == "lane_isolation_runtime_smoke")
    if _parse_json_field(runtime_smoke_row["lane_scope"], "lane_isolation_runtime_smoke.lane_scope") != LANE_ORDER:
        raise SystemExit("lane_isolation_runtime_smoke must expose all 7 public lanes")
    runtime_smoke_budgets = json.loads(runtime_smoke_row["sample_budget"])
    runtime_smoke_pass_modes = json.loads(runtime_smoke_row["pass_mode"])
    runtime_smoke_policies = json.loads(runtime_smoke_row["baseline_cache_policy"])
    if runtime_smoke_budgets["FULLER"] != 256 or any(runtime_smoke_budgets[lane] != 512 for lane in LANE_ORDER if lane != "FULLER"):
        raise SystemExit("lane_isolation_runtime_smoke must expose the 512/256 engineering-smoke budgets")
    if runtime_smoke_pass_modes["ASTRA"] != "paired" or any(runtime_smoke_pass_modes[lane] != "quantized_only" for lane in ENGINEERING_SMOKE_LANES):
        raise SystemExit("lane_isolation_runtime_smoke must expose paired ASTRA + quantized_only engineering-smoke lanes")
    if runtime_smoke_policies["ASTRA"] != "produce" or any(runtime_smoke_policies[lane] != "reuse_required" for lane in ENGINEERING_SMOKE_LANES):
        raise SystemExit("lane_isolation_runtime_smoke baseline-cache policy drifted")
    analysis_grade_row = next(row for row in experiment_rows if row["experiment_family_id"] == "analysis_grade_replay")
    if _parse_json_field(analysis_grade_row["lane_scope"], "analysis_grade_replay.lane_scope") != ANALYSIS_GRADE_LANES:
        raise SystemExit("analysis_grade_replay must scope only the canonical + mainline claim lanes")
    if analysis_grade_row["status"] != "materialized_redesigned_current":
        raise SystemExit("analysis_grade_replay must expose the redesigned current status")
    analysis_budgets = json.loads(analysis_grade_row["sample_budget"])
    analysis_pass_modes = json.loads(analysis_grade_row["pass_mode"])
    analysis_policies = json.loads(analysis_grade_row["baseline_cache_policy"])
    if set(analysis_budgets) != set(ANALYSIS_GRADE_LANES) or any(int(analysis_budgets[lane]) != 45000 for lane in ANALYSIS_GRADE_LANES):
        raise SystemExit("analysis_grade_replay must keep 45k full-manifest budgets for all claim lanes")
    if analysis_pass_modes["ASTRA"] != "paired" or any(analysis_pass_modes[lane] != "quantized_only" for lane in ANALYSIS_GRADE_LANES if lane != "ASTRA"):
        raise SystemExit("analysis_grade_replay must keep ASTRA paired and mainline lanes quantized_only")
    if analysis_policies["ASTRA"] != "produce" or any(analysis_policies[lane] != "reuse_required" for lane in ANALYSIS_GRADE_LANES if lane != "ASTRA"):
        raise SystemExit("analysis_grade_replay baseline-cache policy drifted")
    realism_row = next(row for row in experiment_rows if row["experiment_family_id"] == "realism_calibration_support")
    if _parse_json_field(realism_row["lane_scope"], "realism_calibration_support.lane_scope") != ["PHY"]:
        raise SystemExit("realism_calibration_support must isolate the PHY lane")
    if realism_row["claim_tier"] != "support_family":
        raise SystemExit("realism_calibration_support must remain support_family tier")

    data_rows = _load_csv_rows(data_contract_csv)
    if list((_load_json(data_contract_json)).get("rows") or []) == []:
        raise SystemExit("Data contract JSON must expose rows")
    for field in DATA_CONTRACT_FIELDS:
        if field not in (data_rows[0].keys() if data_rows else []):
            raise SystemExit(f"Data contract is missing field: {field}")
    data_ids = {row["artifact_id"] for row in data_rows}
    for lane_id in LANE_ORDER:
        if f"lane_isolation_runtime_smoke:{lane_id}:module_specific_surface" not in data_ids:
            raise SystemExit(f"lane_isolation_runtime_smoke is missing module_specific_surface for {lane_id}")
    for lane_id in ANALYSIS_GRADE_LANES:
        if f"analysis_grade_replay:{lane_id}:module_specific_surface" not in data_ids:
            raise SystemExit(f"analysis_grade_replay is missing module_specific_surface for {lane_id}")
    if "analysis_grade_replay:PHY:module_specific_surface" in data_ids:
        raise SystemExit("analysis_grade_replay must not expose a standalone PHY lane row")
    if "realism_calibration_support:PHY:module_specific_surface" not in data_ids:
        raise SystemExit("realism_calibration_support must expose the PHY support surface")
    forbidden_noise_rows = [
        row for row in data_rows
        if row["experiment_family_id"] == "noise_robustness" and any(lane in row["artifact_id"] for lane in ["MESO", "HOPS", "DET", "SPARSE", "PHY"])
    ]
    if forbidden_noise_rows:
        raise SystemExit("noise_robustness must stay scoped to declared anchors, not blanket lane rows")

    readme_text = _resolve_path(root_dir, coordination["active_readme_md"]).read_text(encoding="utf-8")
    expected_pointer = str(coordination.get("expected_active_pointer") or "")
    forbidden_pointer = str(coordination.get("forbidden_legacy_pointer") or "")
    if forbidden_pointer and forbidden_pointer in readme_text:
        raise SystemExit("Active coordination README still points at the superseded true_sc plan")
    if expected_pointer not in readme_text:
        raise SystemExit("Active coordination README does not point at the fuller plan")

    plan_text = _resolve_path(root_dir, coordination["active_phase_plan_md"]).read_text(encoding="utf-8")
    if "`Phase3.5 Experiment Program Refactor`: `completed_2026-04-22`" not in plan_text:
        raise SystemExit("Active phase plan does not record Phase3.5 Experiment Program Refactor")

    note_text = program_refactor_note_md.read_text(encoding="utf-8")
    for needle in (
        "full-dataset runtime-smoke queue is frozen as a legacy current-run surface",
        "remaining lanes run quantized-only against deterministic 512/256-sample slices",
        "ASTRA as the canonical paired full-manifest baseline",
        "`realism_calibration_support` holds `PHY` outside the paper's main claim-tier replay family.",
        "`noise/scaling/device/holdout` are explicit support or audit families",
    ):
        if needle not in note_text:
            raise SystemExit(f"Program refactor note is missing required phrase: {needle}")

    manifest = _load_json(program_manifest_json)
    if manifest.get("phase35_status") != "completed":
        raise SystemExit("Program manifest phase35_status must be completed")

    return {
        "status": "pass",
        "experiment_family_ids": EXPERIMENT_FAMILY_ORDER,
        "experiment_matrix_csv": str(experiment_matrix_csv.resolve()),
        "data_contract_csv": str(data_contract_csv.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the active FULLER experiment program.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    payload = check_fuller_experiment_program(args.contract)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
