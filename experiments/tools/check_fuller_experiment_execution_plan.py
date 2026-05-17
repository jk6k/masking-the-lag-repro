#!/usr/bin/env python3
"""Validate the current FULLER experiment execution plan."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from .fuller_experiment_program_common import (
        DEFAULT_CONTRACT,
        EXECUTION_PLAN_FIELDS,
        LANE_ORDER,
        ROOT,
        _load_json,
        _resolve_path,
        load_program_context,
    )
except ImportError:
    from fuller_experiment_program_common import (  # type: ignore
        DEFAULT_CONTRACT,
        EXECUTION_PLAN_FIELDS,
        LANE_ORDER,
        ROOT,
        _load_json,
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


def check_fuller_experiment_execution_plan(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
) -> dict[str, Any]:
    ctx = load_program_context(contract_path, root_dir=root_dir)
    outputs = ctx.contract.get("outputs") or {}
    governance = ctx.contract.get("governance") or {}
    execution_plan_csv = _resolve_path(root_dir, outputs["execution_plan_csv"])
    execution_plan_json = _resolve_path(root_dir, outputs["execution_plan_json"])
    execution_plan_md = _resolve_path(root_dir, outputs["execution_plan_md"])
    for path in (execution_plan_csv, execution_plan_json, execution_plan_md):
        if not path.exists():
            raise SystemExit(f"Missing execution-plan output: {path}")
    rows = _load_csv_rows(execution_plan_csv)
    json_rows = _load_json(execution_plan_json).get("rows")
    if not isinstance(json_rows, list):
        raise SystemExit("Execution-plan JSON must expose rows")
    for field in EXECUTION_PLAN_FIELDS:
        if field not in (rows[0].keys() if rows else []):
            raise SystemExit(f"Execution plan is missing field: {field}")
    anchor_rows = [row for row in rows if row["experiment_family_id"] == "anchor_validation"]
    runtime_rows = [row for row in rows if row["experiment_family_id"] == "lane_isolation_runtime_smoke"]
    if len(anchor_rows) != 1 or anchor_rows[0]["lane_id"] != "ASTRA":
        raise SystemExit("anchor_validation must materialize exactly one ASTRA engineering-smoke row")
    if len(runtime_rows) != 6:
        raise SystemExit("lane_isolation_runtime_smoke must materialize exactly 6 non-ASTRA engineering-smoke rows")
    if [row["lane_id"] for row in runtime_rows] != [lane for lane in LANE_ORDER if lane != "ASTRA"]:
        raise SystemExit("lane_isolation_runtime_smoke lane order drifted")
    analysis_grade_rows = [row for row in rows if row["experiment_family_id"] == "analysis_grade_replay"]
    if analysis_grade_rows:
        raise SystemExit("analysis_grade_replay must not materialize unauthorized lane rows by default")
    required_device = str(governance.get("required_device") or "")
    wrapper = [str(item) for item in governance.get("launch_wrapper") or []]
    for row in rows:
        if row["required_device"] != required_device:
            raise SystemExit(f"Execution plan row {row['step_id']} drifted away from required_device={required_device}")
        if _parse_json_field(row["launch_wrapper_json"], f"{row['step_id']}.launch_wrapper_json") != wrapper:
            raise SystemExit(f"Execution plan row {row['step_id']} drifted away from launch wrapper {wrapper}")
        if row["checker_kind"] != "v2_result_surface_contract":
            raise SystemExit(
                f"Execution plan row {row['step_id']} must keep checker_kind=v2_result_surface_contract"
            )
        if "check_fuller_v2_runtime_smoke_result_surface.py" not in row["checker_command_json"]:
            raise SystemExit(
                f"Execution plan row {row['step_id']} must use check_fuller_v2_runtime_smoke_result_surface.py"
            )
        if row["current_run_policy"] != "active_engineering_smoke_v2":
            raise SystemExit(f"Execution plan row {row['step_id']} must keep current_run_policy=active_engineering_smoke_v2")
        if row["stop_on_failure"] != "true":
            raise SystemExit(f"Execution plan row {row['step_id']} must keep stop_on_failure=true")
        if row["resume_enabled"] != "true":
            raise SystemExit(f"Execution plan row {row['step_id']} must keep resume_enabled=true")
    if anchor_rows[0]["pass_mode"] != "paired" or anchor_rows[0]["sample_budget"] != "512":
        raise SystemExit("anchor_validation ASTRA row must stay paired/512")
    if any(row["pass_mode"] != "quantized_only" for row in runtime_rows):
        raise SystemExit("engineering-smoke runtime rows must be quantized_only")
    if any(row["baseline_source_step_id"] != "anchor_validation__ASTRA__engineering_smoke_current" for row in runtime_rows):
        raise SystemExit("engineering-smoke runtime rows must depend on the ASTRA baseline source step")
    note_text = execution_plan_md.read_text(encoding="utf-8")
    for needle in (
        "ASTRA paired `512`-sample baseline-cache producer",
        "quantized-only engineering-smoke rows",
        "analysis-grade replay is now materialized separately as a redesigned current family",
        "PHY is no longer queued inside the main claim-tier analysis-grade family",
    ):
        if needle not in note_text:
            raise SystemExit(f"Execution-plan note is missing required phrase: {needle}")
    queue_status = ctx.phase3_queue_status
    if str(queue_status.get("queue_state") or "") == "":
        raise SystemExit("Legacy queue status must remain readable to the unified execution plan")
    return {
        "status": "pass",
        "row_count": len(rows),
        "runtime_smoke_lane_ids": [row["lane_id"] for row in runtime_rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the FULLER experiment execution plan.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    payload = check_fuller_experiment_execution_plan(args.contract)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
