#!/usr/bin/env python3
"""Validate the current FULLER phase3 execution-infrastructure package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from repo_python_bootstrap import maybe_reexec_for_module
except ImportError:
    def maybe_reexec_for_module(_module: str, *, anchor: Path | None = None) -> None:
        return None

maybe_reexec_for_module("yaml", anchor=Path(__file__))

import yaml

try:
    from .build_fuller_phase3_execution_package import (
        BLOCKER_MATRIX_FIELDS,
        DEFAULT_CONTRACT,
        EXECUTION_PACKET_FIELDS,
        LANE_ORDER,
        ROOT,
        _load_json,
        _load_launch_rows,
        _load_phase2_handoff_rows,
        _normalize_variant_lookup,
        _parse_json_list,
        _resolve_path,
        _to_bool,
    )
except ImportError:
    from build_fuller_phase3_execution_package import (  # type: ignore
        BLOCKER_MATRIX_FIELDS,
        DEFAULT_CONTRACT,
        EXECUTION_PACKET_FIELDS,
        LANE_ORDER,
        ROOT,
        _load_json,
        _load_launch_rows,
        _load_phase2_handoff_rows,
        _normalize_variant_lookup,
        _parse_json_list,
        _resolve_path,
        _to_bool,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
    return payload


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_bool(raw: str, field_name: str) -> bool:
    lowered = str(raw or "").strip().lower()
    if lowered not in {"true", "false"}:
        raise SystemExit(f"Expected boolean string in {field_name}, got {raw!r}")
    return lowered == "true"


def check_phase3_execution_package(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
) -> dict[str, Any]:
    resolved_contract_path = contract_path if contract_path.is_absolute() else root_dir / contract_path
    contract = _load_yaml(resolved_contract_path)
    sources = contract.get("sources") or {}
    outputs = contract.get("outputs") or {}
    coordination = contract.get("coordination") or {}
    governance = contract.get("governance") or {}

    runtime_bundle = _load_yaml(_resolve_path(root_dir, sources["phase1_runtime_smoke_bundle"]))
    overlay_lookup, _ = _normalize_variant_lookup(runtime_bundle)
    _, launch_rows = _load_launch_rows(
        _resolve_path(root_dir, sources["phase1_runtime_smoke_launch_manifest_json"])
    )
    handoff_rows = _load_phase2_handoff_rows(_resolve_path(root_dir, sources["phase2_handoff_contract_csv"]))

    execution_packet_csv = _resolve_path(root_dir, outputs["execution_packet_csv"])
    execution_packet_json = _resolve_path(root_dir, outputs["execution_packet_json"])
    blocker_matrix_csv = _resolve_path(root_dir, outputs["blocker_matrix_csv"])
    blocker_matrix_json = _resolve_path(root_dir, outputs["blocker_matrix_json"])
    execution_manifest_json = _resolve_path(root_dir, outputs["execution_manifest_json"])
    implementation_note_md = _resolve_path(root_dir, outputs["implementation_note_md"])
    execution_readiness_matrix_md = _resolve_path(root_dir, outputs["execution_readiness_matrix_md"])
    blocker_closure_note_md = _resolve_path(root_dir, outputs["blocker_closure_note_md"])
    phase3_status_md = _resolve_path(root_dir, outputs["phase3_status_md"])

    for path in (
        execution_packet_csv,
        execution_packet_json,
        blocker_matrix_csv,
        blocker_matrix_json,
        execution_manifest_json,
        implementation_note_md,
        execution_readiness_matrix_md,
        blocker_closure_note_md,
        phase3_status_md,
    ):
        if not path.exists():
            raise SystemExit(f"Missing phase3 output: {path}")

    execution_csv_rows = _load_csv_rows(execution_packet_csv)
    blocker_csv_rows = _load_csv_rows(blocker_matrix_csv)
    execution_json_payload = _load_json(execution_packet_json)
    blocker_json_payload = _load_json(blocker_matrix_json)
    execution_json_rows = execution_json_payload.get("rows")
    blocker_json_rows = blocker_json_payload.get("rows")
    if not isinstance(execution_json_rows, list):
        raise SystemExit("Execution packet JSON must expose rows")
    if not isinstance(blocker_json_rows, list):
        raise SystemExit("Blocker matrix JSON must expose rows")

    if len(execution_csv_rows) != 7 or len(execution_json_rows) != 7:
        raise SystemExit("Execution packet must contain exactly 7 rows")
    if len(blocker_csv_rows) != 7 or len(blocker_json_rows) != 7:
        raise SystemExit("Blocker matrix must contain exactly 7 rows")

    execution_variant_ids = [str(row.get("variant_id") or "").strip().upper() for row in execution_csv_rows]
    blocker_variant_ids = [str(row.get("variant_id") or "").strip().upper() for row in blocker_csv_rows]
    if execution_variant_ids != LANE_ORDER:
        raise SystemExit(f"Execution packet variants drift from lane order: {execution_variant_ids}")
    if blocker_variant_ids != LANE_ORDER:
        raise SystemExit(f"Blocker matrix variants drift from lane order: {blocker_variant_ids}")
    if "FLOW" in execution_variant_ids or "FLOW" in blocker_variant_ids:
        raise SystemExit("HOPS public naming regressed to FLOW")

    execution_by_variant = {str(row["variant_id"]).upper(): row for row in execution_csv_rows}
    blocker_by_variant = {str(row["variant_id"]).upper(): row for row in blocker_csv_rows}

    for variant_id in LANE_ORDER:
        execution_row = execution_by_variant[variant_id]
        blocker_row = blocker_by_variant[variant_id]
        handoff_row = handoff_rows.get(variant_id)
        if handoff_row is None:
            raise SystemExit(f"Phase2 handoff missing row for {variant_id}")

        if variant_id == "ASTRA":
            if str(execution_row.get("package_kind") or "") != "anchor_context_repair":
                raise SystemExit("ASTRA must remain on the anchor_context_repair package path")
            if str(execution_row.get("blocker_class") or "") != "context_repair":
                raise SystemExit("ASTRA blocker_class must remain context_repair")
            if str(execution_row.get("blocker_reason") or "") != "context_match_incomplete":
                raise SystemExit("ASTRA blocker_reason must remain context_match_incomplete")
            if _parse_bool(str(execution_row.get("runtime_launch_ready") or ""), "ASTRA.runtime_launch_ready"):
                raise SystemExit("ASTRA runtime_launch_ready must stay false until context repair closes")
            progress_manifest_path = Path(str(execution_row.get("progress_manifest_json") or ""))
            if not progress_manifest_path.exists():
                raise SystemExit("ASTRA context-repair progress manifest is missing")
            progress_manifest = _load_json(progress_manifest_path)
            if progress_manifest.get("execution_started") is not False:
                raise SystemExit("ASTRA context-repair manifest must record execution_started=false")
            if "materialize_context_match_repair_inputs_then_run_anchor_validation" not in str(
                execution_row.get("blocker_closure_action") or ""
            ):
                raise SystemExit("ASTRA closure action drifted away from context-repair semantics")
        else:
            if str(execution_row.get("package_kind") or "") != "runtime_smoke_launch":
                raise SystemExit(f"{variant_id} must remain a runtime_smoke_launch package")
            if str(execution_row.get("blocker_class") or "") != "missing_current_accuracy_row":
                raise SystemExit(f"{variant_id} blocker_class must remain missing_current_accuracy_row")
            if not _parse_bool(
                str(execution_row.get("runtime_launch_ready") or ""),
                f"{variant_id}.runtime_launch_ready",
            ):
                raise SystemExit(f"{variant_id} runtime_launch_ready must import as true from phase1 launch prep")
            if str(execution_row.get("progress_manifest_json") or "") != str(
                launch_rows[variant_id]["progress_manifest_json"]
            ):
                raise SystemExit(f"{variant_id} progress manifest path drifted from phase1 launch manifest")

        launch_command = _parse_json_list(
            str(execution_row.get("launch_command_json") or ""),
            f"{variant_id}.launch_command_json",
        )
        if not launch_command:
            raise SystemExit(f"{variant_id} launch command must not be empty")

        required_outputs = _parse_json_list(
            str(execution_row.get("required_outputs_json") or ""),
            f"{variant_id}.required_outputs_json",
        )
        if required_outputs != handoff_row["required_outputs"]:
            raise SystemExit(f"{variant_id} required_outputs drifted from phase2 handoff contract")
        required_manifest_fields = _parse_json_list(
            str(execution_row.get("required_manifest_fields_json") or ""),
            f"{variant_id}.required_manifest_fields_json",
        )
        if required_manifest_fields != handoff_row["required_manifest_fields"]:
            raise SystemExit(f"{variant_id} required_manifest_fields drifted from phase2 handoff contract")
        required_summary_fields = _parse_json_list(
            str(execution_row.get("required_summary_fields_json") or ""),
            f"{variant_id}.required_summary_fields_json",
        )
        if required_summary_fields != handoff_row["required_summary_fields"]:
            raise SystemExit(f"{variant_id} required_summary_fields drifted from phase2 handoff contract")

        if str(execution_row.get("required_device") or "") != str(governance.get("required_device") or ""):
            raise SystemExit(f"{variant_id} required_device drifted from contract governance")
        wrapper = _parse_json_list(
            str(execution_row.get("long_run_wrapper_json") or ""),
            f"{variant_id}.long_run_wrapper_json",
        )
        if wrapper != [str(item) for item in governance.get("long_run_wrapper") or []]:
            raise SystemExit(f"{variant_id} long_run_wrapper drifted from contract governance")
        if not _parse_bool(
            str(execution_row.get("cpu_fallback_forbidden") or ""),
            f"{variant_id}.cpu_fallback_forbidden",
        ):
            raise SystemExit(f"{variant_id} must keep cpu_fallback_forbidden=true")
        if not _parse_bool(
            str(execution_row.get("archived_row_relabel_forbidden") or ""),
            f"{variant_id}.archived_row_relabel_forbidden",
        ):
            raise SystemExit(f"{variant_id} must keep archived_row_relabel_forbidden=true")
        if _parse_bool(str(execution_row.get("execution_authorized") or ""), f"{variant_id}.execution_authorized"):
            raise SystemExit(f"{variant_id} execution_authorized must remain false at phase3 implementation time")
        if _parse_bool(str(execution_row.get("execution_started") or ""), f"{variant_id}.execution_started"):
            raise SystemExit(f"{variant_id} execution_started must remain false at phase3 implementation time")

        if str(blocker_row.get("blocker_reason") or "") != str(execution_row.get("blocker_reason") or ""):
            raise SystemExit(f"{variant_id} blocker reason drifted between execution packet and blocker matrix")
        if variant_id == "ASTRA":
            if str(blocker_row.get("closure_dependency") or "") != "current_anchor_context_alignment":
                raise SystemExit("ASTRA closure_dependency must remain current_anchor_context_alignment")
        else:
            if str(blocker_row.get("closure_dependency") or "") != "authorized_runtime_smoke_execution":
                raise SystemExit(f"{variant_id} closure_dependency must remain authorized_runtime_smoke_execution")

        if variant_id in {"HOPS", "FULLER"} and str(execution_row.get("hops_scheduler_mode") or "") != "elastic_residency_v3":
            raise SystemExit(f"{variant_id} must render hops_scheduler_mode=elastic_residency_v3")
        if variant_id == "HOPS" and "FLOW" in str(execution_row.get("notes") or ""):
            raise SystemExit("HOPS notes must not regress to FLOW naming")
        if variant_id == "FULLER":
            for expected_suffix in ("/flow_buffer_trace.csv", "/per_layer_phy.csv", "/phy_budget.json", "/det_prefix_error.csv"):
                if not any(item.endswith(expected_suffix) for item in required_outputs):
                    raise SystemExit(f"FULLER required outputs missing {expected_suffix}")

    readme_text = _read_text(_resolve_path(root_dir, coordination["active_readme_md"]))
    expected_pointer = str(coordination.get("expected_active_pointer") or "")
    forbidden_pointer = str(coordination.get("forbidden_legacy_pointer") or "")
    if forbidden_pointer and forbidden_pointer in readme_text:
        raise SystemExit("Active coordination README still points at the superseded true_sc plan")
    if expected_pointer not in readme_text:
        raise SystemExit("Active coordination README does not point at the fuller plan")

    plan_text = _read_text(_resolve_path(root_dir, coordination["active_phase_plan_md"]))
    if "`Phase3 Experiment Implementation`: `completed_2026-04-22`" not in plan_text:
        raise SystemExit("Active phase plan does not mark Phase3 Experiment Implementation as completed")
    if "`Phase3 Execution`: `pending_authorized_runtime_smoke_runs`" not in plan_text:
        raise SystemExit("Active phase plan does not leave Phase3 Execution pending")
    if "`Phase3 Experiment Implementation / Execution`: `completed" in plan_text:
        raise SystemExit("Active phase plan incorrectly marks phase3 execution as completed")

    execution_manifest = _load_json(execution_manifest_json)
    if execution_manifest.get("phase3_implementation_status") != "completed":
        raise SystemExit("Execution manifest phase3_implementation_status must be completed")
    if execution_manifest.get("execution_started") is not False:
        raise SystemExit("Execution manifest must record execution_started=false")
    if execution_manifest.get("no_execution_started") is not True:
        raise SystemExit("Execution manifest must explicitly record no_execution_started=true")

    implementation_note = _read_text(implementation_note_md)
    for needle in (
        "execution-infrastructure layer",
        "This does not mean:",
        "`analysis_grade` is enabled",
        "`benchmark_claim_ready=True`",
    ):
        if needle not in implementation_note:
            raise SystemExit(f"Implementation note missing required phrase: {needle}")

    readiness_note = _read_text(execution_readiness_matrix_md)
    for needle in (
        "`ASTRA` is on the dedicated context-repair path",
        "`HOPS` remains the public lane name",
        "`elastic_residency_v3`",
    ):
        if needle not in readiness_note:
            raise SystemExit(f"Execution readiness note missing required phrase: {needle}")

    blocker_note = _read_text(blocker_closure_note_md)
    for needle in (
        "`ASTRA`: `context_repair`",
        "`MESO/HOPS/DET/SPARSE/PHY/FULLER`: `missing_current_accuracy_row`",
        "authorized runtime-smoke execution",
    ):
        if needle not in blocker_note:
            raise SystemExit(f"Blocker closure note missing required phrase: {needle}")

    phase3_status_text = _read_text(phase3_status_md)
    if "`Phase3 Execution` is now the next active step." not in phase3_status_text:
        raise SystemExit("PHASE3_STATUS must point to Phase3 Execution as the next active step")

    return {
        "status": "pass",
        "variant_ids": execution_variant_ids,
        "execution_packet_csv": str(execution_packet_csv.resolve()),
        "blocker_matrix_csv": str(blocker_matrix_csv.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the FULLER phase3 execution package.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    payload = check_phase3_execution_package(args.contract)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
