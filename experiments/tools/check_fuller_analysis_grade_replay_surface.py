#!/usr/bin/env python3
"""Validate FULLER analysis-grade replay materialization and result surfaces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

try:
    from .fuller_experiment_program_common import (
        ANALYSIS_GRADE_LANES,
        _host_tuning_profile_path,
        command_runtime_policy_from_bundle,
        DEFAULT_CONTRACT,
        host_tuning_provenance_fields,
        load_program_context,
        resolve_fuller_accuracy_policy_bundle,
    )
except ImportError:
    from fuller_experiment_program_common import (  # type: ignore
        ANALYSIS_GRADE_LANES,
        _host_tuning_profile_path,
        command_runtime_policy_from_bundle,
        DEFAULT_CONTRACT,
        host_tuning_provenance_fields,
        load_program_context,
        resolve_fuller_accuracy_policy_bundle,
    )

try:
    from .fuller_v2_runtime_smoke_surface import required_result_surface_fields
except ImportError:
    try:
        from fuller_v2_runtime_smoke_surface import required_result_surface_fields  # type: ignore
    except ImportError:

        def required_result_surface_fields(
            *,
            variant_id: str,
            contract_path: Path = DEFAULT_CONTRACT,
        ) -> list[str]:
            contract = _load_yaml(contract_path)
            lanes = {
                str(row.get("variant_id") or "").strip().upper(): row
                for row in contract.get("lanes") or []
                if str(row.get("variant_id") or "").strip()
            }
            lane = lanes.get(str(variant_id).strip().upper()) or {}
            required = [
                str(field).strip()
                for field in lane.get("module_specific_fields") or []
                if str(field).strip()
            ]
            if str(variant_id).strip().upper() == "DET":
                required.extend(
                    [
                        "det_k_global",
                        "det_prefix_error_mean",
                        "det_prefix_error_p95",
                        "det_perturbation",
                    ]
                )
            deduped: list[str] = []
            for field in required:
                if field not in deduped:
                    deduped.append(field)
            return deduped

LANE_ORDER = ANALYSIS_GRADE_LANES
REQUIRED_SEEDS = [0, 1, 2]
REQUIRED_QUANTIZED_ROW_VALUES = {
    "device": "mps",
    "accuracy_backend": "mlx",
    "accuracy_evidence_tier": "analysis_grade",
    "analysis_grade_ready": "true",
}
REQUIRED_QUANTIZED_ROW_NONEMPTY_FIELDS = [
    "analysis_grade_blockers",
    "experiment_id",
    "workload",
    "git_hash",
    "host_profile_id",
    "pass_kind_profile",
    "runtime_policy_fingerprint",
    "semantic_fingerprint",
    "calibration_artifact_path",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
    return payload


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_output(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _parse_json_list(raw: str, field_name: str) -> list[Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {field_name}: {raw!r}") from exc
    if not isinstance(payload, list):
        raise SystemExit(f"Expected JSON list in {field_name}")
    return payload


def _parse_json_object(raw: str, field_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {field_name}: {raw!r}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object in {field_name}")
    return payload


def _ensure_plan_row_by_lane(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        lane_id = str(row.get("lane_id") or "").strip().upper()
        if not lane_id:
            continue
        lookup[lane_id] = row
    return lookup


def _command_flag_value(command: list[Any], flag: str) -> str | None:
    try:
        index = command.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    value = str(command[index + 1] or "").strip()
    if value.startswith("--"):
        return None
    return value or None


def check_materialization_plan(
    *,
    gate_csv: Path,
    plan_csv: Path,
    contract_path: Path = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    ctx = load_program_context(contract_path)
    gate_rows = _load_rows(gate_csv)
    plan_rows = _load_rows(plan_csv)
    gate_by_lane = _ensure_plan_row_by_lane(gate_rows)
    plan_by_lane = _ensure_plan_row_by_lane(plan_rows)
    missing: list[str] = []
    expected_lanes = set(LANE_ORDER)
    host_tuning_profile_json = str(_host_tuning_profile_path(ctx))

    if set(gate_by_lane) != expected_lanes:
        missing.append("gate_lane_scope")
    if set(plan_by_lane) != expected_lanes:
        missing.append("plan_lane_scope")

    canonical_lane = "ASTRA"

    for lane_id in LANE_ORDER:
        gate_row = gate_by_lane.get(lane_id)
        plan_row = plan_by_lane.get(lane_id)
        if gate_row is None:
            missing.append(f"{lane_id}:gate_row_missing")
            continue
        if plan_row is None:
            missing.append(f"{lane_id}:plan_row_missing")
            continue

        gate_status = str(gate_row.get("analysis_grade_gate_status") or "").strip()
        expected_gate_status = (
            "ready_pending_explicit_authorization"
            if lane_id == canonical_lane
            else "ready_pending_astra_baseline_reference"
        )
        expected_pass_mode = "paired" if lane_id == canonical_lane else "quantized_only"
        expected_bundle = resolve_fuller_accuracy_policy_bundle(
            ctx,
            experiment_family_id="analysis_grade_replay",
            lane_id=lane_id,
            pass_mode=expected_pass_mode,
        )
        expected_provenance = host_tuning_provenance_fields(expected_bundle)
        expected_command_policy = command_runtime_policy_from_bundle(
            expected_bundle,
            pass_mode=expected_pass_mode,
        )
        if gate_status != expected_gate_status:
            missing.append(f"{lane_id}:gate_status={gate_status or 'missing'}")
        blockers = _parse_json_list(
            str(gate_row.get("analysis_grade_gate_blockers_json") or "[]"),
            f"{lane_id}.analysis_grade_gate_blockers_json",
        )
        expected_blockers = (
            ["explicit_authorization_required"]
            if lane_id == canonical_lane
            else ["canonical_astra_baseline_reference_required"]
        )
        if blockers != expected_blockers:
            missing.append(f"{lane_id}:gate_blockers")
        if str(gate_row.get("phase4_eligible") or "").strip().lower() != "true":
            missing.append(f"{lane_id}:phase4_eligible")
        if str(gate_row.get("host_profile_id") or "").strip() != expected_provenance["host_profile_id"]:
            missing.append(f"{lane_id}:gate_host_profile_id")
        if (
            _parse_json_object(
                str(gate_row.get("pass_kind_profile") or "{}"),
                f"{lane_id}.gate.pass_kind_profile",
            )
            != expected_provenance["pass_kind_profile"]
        ):
            missing.append(f"{lane_id}:gate_pass_kind_profile")
        if (
            _parse_json_object(
                str(gate_row.get("runtime_policy_fingerprint") or "{}"),
                f"{lane_id}.gate.runtime_policy_fingerprint",
            )
            != expected_provenance["runtime_policy_fingerprint"]
        ):
            missing.append(f"{lane_id}:gate_runtime_policy_fingerprint")
        if (
            _parse_json_object(
                str(gate_row.get("semantic_fingerprint") or "{}"),
                f"{lane_id}.gate.semantic_fingerprint",
            )
            != expected_provenance["semantic_fingerprint"]
        ):
            missing.append(f"{lane_id}:gate_semantic_fingerprint")
        if (
            str(gate_row.get("calibration_artifact_path") or "").strip()
            != expected_provenance["calibration_artifact_path"]
        ):
            missing.append(f"{lane_id}:gate_calibration_artifact_path")

        command = _parse_json_list(
            str(plan_row.get("command_json") or "[]"),
            f"{lane_id}.command_json",
        )
        wrapper = _parse_json_list(
            str(plan_row.get("launch_wrapper_json") or "[]"),
            f"{lane_id}.launch_wrapper_json",
        )
        checker_command = _parse_json_list(
            str(plan_row.get("checker_command_json") or "[]"),
            f"{lane_id}.checker_command_json",
        )
        if wrapper != ["caffeinate", "-dimsu"]:
            missing.append(f"{lane_id}:launch_wrapper")
        if "--evidence_tier" not in command or command[command.index("--evidence_tier") + 1] != "analysis_grade":
            missing.append(f"{lane_id}:evidence_tier")
        if "--seeds" not in command or command[command.index("--seeds") + 1] != "0,1,2":
            missing.append(f"{lane_id}:seeds")
        if "--pass_mode" not in command or command[command.index("--pass_mode") + 1] != expected_pass_mode:
            missing.append(f"{lane_id}:pass_mode")
        if "--device" not in command or command[command.index("--device") + 1] != "mps":
            missing.append(f"{lane_id}:device")
        if "--manifest_override" not in command:
            missing.append(f"{lane_id}:manifest_override")
        if _command_flag_value(command, "--host_tuning_profile_json") != host_tuning_profile_json:
            missing.append(f"{lane_id}:host_tuning_profile_json")
        if _command_flag_value(command, "--host_id") != str(expected_bundle.get("host_id") or ""):
            missing.append(f"{lane_id}:host_id")
        if _command_flag_value(command, "--experiment_family_id") != "analysis_grade_replay":
            missing.append(f"{lane_id}:experiment_family_id")
        if _command_flag_value(command, "--lane_id") != lane_id:
            missing.append(f"{lane_id}:lane_id")
        if _command_flag_value(command, "--workers") != str(int(expected_command_policy.get("workers") or 0)):
            missing.append(f"{lane_id}:workers")
        if _command_flag_value(command, "--eval_batch_size") != str(
            int(expected_command_policy.get("eval_batch_size") or 0)
        ):
            missing.append(f"{lane_id}:eval_batch_size")
        if str(plan_row.get("host_tuning_profile_json") or "").strip() != host_tuning_profile_json:
            missing.append(f"{lane_id}:plan_host_tuning_profile_json")
        if str(plan_row.get("host_id") or "").strip() != str(expected_bundle.get("host_id") or ""):
            missing.append(f"{lane_id}:plan_host_id")
        if str(plan_row.get("host_profile_id") or "").strip() != expected_provenance["host_profile_id"]:
            missing.append(f"{lane_id}:plan_host_profile_id")
        if (
            _parse_json_object(
                str(plan_row.get("pass_kind_profile") or "{}"),
                f"{lane_id}.plan.pass_kind_profile",
            )
            != expected_provenance["pass_kind_profile"]
        ):
            missing.append(f"{lane_id}:plan_pass_kind_profile")
        if (
            _parse_json_object(
                str(plan_row.get("runtime_policy_fingerprint") or "{}"),
                f"{lane_id}.plan.runtime_policy_fingerprint",
            )
            != expected_provenance["runtime_policy_fingerprint"]
        ):
            missing.append(f"{lane_id}:plan_runtime_policy_fingerprint")
        if (
            _parse_json_object(
                str(plan_row.get("semantic_fingerprint") or "{}"),
                f"{lane_id}.plan.semantic_fingerprint",
            )
            != expected_provenance["semantic_fingerprint"]
        ):
            missing.append(f"{lane_id}:plan_semantic_fingerprint")
        if (
            str(plan_row.get("calibration_artifact_path") or "").strip()
            != expected_provenance["calibration_artifact_path"]
        ):
            missing.append(f"{lane_id}:plan_calibration_artifact_path")
        if lane_id == canonical_lane:
            if "--baseline_reference_csv" in command or "--baseline_reference_run_id" in command:
                missing.append(f"{lane_id}:baseline_reuse_not_allowed")
            if str(plan_row.get("baseline_source_step_id") or "").strip():
                missing.append(f"{lane_id}:baseline_source_step_id")
        else:
            if not _command_flag_value(command, "--baseline_reference_csv"):
                missing.append(f"{lane_id}:baseline_reference_csv")
            if not _command_flag_value(command, "--baseline_reference_run_id"):
                missing.append(f"{lane_id}:baseline_reference_run_id")
            if str(plan_row.get("baseline_source_step_id") or "").strip() != "analysis_grade_replay__ASTRA__current":
                missing.append(f"{lane_id}:baseline_source_step_id")
        if "--annotated_results_csv" not in command:
            missing.append(f"{lane_id}:annotated_results_csv")
        if "--max_eval_samples" in command:
            missing.append(f"{lane_id}:max_eval_samples_forbidden")
        if str(plan_row.get("execution_authorized") or "").strip().lower() != "false":
            missing.append(f"{lane_id}:execution_authorized")
        if str(plan_row.get("phase4_eligible") or "").strip().lower() != "true":
            missing.append(f"{lane_id}:plan_phase4_eligible")
        if not checker_command:
            missing.append(f"{lane_id}:checker_command")

    payload = {
        "lane_count": len(plan_by_lane),
        "materialization_status": "complete" if not missing else "incomplete",
        "missing_required_fields_json": json.dumps(missing, ensure_ascii=False, sort_keys=True),
        "ready_for_authorization": not missing,
    }
    return payload


def check_completed_analysis_grade_surface(
    *,
    lane_id: str,
    annotated_csv: Path,
    eval_run_id_prefix: str,
    contract_path: Path = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    rows = _load_rows(annotated_csv) if annotated_csv.exists() else []
    required_fields = required_result_surface_fields(
        variant_id=lane_id,
        contract_path=contract_path,
    )
    missing_fields: list[str] = []
    verified_seeds: list[int] = []

    for seed in REQUIRED_SEEDS:
        eval_run_id = f"{eval_run_id_prefix}{seed}"
        target_row = None
        for row in rows:
            if str(row.get("run_id") or "").strip() != eval_run_id:
                continue
            if str(row.get("baseline") or "").strip().lower() == "true":
                continue
            target_row = row
            break
        if target_row is None:
            missing_fields.append(f"seed{seed}:completed_quantized_row_missing")
            continue
        verified_seeds.append(seed)
        for field, expected in REQUIRED_QUANTIZED_ROW_VALUES.items():
            value = str(target_row.get(field) or "").strip().lower()
            if value != expected:
                missing_fields.append(f"seed{seed}:{field}")
        for field in REQUIRED_QUANTIZED_ROW_NONEMPTY_FIELDS:
            if _is_missing(target_row.get(field)):
                missing_fields.append(f"seed{seed}:{field}")
        for field in required_fields:
            if _is_missing(target_row.get(field)):
                missing_fields.append(f"seed{seed}:{field}")

    payload = {
        "lane_id": lane_id,
        "result_surface_status": "complete" if not missing_fields else "incomplete",
        "missing_required_fields_json": json.dumps(missing_fields, ensure_ascii=False, sort_keys=True),
        "source_row_path": str(annotated_csv.resolve()),
        "seeds_verified_json": json.dumps(verified_seeds, ensure_ascii=False, sort_keys=True),
        "ready_for_phase4_intake": not missing_fields,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check FULLER analysis-grade replay materialization or postrun result surfaces."
    )
    parser.add_argument("--mode", choices=["plan", "postrun"], required=True)
    parser.add_argument("--gate_csv", type=Path, default=None)
    parser.add_argument("--plan_csv", type=Path, default=None)
    parser.add_argument("--lane_id", default=None)
    parser.add_argument("--annotated_csv", type=Path, default=None)
    parser.add_argument("--eval_run_id_prefix", default=None)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "plan":
        if args.gate_csv is None or args.plan_csv is None:
            raise SystemExit("--gate_csv and --plan_csv are required for --mode plan")
        payload = check_materialization_plan(
            gate_csv=args.gate_csv,
            plan_csv=args.plan_csv,
            contract_path=args.contract,
        )
    else:
        if args.annotated_csv is None or not args.lane_id or not args.eval_run_id_prefix:
            raise SystemExit(
                "--lane_id, --annotated_csv, and --eval_run_id_prefix are required for --mode postrun"
            )
        payload = check_completed_analysis_grade_surface(
            lane_id=str(args.lane_id).strip().upper(),
            annotated_csv=args.annotated_csv,
            eval_run_id_prefix=str(args.eval_run_id_prefix),
            contract_path=args.contract,
        )

    _write_output(args.output_json, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    if not payload.get("ready_for_authorization", payload.get("ready_for_phase4_intake", False)):
        raise SystemExit(payload["missing_required_fields_json"])


if __name__ == "__main__":
    main()
