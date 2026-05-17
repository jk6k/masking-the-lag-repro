#!/usr/bin/env python3
"""Build the current FULLER phase3 execution-infrastructure package."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
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
    from .fuller_phase1_registry import default_fuller_phase1_variants
except ImportError:
    from fuller_phase1_registry import default_fuller_phase1_variants  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "configs" / "fuller_phase3_execution_contract_20260422.yaml"

EXECUTION_PACKET_FIELDS = [
    "variant_id",
    "internal_experiment_id",
    "public_module_stack",
    "package_kind",
    "config_stub",
    "generated_config_path",
    "package_status",
    "blocker_class",
    "blocker_reason",
    "blocker_closure_action",
    "runtime_launch_ready",
    "execution_authorized",
    "execution_started",
    "phase4_intake_ready",
    "progress_manifest_json",
    "launch_command_json",
    "required_outputs_json",
    "required_summary_fields_json",
    "required_manifest_fields_json",
    "required_device",
    "long_run_wrapper_json",
    "cpu_fallback_forbidden",
    "archived_row_relabel_forbidden",
    "hops_scheduler_mode",
    "notes",
]

BLOCKER_MATRIX_FIELDS = [
    "variant_id",
    "internal_experiment_id",
    "package_kind",
    "blocker_class",
    "blocker_reason",
    "closure_action",
    "closure_dependency",
    "closure_artifact",
    "governance_dependency",
    "phase3_ready_after_closure",
    "phase4_intake_ready_after_execution",
    "hops_scheduler_mode",
]

LANE_ORDER = ["ASTRA", "MESO", "HOPS", "DET", "SPARSE", "PHY", "FULLER"]
DEFAULT_RUNTIME_SAMPLES = "16"
DEFAULT_PROGRESS_HEARTBEAT_INTERVAL = "15.0"
DEFAULT_STALL_TIMEOUT = "180.0"
DEFAULT_PRELAUNCH_MIN_SAMPLES_PER_HOUR = "30.0"
DEFAULT_PRELAUNCH_MAX_SECONDS_PER_SAMPLE = "120.0"
DEFAULT_PATHOLOGICAL_MIN_SAMPLES_PER_HOUR = "12.0"
DEFAULT_PATHOLOGICAL_MAX_SECONDS_PER_SAMPLE = "300.0"
DEFAULT_PATHOLOGICAL_MAX_ETA_CURRENT_RATE_SECONDS = "86400.0"
DEFAULT_PATHOLOGICAL_MIN_PROCESSED_SAMPLES = "4"
DEFAULT_PATHOLOGICAL_MIN_ELAPSED_SECONDS = "300.0"
DEFAULT_ASTRA_CONTEXT_REPAIR_MAX_EVAL_SAMPLES = "128"
DEFAULT_MODELS = "mobilevit_s"
DEFAULT_ACCURACY_BACKEND = "mlx"
DEFAULT_BITSTREAM_SURFACE_SCOPE = "limited_linear_attention_pilot"
DEFAULT_ASTRA_ANNOTATION_MEASUREMENT_TRUTH_CLASS = "bridge_only_nonbitstream_measured"
DEFAULT_ASTRA_BITSTREAM_MEASUREMENT_TRUTH_CLASS = "bitstream_limited_surface_pilot"
DEFAULT_ASTRA_CONTRACT_NOTE = "limited_surface_runtime_pilot_not_full_model_measured"
DEFAULT_RUNTIME_SMOKE_TARGET_MODULE_KEYS = [
    "layer_4.1.global_rep.0.pre_norm_mha.1.attn_scores",
    "layer_4.1.global_rep.0.pre_norm_mha.1.attn_output",
]
DEFAULT_ANNOTATION_CONTRACT_NOTE = "runtime_smoke_model_level_measured_not_analysis_grade"
DEFAULT_BITSTREAM_MEASUREMENT_TRUTH_CLASS = "bitstream_model_level_measured"
DEFAULT_SUPPORT_ROOT_NAME = "20260422_fuller_phase3_execution_package"


def _resolve_path(root_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return root_dir / path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return payload


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _serialize_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serialize_csv_value(row.get(key)) for key in fieldnames})


def _parse_json_list(raw: str, field_name: str) -> list[str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {field_name}: {raw!r}") from exc
    if not isinstance(payload, list):
        raise SystemExit(f"Expected JSON list in {field_name}")
    return [str(item) for item in payload]


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_variant_lookup(bundle_payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], Path]:
    variants = bundle_payload.get("variants") or []
    if not isinstance(variants, list):
        raise SystemExit("Bundle variants must be a list")
    generated_config_dir = bundle_payload.get("paths", {}).get("generated_config_dir")
    if not generated_config_dir:
        raise SystemExit("Bundle missing paths.generated_config_dir")
    lookup: dict[str, dict[str, Any]] = {}
    for raw_item in variants:
        if not isinstance(raw_item, dict):
            raise SystemExit("Bundle variants must contain mappings")
        variant_id = str(raw_item.get("variant_id") or "").strip().upper()
        internal_experiment_id = str(raw_item.get("internal_experiment_id") or "").strip().upper()
        if not variant_id or not internal_experiment_id:
            raise SystemExit("Bundle variants must define variant_id and internal_experiment_id")
        lookup[variant_id] = {
            "variant_id": variant_id,
            "internal_experiment_id": internal_experiment_id,
            "public_module_stack": list(raw_item.get("public_module_stack") or []),
            "config_stub": str(raw_item.get("config_stub") or "").strip(),
            "switches": dict(raw_item.get("switches") or {}),
            "accuracy_context_run_id": str(raw_item.get("accuracy_context_run_id") or "").strip(),
        }
    return lookup, Path(str(generated_config_dir))


def _load_launch_rows(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = _load_json(path)
    rows = payload.get("manifest_rows")
    if not isinstance(rows, list):
        raise SystemExit("Launch manifest JSON must expose manifest_rows")
    lookup: dict[str, dict[str, Any]] = {}
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            raise SystemExit("Launch manifest rows must be mappings")
        variant_id = str(raw_row.get("variant_id") or "").strip().upper()
        if not variant_id:
            raise SystemExit("Launch manifest row missing variant_id")
        lookup[variant_id] = raw_row
    return payload, lookup


def _load_phase2_gate_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = _load_csv(path)
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        variant_id = str(row.get("variant_id") or "").strip().upper()
        if variant_id:
            lookup[variant_id] = row
    return lookup


def _load_phase2_handoff_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = _load_csv(path)
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        variant_id = str(row.get("variant_id") or "").strip().upper()
        if not variant_id:
            continue
        lookup[variant_id] = {
            "variant_id": variant_id,
            "internal_experiment_id": str(row.get("internal_experiment_id") or "").strip().upper(),
            "config_stub": str(row.get("config_stub") or "").strip(),
            "expected_generated_config_path": str(row.get("expected_generated_config_path") or "").strip(),
            "required_outputs": _parse_json_list(
                str(row.get("required_outputs_json") or ""),
                f"{variant_id}.required_outputs_json",
            ),
            "required_manifest_fields": _parse_json_list(
                str(row.get("required_manifest_fields_json") or ""),
                f"{variant_id}.required_manifest_fields_json",
            ),
            "required_summary_fields": _parse_json_list(
                str(row.get("required_summary_fields_json") or ""),
                f"{variant_id}.required_summary_fields_json",
            ),
            "required_device": str(row.get("required_device") or "").strip(),
            "long_run_wrapper": _parse_json_list(
                str(row.get("long_run_wrapper_json") or ""),
                f"{variant_id}.long_run_wrapper_json",
            ),
            "authorization_required": _to_bool(row.get("authorization_required")),
            "archived_row_relabel_forbidden": _to_bool(row.get("archived_row_relabel_forbidden")),
            "cpu_fallback_forbidden": _to_bool(row.get("cpu_fallback_forbidden")),
            "notes": str(row.get("notes") or "").strip(),
        }
    return lookup


def _candidate_rows_by_experiment(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    lookup: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        experiment_id = str(row.get("internal_experiment_id") or row.get("experiment_id") or "").strip().upper()
        if experiment_id:
            lookup.setdefault(experiment_id, []).append(row)
    return lookup


def _default_variant_cfg_lookup() -> dict[str, dict[str, Any]]:
    return {
        str(item["variant_id"]).upper(): item
        for item in default_fuller_phase1_variants()
    }


def _default_hops_scheduler_mode(variant_id: str, default_variants: dict[str, dict[str, Any]]) -> str:
    variant_cfg = default_variants.get(str(variant_id).upper()) or {}
    return str(
        ((variant_cfg.get("default_module_cfg") or {}).get("flow") or {}).get("scheduler_mode") or ""
    ).strip()


def _preferred_python_bin(launch_payload: dict[str, Any]) -> str:
    rows = launch_payload.get("manifest_rows") or []
    if not isinstance(rows, list):
        return sys.executable
    for row in rows:
        if not isinstance(row, dict):
            continue
        command = row.get("launch_command") or []
        if isinstance(command, list) and command:
            return str(command[0])
    return sys.executable


def _support_root(execution_packet_csv: Path) -> Path:
    return execution_packet_csv.parent / DEFAULT_SUPPORT_ROOT_NAME


def _select_anchor_candidate_row(candidate_rows: list[dict[str, str]]) -> dict[str, str]:
    if not candidate_rows:
        raise SystemExit("ASTRA anchor candidate rows are missing from current runtime-smoke candidate CSV")
    for row in candidate_rows:
        if str(row.get("baseline") or "").strip().lower() == "true":
            return row
    return candidate_rows[0]


def _target_module_keys_from_candidate(row: dict[str, str]) -> list[str]:
    raw = str(row.get("bitstream_target_module_keys_json") or "").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload if str(item).strip()]


def _resolve_astra_target_module_keys(candidate_row: dict[str, str]) -> list[str]:
    candidate_keys = _target_module_keys_from_candidate(candidate_row)
    if candidate_keys:
        return candidate_keys
    return list(DEFAULT_RUNTIME_SMOKE_TARGET_MODULE_KEYS)


def _build_astra_command(
    *,
    python_bin: str,
    run_id: str,
    repair_root: Path,
    target_module_keys: list[str],
    surface_scope: str,
) -> list[str]:
    command = [
        python_bin,
        str(ROOT / "experiments" / "tools" / "run_config_conditioned_accuracy_matrix.py"),
        "--run_ids",
        run_id,
        "--results_csv",
        str(repair_root / "raw_accuracy.csv"),
        "--annotated_results_csv",
        str(repair_root / "annotated_accuracy.csv"),
        "--prepared_phase1_config_root",
        str(repair_root / "prepared_phase1_configs"),
        "--prepared_eligibility_report_root",
        str(repair_root / "prepared_eligibility"),
        "--progress_root",
        str(repair_root / "progress"),
        "--progress_heartbeat_interval_seconds",
        DEFAULT_PROGRESS_HEARTBEAT_INTERVAL,
        "--stall_timeout_seconds",
        DEFAULT_STALL_TIMEOUT,
        "--prelaunch_runtime_smoke_samples",
        DEFAULT_RUNTIME_SAMPLES,
        "--prelaunch_min_samples_per_hour",
        DEFAULT_PRELAUNCH_MIN_SAMPLES_PER_HOUR,
        "--prelaunch_max_seconds_per_sample",
        DEFAULT_PRELAUNCH_MAX_SECONDS_PER_SAMPLE,
        "--pathological_min_samples_per_hour",
        DEFAULT_PATHOLOGICAL_MIN_SAMPLES_PER_HOUR,
        "--pathological_max_seconds_per_sample",
        DEFAULT_PATHOLOGICAL_MAX_SECONDS_PER_SAMPLE,
        "--pathological_max_eta_current_rate_seconds",
        DEFAULT_PATHOLOGICAL_MAX_ETA_CURRENT_RATE_SECONDS,
        "--pathological_min_processed_samples",
        DEFAULT_PATHOLOGICAL_MIN_PROCESSED_SAMPLES,
        "--pathological_min_elapsed_seconds",
        DEFAULT_PATHOLOGICAL_MIN_ELAPSED_SECONDS,
        "--accuracy_backend",
        DEFAULT_ACCURACY_BACKEND,
        "--models",
        DEFAULT_MODELS,
        "--device",
        "mps",
        "--workers",
        "0",
        "--max_eval_samples",
        DEFAULT_ASTRA_CONTEXT_REPAIR_MAX_EVAL_SAMPLES,
        "--seeds",
        "0",
        "--evidence_tier",
        "runtime_smoke",
        "--annotation_measurement_truth_class",
        DEFAULT_ASTRA_ANNOTATION_MEASUREMENT_TRUTH_CLASS,
        "--bitstream_measurement_truth_class",
        DEFAULT_ASTRA_BITSTREAM_MEASUREMENT_TRUTH_CLASS,
        "--annotation_contract_note",
        DEFAULT_ASTRA_CONTRACT_NOTE,
        "--bitstream_contract_note",
        DEFAULT_ASTRA_CONTRACT_NOTE,
        "--enable_bitstream_pilot",
        "--bitstream_surface_scope",
        surface_scope or DEFAULT_BITSTREAM_SURFACE_SCOPE,
        "--dry_run",
    ]
    if target_module_keys:
        command.extend(["--bitstream_target_module_keys", ",".join(target_module_keys)])
    return command


def _materialize_astra_support_files(
    *,
    generated_config_path: Path,
    repair_run_id: str,
    support_root: Path,
    progress_manifest_path: Path,
    context_run_id: str,
    candidate_row: dict[str, str],
    launch_command: list[str],
) -> tuple[Path, Path, Path]:
    prepared_phase1_config_root = support_root / "prepared_phase1_configs"
    prepared_eligibility_root = support_root / "prepared_eligibility"
    progress_root = support_root / "progress"
    run_snapshot_root = ROOT / "experiments" / "results" / "runs" / repair_run_id
    prepared_phase1_config_root.mkdir(parents=True, exist_ok=True)
    prepared_eligibility_root.mkdir(parents=True, exist_ok=True)
    progress_root.mkdir(parents=True, exist_ok=True)
    run_snapshot_root.mkdir(parents=True, exist_ok=True)

    prepared_config_path = prepared_phase1_config_root / generated_config_path.name
    run_snapshot_path = run_snapshot_root / "config_snapshot.yaml"
    if generated_config_path.exists():
        shutil.copyfile(generated_config_path, prepared_config_path)
        shutil.copyfile(generated_config_path, run_snapshot_path)
    else:
        prepared_config_path.write_text("# generated config missing at packet build time\n", encoding="utf-8")
        run_snapshot_path.write_text("# generated config missing at packet build time\n", encoding="utf-8")

    eligibility_path = prepared_eligibility_root / "astra_context_repair_eligibility.json"
    _write_json(
        eligibility_path,
        {
            "variant_id": "ASTRA",
            "internal_experiment_id": "E0",
            "blocker_class": "context_repair",
            "blocker_reason": "context_match_incomplete",
            "closure_action": "materialize_context_match_repair_inputs_then_run_anchor_validation",
            "context_run_id": context_run_id,
            "candidate_run_id": str(candidate_row.get("run_id") or ""),
            "candidate_source_run_id": str(candidate_row.get("source_run_id") or candidate_row.get("run_id") or ""),
            "prepared_config_path": str(prepared_config_path.resolve()),
            "run_snapshot_path": str(run_snapshot_path.resolve()),
        },
    )

    _write_json(
        progress_manifest_path,
        {
            "variant_id": "ASTRA",
            "internal_experiment_id": "E0",
            "package_kind": "anchor_context_repair",
            "package_status": "blocked_context_repair",
            "runtime_launch_ready": False,
            "execution_authorized": False,
            "execution_started": False,
            "phase4_intake_ready": False,
            "context_run_id": context_run_id,
            "candidate_run_id": str(candidate_row.get("run_id") or ""),
            "candidate_source_run_id": str(candidate_row.get("source_run_id") or candidate_row.get("run_id") or ""),
            "launch_command": launch_command,
            "run_snapshot_path": str(run_snapshot_path.resolve()),
            "required_device": "mps",
            "long_run_wrapper": ["caffeinate", "-dimsu"],
            "analysis_grade_enabled": False,
            "notes": "phase3 scaffold only; no execution started",
        },
    )
    return prepared_config_path, eligibility_path, run_snapshot_path


def _build_execution_rows(
    *,
    root_dir: Path,
    contract: dict[str, Any],
    overlay_lookup: dict[str, dict[str, Any]],
    overlay_generated_dir: Path,
    gate_rows: dict[str, dict[str, str]],
    handoff_rows: dict[str, dict[str, Any]],
    launch_payload: dict[str, Any],
    launch_rows: dict[str, dict[str, Any]],
    candidate_rows_by_experiment: dict[str, list[dict[str, str]]],
    execution_packet_csv: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Path]]:
    default_variants = _default_variant_cfg_lookup()
    python_bin = _preferred_python_bin(launch_payload)
    support_root = _support_root(execution_packet_csv)
    support_root.mkdir(parents=True, exist_ok=True)

    execution_rows: list[dict[str, Any]] = []
    blocker_rows: list[dict[str, Any]] = []
    auxiliary_paths: dict[str, Path] = {}

    governance = contract.get("governance") or {}
    required_device = str(governance.get("required_device") or "").strip()
    long_run_wrapper = [str(item) for item in governance.get("long_run_wrapper") or []]
    cpu_fallback_forbidden = bool(governance.get("cpu_fallback_forbidden"))
    archived_row_relabel_forbidden = bool(governance.get("archived_row_relabel_forbidden"))

    for variant_id in LANE_ORDER:
        if variant_id not in overlay_lookup:
            raise SystemExit(f"Runtime-smoke bundle missing variant {variant_id}")
        if variant_id not in gate_rows:
            raise SystemExit(f"Phase2 gate matrix missing variant {variant_id}")
        if variant_id not in handoff_rows:
            raise SystemExit(f"Phase2 handoff contract missing variant {variant_id}")

        bundle_variant = overlay_lookup[variant_id]
        gate_row = gate_rows[variant_id]
        handoff_row = handoff_rows[variant_id]
        generated_config_path = (
            _resolve_path(root_dir, overlay_generated_dir) / f"{bundle_variant['config_stub']}.yaml"
        ).resolve()
        hops_scheduler_mode = _default_hops_scheduler_mode(variant_id, default_variants)

        if variant_id == "ASTRA":
            candidate_rows = candidate_rows_by_experiment.get("E0") or []
            candidate_row = _select_anchor_candidate_row(candidate_rows)
            context_run_id = str(bundle_variant.get("accuracy_context_run_id") or "").strip()
            if not context_run_id:
                context_run_id = str(
                    candidate_row.get("source_run_id")
                    or candidate_row.get("run_id")
                    or ""
                ).strip()
            surface_scope = str(
                candidate_row.get("bitstream_surface_scope") or DEFAULT_BITSTREAM_SURFACE_SCOPE
            ).strip()
            target_module_keys = _resolve_astra_target_module_keys(candidate_row)
            repair_root = support_root / "astra"
            progress_manifest_path = repair_root / "progress" / "manifest.json"
            repair_run_id = f"20260421_fuller_phase1_preflight_{bundle_variant['config_stub']}"
            launch_command = _build_astra_command(
                python_bin=python_bin,
                run_id=repair_run_id,
                repair_root=repair_root,
                target_module_keys=target_module_keys,
                surface_scope=surface_scope,
            )
            _, eligibility_path, run_snapshot_path = _materialize_astra_support_files(
                generated_config_path=generated_config_path,
                repair_run_id=repair_run_id,
                support_root=repair_root,
                progress_manifest_path=progress_manifest_path,
                context_run_id=context_run_id,
                candidate_row=candidate_row,
                launch_command=launch_command,
            )
            auxiliary_paths["astra_progress_manifest"] = progress_manifest_path
            auxiliary_paths["astra_context_repair_eligibility"] = eligibility_path
            auxiliary_paths["astra_run_snapshot"] = run_snapshot_path
            notes = (
                "ASTRA anchor context repair packet only; current candidate input exists but "
                f"context alignment remains open against `{context_run_id}`. No execution started."
            )
            execution_rows.append(
                {
                    "variant_id": variant_id,
                    "internal_experiment_id": "E0",
                    "public_module_stack": bundle_variant["public_module_stack"],
                    "package_kind": "anchor_context_repair",
                    "config_stub": bundle_variant["config_stub"],
                    "generated_config_path": str(generated_config_path),
                    "package_status": "blocked_context_repair",
                    "blocker_class": "context_repair",
                    "blocker_reason": "context_match_incomplete",
                    "blocker_closure_action": "materialize_context_match_repair_inputs_then_run_anchor_validation",
                    "runtime_launch_ready": False,
                    "execution_authorized": False,
                    "execution_started": False,
                    "phase4_intake_ready": False,
                    "progress_manifest_json": str(progress_manifest_path.resolve()),
                    "launch_command_json": launch_command,
                    "required_outputs_json": handoff_row["required_outputs"],
                    "required_summary_fields_json": handoff_row["required_summary_fields"],
                    "required_manifest_fields_json": handoff_row["required_manifest_fields"],
                    "required_device": required_device,
                    "long_run_wrapper_json": long_run_wrapper,
                    "cpu_fallback_forbidden": cpu_fallback_forbidden,
                    "archived_row_relabel_forbidden": archived_row_relabel_forbidden,
                    "hops_scheduler_mode": hops_scheduler_mode,
                    "notes": notes,
                }
            )
            blocker_rows.append(
                {
                    "variant_id": variant_id,
                    "internal_experiment_id": "E0",
                    "package_kind": "anchor_context_repair",
                    "blocker_class": "context_repair",
                    "blocker_reason": "context_match_incomplete",
                    "closure_action": "materialize_context_match_repair_inputs_then_run_anchor_validation",
                    "closure_dependency": "current_anchor_context_alignment",
                    "closure_artifact": str(eligibility_path.resolve()),
                    "governance_dependency": "authorized_anchor_validation_under_mps_only_with_caffeinate",
                    "phase3_ready_after_closure": True,
                    "phase4_intake_ready_after_execution": True,
                    "hops_scheduler_mode": hops_scheduler_mode,
                }
            )
            continue

        launch_row = launch_rows.get(variant_id)
        if launch_row is None:
            raise SystemExit(f"Phase1 runtime-smoke launch manifest missing variant {variant_id}")
        notes = (
            f"{handoff_row['notes']}; imported phase1 dry-run launch package; "
            "phase3 implementation complete but execution remains unauthorized and unstarted."
        )
        execution_rows.append(
            {
                "variant_id": variant_id,
                "internal_experiment_id": handoff_row["internal_experiment_id"],
                "public_module_stack": bundle_variant["public_module_stack"],
                "package_kind": "runtime_smoke_launch",
                "config_stub": bundle_variant["config_stub"],
                "generated_config_path": str(launch_row.get("generated_config") or str(generated_config_path)),
                "package_status": "ready_waiting_authorization",
                "blocker_class": "missing_current_accuracy_row",
                "blocker_reason": "missing_current_accuracy_row",
                "blocker_closure_action": "authorized_runtime_smoke_launch_then_phase4_intake",
                "runtime_launch_ready": bool(launch_row.get("runtime_launch_ready")),
                "execution_authorized": False,
                "execution_started": False,
                "phase4_intake_ready": False,
                "progress_manifest_json": str(launch_row.get("progress_manifest_json") or ""),
                "launch_command_json": list(launch_row.get("launch_command") or []),
                "required_outputs_json": handoff_row["required_outputs"],
                "required_summary_fields_json": handoff_row["required_summary_fields"],
                "required_manifest_fields_json": handoff_row["required_manifest_fields"],
                "required_device": required_device,
                "long_run_wrapper_json": long_run_wrapper,
                "cpu_fallback_forbidden": cpu_fallback_forbidden,
                "archived_row_relabel_forbidden": archived_row_relabel_forbidden,
                "hops_scheduler_mode": hops_scheduler_mode,
                "notes": notes,
            }
        )
        blocker_rows.append(
            {
                "variant_id": variant_id,
                "internal_experiment_id": handoff_row["internal_experiment_id"],
                "package_kind": "runtime_smoke_launch",
                "blocker_class": "missing_current_accuracy_row",
                "blocker_reason": "missing_current_accuracy_row",
                "closure_action": "authorized_runtime_smoke_launch_then_phase4_intake",
                "closure_dependency": "authorized_runtime_smoke_execution",
                "closure_artifact": str(launch_row.get("progress_manifest_json") or ""),
                "governance_dependency": "authorized_runtime_smoke_run_under_mps_only_with_caffeinate",
                "phase3_ready_after_closure": True,
                "phase4_intake_ready_after_execution": True,
                "hops_scheduler_mode": hops_scheduler_mode,
            }
        )

    return execution_rows, blocker_rows, auxiliary_paths


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(cell for cell in row) + " |" for row in rows]
    return "\n".join([header_line, divider, *body])


def _build_implementation_note(execution_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# FULLER Phase3 Implementation Note",
        "",
        "Date: `2026-04-22`",
        "Status: `phase3_implementation_completed_execution_pending`",
        "Scope: `current fuller phase3 execution infrastructure only`",
        "",
        "## Decision",
        "",
        "Phase3 is now implemented as an execution-infrastructure layer for the active fuller public stack.",
        "This phase packages the current lanes into executable packets without starting any runtime-smoke or",
        "long-running evaluation jobs.",
        "",
        "This does not mean:",
        "",
        "- new measured evidence was collected",
        "- `analysis_grade` is enabled",
        "- `benchmark_claim_ready=True`",
        "- `ASTRA` context repair is closed",
        "- `MESO/HOPS/DET/SPARSE/PHY/FULLER` already have current accuracy rows",
        "",
        "## Lane Packages",
        "",
    ]
    for row in execution_rows:
        lines.append(
            "- "
            f"`{row['variant_id']}` (`{row['internal_experiment_id']}`) "
            f"package=`{row['package_kind']}` status=`{row['package_status']}` "
            f"blocker=`{row['blocker_reason']}`"
        )
    lines.extend(
        [
            "",
            "## Governance Invariants",
            "",
            "- `runtime_smoke` remains the only execution tier prepared in the active phase3 surface",
            "- all accelerator-backed runs remain `mps` only",
            "- long runs remain under `caffeinate -dimsu`",
            "- CPU fallback remains forbidden",
            "- archived rows remain read-only historical input",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_execution_readiness_note(execution_rows: list[dict[str, Any]]) -> str:
    table = _render_table(
        ["variant", "kind", "runtime_launch_ready", "blocker", "authorized", "started"],
        [
            [
                f"`{row['variant_id']}`",
                f"`{row['package_kind']}`",
                f"`{str(row['runtime_launch_ready']).lower()}`",
                f"`{row['blocker_reason']}`",
                f"`{str(row['execution_authorized']).lower()}`",
                f"`{str(row['execution_started']).lower()}`",
            ]
            for row in execution_rows
        ],
    )
    lines = [
        "# FULLER Phase3 Execution Readiness Matrix",
        "",
        "Date: `2026-04-22`",
        "Status: `current_phase3_execution_readiness_surface`",
        "",
        "## Lane Table",
        "",
        table,
        "",
        "## Boundary",
        "",
        "This matrix records which phase3 packages are materialized and what still blocks actual execution.",
        "Only `ASTRA` is on the dedicated context-repair path. The other six lanes are packaged as runtime-smoke",
        "launch packets that remain unauthorized and unstarted.",
        "",
        "- `HOPS` remains the public lane name; it is not restated as `FLOW`",
        "- the default HOPS scheduler remains `elastic_residency_v3` on the current packet surface",
        "- phase4 intake is still pending future authorized execution outputs",
        "",
    ]
    return "\n".join(lines)


def _build_blocker_closure_note(blocker_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# FULLER Phase3 Blocker Closure Note",
        "",
        "Date: `2026-04-22`",
        "Status: `current_phase3_blocker_surface`",
        "",
        "## Blocker Classes",
        "",
        "- `ASTRA`: `context_repair` via `current_anchor_context_alignment`",
        "- `MESO/HOPS/DET/SPARSE/PHY/FULLER`: `missing_current_accuracy_row` via authorized runtime-smoke execution",
        "",
        "## Closure Table",
        "",
    ]
    for row in blocker_rows:
        lines.append(
            "- "
            f"`{row['variant_id']}` depends on `{row['closure_dependency']}` "
            f"and closes through `{row['closure_action']}`"
        )
    lines.extend(
        [
            "",
            "## Next Steps",
            "",
            "1. close the `ASTRA` context repair path",
            "2. authorize and run the six runtime-smoke launch packets under `mps` with `caffeinate -dimsu`",
            "3. intake the resulting outputs into phase4 only after the governed artifacts exist on disk",
            "",
        ]
    )
    return "\n".join(lines)


def _build_phase3_status(contract: dict[str, Any], outputs: dict[str, Any]) -> str:
    lines = [
        "# Phase3 Status",
        "",
        "Date: `2026-04-22`",
        "Phase: `Experiment Implementation`",
        "Status: `completed`",
        "",
        "## Delivered Artifacts",
        "",
    ]
    for key in (
        "execution_packet_csv",
        "execution_packet_json",
        "blocker_matrix_csv",
        "blocker_matrix_json",
        "execution_manifest_json",
        "implementation_note_md",
        "execution_readiness_matrix_md",
        "blocker_closure_note_md",
    ):
        lines.append(f"- `{outputs[key]}`")
    lines.extend(
        [
            "",
            "## Exit Criteria Check",
            "",
            "- one current phase3 execution packet for all 7 lanes: `pass`",
            "- one current phase3 blocker matrix with explicit closure semantics: `pass`",
            "- active phase plan marks implementation complete while leaving execution pending: `pass`",
            "- no new measured evidence or analysis-grade enablement claimed here: `pass`",
            "",
            "## Next Phase",
            "",
            "`Phase3 Execution` is now the next active step.",
            "This file does not claim new measured evidence, analysis-grade enablement, or benchmark-readiness promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def build_phase3_execution_package(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
) -> dict[str, Any]:
    resolved_contract_path = contract_path if contract_path.is_absolute() else root_dir / contract_path
    contract = _load_yaml(resolved_contract_path)
    sources = contract.get("sources") or {}
    outputs = contract.get("outputs") or {}

    runtime_bundle = _load_yaml(_resolve_path(root_dir, sources["phase1_runtime_smoke_bundle"]))
    overlay_lookup, overlay_generated_dir = _normalize_variant_lookup(runtime_bundle)
    phase2_contract = _load_yaml(_resolve_path(root_dir, sources["phase2_modeling_contract"]))
    phase2_gate_rows = _load_phase2_gate_rows(_resolve_path(root_dir, sources["phase2_gate_matrix_csv"]))
    phase2_handoff_rows = _load_phase2_handoff_rows(
        _resolve_path(root_dir, sources["phase2_handoff_contract_csv"])
    )
    launch_payload, launch_rows = _load_launch_rows(
        _resolve_path(root_dir, sources["phase1_runtime_smoke_launch_manifest_json"])
    )
    current_candidate_rows = _load_csv(_resolve_path(root_dir, sources["phase1_current_runtime_smoke_candidate_csv"]))
    candidate_rows_by_experiment = _candidate_rows_by_experiment(current_candidate_rows)

    execution_packet_csv = _resolve_path(root_dir, outputs["execution_packet_csv"])
    execution_packet_json = _resolve_path(root_dir, outputs["execution_packet_json"])
    blocker_matrix_csv = _resolve_path(root_dir, outputs["blocker_matrix_csv"])
    blocker_matrix_json = _resolve_path(root_dir, outputs["blocker_matrix_json"])
    execution_manifest_json = _resolve_path(root_dir, outputs["execution_manifest_json"])
    implementation_note_md = _resolve_path(root_dir, outputs["implementation_note_md"])
    execution_readiness_matrix_md = _resolve_path(root_dir, outputs["execution_readiness_matrix_md"])
    blocker_closure_note_md = _resolve_path(root_dir, outputs["blocker_closure_note_md"])
    phase3_status_md = _resolve_path(root_dir, outputs["phase3_status_md"])

    execution_rows, blocker_rows, auxiliary_paths = _build_execution_rows(
        root_dir=root_dir,
        contract=contract,
        overlay_lookup=overlay_lookup,
        overlay_generated_dir=overlay_generated_dir,
        gate_rows=phase2_gate_rows,
        handoff_rows=phase2_handoff_rows,
        launch_payload=launch_payload,
        launch_rows=launch_rows,
        candidate_rows_by_experiment=candidate_rows_by_experiment,
        execution_packet_csv=execution_packet_csv,
    )

    _write_csv(execution_packet_csv, EXECUTION_PACKET_FIELDS, execution_rows)
    _write_json(
        execution_packet_json,
        {
            "contract_path": str(resolved_contract_path.resolve()),
            "rows": execution_rows,
            "row_count": len(execution_rows),
        },
    )
    _write_csv(blocker_matrix_csv, BLOCKER_MATRIX_FIELDS, blocker_rows)
    _write_json(
        blocker_matrix_json,
        {
            "contract_path": str(resolved_contract_path.resolve()),
            "rows": blocker_rows,
            "row_count": len(blocker_rows),
        },
    )

    _write_text(implementation_note_md, _build_implementation_note(execution_rows))
    _write_text(execution_readiness_matrix_md, _build_execution_readiness_note(execution_rows))
    _write_text(blocker_closure_note_md, _build_blocker_closure_note(blocker_rows))
    _write_text(phase3_status_md, _build_phase3_status(phase2_contract, outputs))

    _write_json(
        execution_manifest_json,
        {
            "active_phase_plan_md": str(
                _resolve_path(root_dir, (contract.get("coordination") or {}).get("active_phase_plan_md") or "")
            ),
            "contract_path": str(resolved_contract_path.resolve()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_outputs": {
                "execution_packet_csv": str(execution_packet_csv.resolve()),
                "execution_packet_json": str(execution_packet_json.resolve()),
                "blocker_matrix_csv": str(blocker_matrix_csv.resolve()),
                "blocker_matrix_json": str(blocker_matrix_json.resolve()),
                "execution_manifest_json": str(execution_manifest_json.resolve()),
                "implementation_note_md": str(implementation_note_md.resolve()),
                "execution_readiness_matrix_md": str(execution_readiness_matrix_md.resolve()),
                "blocker_closure_note_md": str(blocker_closure_note_md.resolve()),
                "phase3_status_md": str(phase3_status_md.resolve()),
            },
            "phase3_implementation_status": "completed",
            "execution_started": False,
            "no_execution_started": True,
            "row_counts": {
                "execution_packet": len(execution_rows),
                "blocker_matrix": len(blocker_rows),
            },
            "source_artifacts": {
                key: str(_resolve_path(root_dir, value).resolve())
                for key, value in sources.items()
            },
            "supporting_artifacts": {
                key: str(path.resolve()) for key, path in auxiliary_paths.items()
            },
        },
    )

    return {
        "status": "pass",
        "contract_path": str(resolved_contract_path.resolve()),
        "execution_packet_csv": str(execution_packet_csv.resolve()),
        "blocker_matrix_csv": str(blocker_matrix_csv.resolve()),
        "execution_manifest_json": str(execution_manifest_json.resolve()),
        "variant_ids": [row["variant_id"] for row in execution_rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FULLER phase3 execution-infrastructure package.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    payload = build_phase3_execution_package(args.contract)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
