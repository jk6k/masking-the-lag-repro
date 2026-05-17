"""Shared loaders, helpers, and row builders for the active FULLER experiment program."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
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
    from .fuller_host_tuning import (
        PASS_KIND_BASELINE_EVAL,
        PASS_KIND_QUANTIZED_EVAL,
        load_host_tuning_profile,
        resolve_accuracy_policy_bundle,
        resolve_pass_policy_from_bundle,
    )
except ImportError:
    from fuller_host_tuning import (  # type: ignore
        PASS_KIND_BASELINE_EVAL,
        PASS_KIND_QUANTIZED_EVAL,
        load_host_tuning_profile,
        resolve_accuracy_policy_bundle,
        resolve_pass_policy_from_bundle,
    )

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "configs" / "fuller_experiment_program_contract_20260422.yaml"

LANE_ORDER = ["ASTRA", "MESO", "HOPS", "DET", "SPARSE", "PHY", "FULLER"]
ENGINEERING_SMOKE_LANES = ["MESO", "HOPS", "DET", "SPARSE", "PHY", "FULLER"]
ANALYSIS_GRADE_LANES = ["ASTRA", "MESO", "HOPS", "DET", "SPARSE", "FULLER"]
ANALYSIS_GRADE_MAINLINE_LANES = ["MESO", "HOPS", "DET", "SPARSE", "FULLER"]
REALISM_CALIBRATION_LANES = ["PHY"]
EXPERIMENT_FAMILY_ORDER = [
    "anchor_validation",
    "lane_isolation_runtime_smoke",
    "analysis_grade_replay",
    "realism_calibration_support",
    "noise_robustness",
    "scaling_support",
    "device_compare",
    "holdout_audit",
    "report_pack",
]
REPORT_DELIVERABLE_ORDER = [
    "runtime_lane_governance_status_matrix",
    "lane_comparison_table",
    "noise_robustness_figure_table",
    "scaling_figure_table",
    "device_comparison_figure_table",
    "holdout_claim_report",
    "integrated_fuller_evidence_summary",
]

PASS_MODE_PAIRED = "paired"
PASS_MODE_BASELINE_ONLY = "baseline_only"
PASS_MODE_QUANTIZED_ONLY = "quantized_only"
CLAIM_TIER_ENGINEERING = "engineering_validation_only"
CLAIM_TIER_ANALYSIS = "analysis_grade"
CLAIM_TIER_SUPPORT = "support_family"
CLAIM_TIER_HOLDOUT = "holdout_claim"
LEGACY_REPLACEMENT_POLICY = "archive_and_replace"
HOST_TUNING_PROVENANCE_FIELDS = [
    "host_profile_id",
    "pass_kind_profile",
    "runtime_policy_fingerprint",
    "semantic_fingerprint",
    "calibration_artifact_path",
]

EXPERIMENT_MATRIX_FIELDS = [
    "experiment_family_id",
    "research_question_id",
    "tier",
    "lane_scope",
    "comparison_role",
    "parameter_axes_json",
    "required_artifacts_json",
    "required_metrics_json",
    "device_policy",
    "execution_policy",
    "stop_rules_json",
    "report_targets_json",
    "sample_budget",
    "slice_manifest_path",
    "pass_mode",
    "baseline_cache_policy",
    "claim_tier",
    "phase4_eligible",
    "legacy_replacement_policy",
    "status",
]

DATA_CONTRACT_FIELDS = [
    "artifact_id",
    "experiment_family_id",
    "artifact_kind",
    "artifact_grain",
    "producer_surface",
    "required_fields_json",
    "phase4_gate",
    "release_scope",
    "notes",
]

EXECUTION_PLAN_FIELDS = [
    "step_id",
    "experiment_family_id",
    "tier",
    "package_kind",
    "lane_id",
    "command_json",
    "checker_kind",
    "checker_command_json",
    "progress_root",
    "required_device",
    "launch_wrapper_json",
    "serial_group",
    "resume_enabled",
    "stop_on_failure",
    "current_run_policy",
    "sample_budget",
    "slice_manifest_path",
    "pass_mode",
    "baseline_source_step_id",
    "baseline_reference_csv",
    "baseline_reference_summary_json",
    "phase4_eligible",
    "legacy_replacement_policy",
]

PHASE4_INTAKE_FIELDS = [
    "experiment_family_id",
    "required_outputs_json",
    "required_manifest_fields_json",
    "required_summary_fields_json",
    "required_provenance_inputs_json",
    *HOST_TUNING_PROVENANCE_FIELDS,
    "claim_boundary",
    "forbidden_promotions_json",
    "claim_tier",
    "phase4_eligible",
    "baseline_dependency",
    "full_dataset_required",
    "intake_status",
]

REPORT_CONTRACT_FIELDS = [
    "deliverable_id",
    "deliverable_kind",
    "source_family_ids_json",
    "required_artifacts_json",
    "required_metrics_json",
    "readiness_gate",
    "release_scope",
    "status",
]

GENERIC_MANIFEST_FIELDS = [
    "run_id",
    "experiment_id",
    "analysis_grade_ready",
    "analysis_grade_blockers",
    "progress_root",
    "results_csv",
    "annotated_results_csv",
    "launch_command",
    "runtime_guardrail",
]
GENERIC_MASTER_FIELDS = ["latency_ms", "energy_j", "avg_power_w"]
GENERIC_PER_LAYER_FIELDS = ["layer_id", "latency_ms", "energy_j", "acc_top1_delta_pp"]
GENERIC_TIMELINE_FIELDS = ["stage_cycles", "bubble_cycles", "utilization_avg"]
GENERIC_PAIRED_COMPARISON_FIELDS = [
    "top1",
    "top1_delta",
    "top5",
    "top5_delta",
    "baseline_reference_run_id",
    "baseline_reference_csv",
]


@dataclass
class ProgramContext:
    contract_path: Path
    contract: dict[str, Any]
    host_tuning_profile_path: Path
    host_tuning_profile: dict[str, Any]
    runtime_bundle_lookup: dict[str, dict[str, Any]]
    phase2_gate_rows: dict[str, dict[str, str]]
    phase2_handoff_rows: dict[str, dict[str, Any]]
    phase3_execution_rows: dict[str, dict[str, Any]]
    phase3_blocker_rows: dict[str, dict[str, str]]
    phase3_efficiency_rows: dict[str, dict[str, str]]
    phase3_queue_rows: list[dict[str, str]]
    phase3_queue_status: dict[str, Any]
    legacy_design_contract: dict[str, Any]
    legacy_execution_bundle: dict[str, Any]


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


def _serialize_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _serialize_csv_value(row.get(field)) for field in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parse_json_list(raw: str, field_name: str) -> list[str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {field_name}: {raw!r}") from exc
    if not isinstance(payload, list):
        raise SystemExit(f"Expected JSON list in {field_name}")
    return [str(item) for item in payload]


def _ensure_command_list(raw: Any, field_name: str) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, str) and raw.strip():
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON command in {field_name}: {raw!r}") from exc
        if not isinstance(payload, list):
            raise SystemExit(f"Expected JSON list for {field_name}")
        return [str(item) for item in payload]
    raise SystemExit(f"Missing command list for {field_name}")


def _command_flag_value(command: list[str], flag: str) -> str | None:
    indexes = [index for index, token in enumerate(command) if token == flag]
    for index in reversed(indexes):
        if index + 1 < len(command):
            candidate = command[index + 1]
            if not candidate.startswith("--"):
                return candidate
    return None


def _normalize_variant_lookup(bundle_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    variants = bundle_payload.get("variants") or []
    if not isinstance(variants, list):
        raise SystemExit("Runtime-smoke bundle variants must be a list")
    lookup: dict[str, dict[str, Any]] = {}
    for raw_item in variants:
        if not isinstance(raw_item, dict):
            raise SystemExit("Runtime-smoke bundle variants must contain mappings")
        variant_id = str(raw_item.get("variant_id") or "").strip().upper()
        if not variant_id:
            raise SystemExit("Bundle variant is missing variant_id")
        lookup[variant_id] = {
            "variant_id": variant_id,
            "internal_experiment_id": str(raw_item.get("internal_experiment_id") or "").strip().upper(),
            "public_module_stack": [str(item) for item in raw_item.get("public_module_stack") or []],
            "config_stub": str(raw_item.get("config_stub") or "").strip(),
            "switches": dict(raw_item.get("switches") or {}),
            "accuracy_context_run_id": str(raw_item.get("accuracy_context_run_id") or "").strip(),
        }
    return lookup


def _load_phase2_handoff_rows(path: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in _load_csv(path):
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
        }
    return lookup


def _rows_by_variant(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        variant_id = str(row.get("variant_id") or "").strip().upper()
        if variant_id:
            lookup[variant_id] = row
    return lookup


def load_program_context(contract_path: Path = DEFAULT_CONTRACT, *, root_dir: Path = ROOT) -> ProgramContext:
    resolved_contract_path = contract_path if contract_path.is_absolute() else root_dir / contract_path
    contract = _load_yaml(resolved_contract_path)
    sources = contract.get("sources") or {}
    host_tuning_profile_path = _resolve_path(root_dir, sources["host_tuning_profile_json"])
    host_tuning_profile = load_host_tuning_profile(host_tuning_profile_path)
    runtime_bundle_lookup = _normalize_variant_lookup(
        _load_yaml(_resolve_path(root_dir, sources["phase1_runtime_smoke_bundle"]))
    )
    phase2_gate_rows = _rows_by_variant(_load_csv(_resolve_path(root_dir, sources["phase2_gate_matrix_csv"])))
    phase2_handoff_rows = _load_phase2_handoff_rows(_resolve_path(root_dir, sources["phase2_handoff_contract_csv"]))
    phase3_execution_rows = _rows_by_variant(_load_csv(_resolve_path(root_dir, sources["phase3_execution_packet_csv"])))
    phase3_blocker_rows = _rows_by_variant(_load_csv(_resolve_path(root_dir, sources["phase3_blocker_matrix_csv"])))
    phase3_efficiency_rows = _rows_by_variant(_load_csv(_resolve_path(root_dir, sources["phase3_efficiency_audit_csv"])))
    phase3_queue_rows = _load_csv(_resolve_path(root_dir, sources["phase3_serial_launch_queue_csv"]))
    phase3_queue_status = _load_json(_resolve_path(root_dir, sources["phase3_serial_launch_status_json"]))
    legacy_design_contract = _load_yaml(_resolve_path(root_dir, sources["legacy_implementation_design_contract"]))
    legacy_execution_bundle = _load_yaml(_resolve_path(root_dir, sources["legacy_implementation_execution_bundle"]))
    return ProgramContext(
        contract_path=resolved_contract_path,
        contract=contract,
        host_tuning_profile_path=host_tuning_profile_path,
        host_tuning_profile=host_tuning_profile,
        runtime_bundle_lookup=runtime_bundle_lookup,
        phase2_gate_rows=phase2_gate_rows,
        phase2_handoff_rows=phase2_handoff_rows,
        phase3_execution_rows=phase3_execution_rows,
        phase3_blocker_rows=phase3_blocker_rows,
        phase3_efficiency_rows=phase3_efficiency_rows,
        phase3_queue_rows=phase3_queue_rows,
        phase3_queue_status=phase3_queue_status,
        legacy_design_contract=legacy_design_contract,
        legacy_execution_bundle=legacy_execution_bundle,
    )


def _family_lookup(ctx: ProgramContext) -> dict[str, dict[str, Any]]:
    return {str(item["experiment_family_id"]).strip(): item for item in ctx.contract.get("families") or []}


def _lane_lookup(ctx: ProgramContext) -> dict[str, dict[str, Any]]:
    return {str(item["variant_id"]).strip().upper(): item for item in ctx.contract.get("lanes") or []}


def _outputs(ctx: ProgramContext) -> dict[str, Any]:
    return dict(ctx.contract.get("outputs") or {})


def _governance(ctx: ProgramContext) -> dict[str, Any]:
    return dict(ctx.contract.get("governance") or {})


def _engineering_smoke_cfg(ctx: ProgramContext) -> dict[str, Any]:
    return dict(ctx.contract.get("engineering_smoke") or {})


def _analysis_grade_cfg(ctx: ProgramContext) -> dict[str, Any]:
    return dict(ctx.contract.get("analysis_grade") or {})


def _analysis_grade_budget_rows(ctx: ProgramContext) -> dict[str, dict[str, Any]]:
    budgets = (_analysis_grade_cfg(ctx).get("lane_budgets") or {})
    return {str(key).strip().upper(): dict(value or {}) for key, value in budgets.items()}


def _analysis_grade_canonical_baseline_lane(ctx: ProgramContext) -> str:
    return str(_analysis_grade_cfg(ctx).get("canonical_baseline_lane") or "ASTRA").strip().upper()


def _analysis_grade_baseline_reference_seed(ctx: ProgramContext) -> int:
    return int(_analysis_grade_cfg(ctx).get("baseline_reference_seed") or 0)


def _analysis_grade_claim_lane_scope(ctx: ProgramContext) -> list[str]:
    scope = [str(item).strip().upper() for item in _analysis_grade_cfg(ctx).get("claim_lane_scope") or []]
    return scope or list(ANALYSIS_GRADE_LANES)


def _analysis_grade_redirected_support_lanes(ctx: ProgramContext) -> dict[str, str]:
    raw = _analysis_grade_cfg(ctx).get("redirected_support_lanes") or {}
    return {
        str(lane_id).strip().upper(): str(family_id).strip()
        for lane_id, family_id in dict(raw).items()
        if str(lane_id).strip()
    }


def _engineering_smoke_budget_rows(ctx: ProgramContext) -> dict[str, dict[str, Any]]:
    budgets = (_engineering_smoke_cfg(ctx).get("budgets") or {})
    return {str(key).strip().upper(): dict(value or {}) for key, value in budgets.items()}


def _slice_manifest_path(ctx: ProgramContext, *, sample_budget: int) -> Path:
    outputs = _outputs(ctx)
    if int(sample_budget) == 512:
        return _resolve_path(ROOT, outputs["runtime_smoke_slice_manifest_512_csv"])
    if int(sample_budget) == 256:
        return _resolve_path(ROOT, outputs["runtime_smoke_slice_manifest_256_csv"])
    raise SystemExit(f"Unsupported engineering smoke slice size: {sample_budget}")


def _full_eval_manifest_path(ctx: ProgramContext) -> Path:
    cfg = _engineering_smoke_cfg(ctx)
    return _resolve_path(ROOT, cfg["full_eval_manifest_csv"])


def _engineering_smoke_output_root(ctx: ProgramContext) -> Path:
    return _resolve_path(ROOT, _outputs(ctx)["engineering_smoke_output_root"])


def _legacy_bridge_root(ctx: ProgramContext) -> Path:
    return _resolve_path(ROOT, _outputs(ctx)["legacy_runtime_smoke_bridge_root"])


def _lane_output_root(ctx: ProgramContext, variant_id: str) -> Path:
    return _engineering_smoke_output_root(ctx) / str(variant_id).strip().lower()


def _baseline_reference_summary_path(ctx: ProgramContext) -> Path:
    return _lane_output_root(ctx, "ASTRA") / "baseline_reference_summary.json"


def _phase1_run_prefix(ctx: ProgramContext) -> str:
    return str((ctx.contract.get("meta") or {}).get("tag") or "").strip() or ""


def _phase1_bundle_meta_run_prefix(ctx: ProgramContext) -> str:
    bundle = _load_yaml(_resolve_path(ROOT, (ctx.contract.get("sources") or {})["phase1_runtime_smoke_bundle"]))
    meta = bundle.get("meta") or {}
    prefix = str(meta.get("run_prefix") or "").strip()
    if not prefix:
        raise SystemExit("Phase1 runtime-smoke bundle must declare meta.run_prefix")
    return prefix


def _phase1_run_id(ctx: ProgramContext, variant_id: str) -> str:
    variant = ctx.runtime_bundle_lookup[variant_id]
    return f"{_phase1_bundle_meta_run_prefix(ctx)}_{variant['config_stub']}"


def _phase1_config_snapshot_path(ctx: ProgramContext, variant_id: str) -> Path:
    return ROOT / "experiments" / "results" / "runs" / _phase1_run_id(ctx, variant_id) / "config_snapshot.yaml"


def _phase1_accuracy_contract_csv(ctx: ProgramContext) -> Path:
    bundle = _load_yaml(_resolve_path(ROOT, (ctx.contract.get("sources") or {})["phase1_runtime_smoke_bundle"]))
    paths = bundle.get("paths") or {}
    return _resolve_path(ROOT, paths["accuracy_contract_csv"])


def _host_tuning_profile_path(ctx: ProgramContext) -> Path:
    return ctx.host_tuning_profile_path


def _host_tuning_family_alias(experiment_family_id: str) -> str:
    family_id = str(experiment_family_id).strip()
    if family_id == "anchor_validation":
        return "lane_isolation_runtime_smoke"
    return family_id


def _host_tuning_host_id(ctx: ProgramContext) -> str | None:
    host_id = str(ctx.host_tuning_profile.get("active_host_id") or "").strip()
    return host_id or None


def resolve_fuller_accuracy_policy_bundle(
    ctx: ProgramContext,
    *,
    experiment_family_id: str,
    lane_id: str,
    pass_mode: str,
    quantized_execution_semantics: str = "bitstream",
) -> dict[str, Any]:
    return resolve_accuracy_policy_bundle(
        ctx.host_tuning_profile,
        experiment_family_id=_host_tuning_family_alias(experiment_family_id),
        lane_id=str(lane_id).strip().upper(),
        pass_mode=pass_mode,
        quantized_execution_semantics=quantized_execution_semantics,
        host_id=_host_tuning_host_id(ctx),
    )


def _command_policy_pass_kind(pass_mode: str) -> str:
    normalized = str(pass_mode).strip()
    if normalized in {PASS_MODE_PAIRED, PASS_MODE_BASELINE_ONLY}:
        return PASS_KIND_BASELINE_EVAL
    return PASS_KIND_QUANTIZED_EVAL


def command_runtime_policy_from_bundle(
    bundle: dict[str, Any],
    *,
    pass_mode: str,
) -> dict[str, Any]:
    policy = resolve_pass_policy_from_bundle(
        bundle,
        pass_kind=_command_policy_pass_kind(pass_mode),
    )
    if policy is None:
        raise SystemExit(f"Missing command runtime policy for pass_mode={pass_mode}")
    return policy


def host_tuning_provenance_fields(bundle: dict[str, Any]) -> dict[str, Any]:
    pass_policies = dict(bundle.get("pass_policies") or {})
    return {
        "host_profile_id": str(bundle.get("host_profile_id") or ""),
        "pass_kind_profile": {
            pass_kind: str(policy.get("profile_id") or "")
            for pass_kind, policy in pass_policies.items()
        },
        "runtime_policy_fingerprint": {
            pass_kind: str(policy.get("runtime_policy_fingerprint") or "")
            for pass_kind, policy in pass_policies.items()
        },
        "semantic_fingerprint": {
            pass_kind: str(policy.get("semantic_fingerprint") or "")
            for pass_kind, policy in pass_policies.items()
        },
        "calibration_artifact_path": str(bundle.get("calibration_artifact_path") or ""),
    }


def runtime_health_gate_from_policy(policy: dict[str, Any]) -> dict[str, Any]:
    return dict(policy.get("runtime_health_gate") or {})


def empty_host_tuning_provenance() -> dict[str, Any]:
    return {
        "host_profile_id": "",
        "pass_kind_profile": {},
        "runtime_policy_fingerprint": {},
        "semantic_fingerprint": {},
        "calibration_artifact_path": "",
    }


def family_host_tuning_provenance(
    ctx: ProgramContext,
    *,
    experiment_family_id: str,
    lane_ids: list[str],
    pass_mode_by_lane: dict[str, str],
    quantized_execution_semantics: str = "bitstream",
) -> dict[str, Any]:
    if not lane_ids:
        return empty_host_tuning_provenance()
    host_profile_id = ""
    calibration_artifact_path = ""
    pass_kind_profile: dict[str, Any] = {}
    runtime_policy_fingerprint: dict[str, Any] = {}
    semantic_fingerprint: dict[str, Any] = {}
    for lane_id in lane_ids:
        bundle = resolve_fuller_accuracy_policy_bundle(
            ctx,
            experiment_family_id=experiment_family_id,
            lane_id=lane_id,
            pass_mode=pass_mode_by_lane[lane_id],
            quantized_execution_semantics=quantized_execution_semantics,
        )
        lane_provenance = host_tuning_provenance_fields(bundle)
        if not host_profile_id:
            host_profile_id = str(lane_provenance["host_profile_id"] or "")
        if not calibration_artifact_path:
            calibration_artifact_path = str(lane_provenance["calibration_artifact_path"] or "")
        pass_kind_profile[lane_id] = lane_provenance["pass_kind_profile"]
        runtime_policy_fingerprint[lane_id] = lane_provenance["runtime_policy_fingerprint"]
        semantic_fingerprint[lane_id] = lane_provenance["semantic_fingerprint"]
    return {
        "host_profile_id": host_profile_id,
        "pass_kind_profile": pass_kind_profile,
        "runtime_policy_fingerprint": runtime_policy_fingerprint,
        "semantic_fingerprint": semantic_fingerprint,
        "calibration_artifact_path": calibration_artifact_path,
    }


def _active_target_module_keys(ctx: ProgramContext) -> str:
    for variant_id in ("MESO", "ASTRA", "FULLER"):
        row = ctx.phase3_execution_rows.get(variant_id)
        if not row:
            continue
        command = _ensure_command_list(row.get("launch_command_json"), f"{variant_id}.launch_command_json")
        target_module_keys = _command_flag_value(command, "--bitstream_target_module_keys")
        if target_module_keys:
            return target_module_keys
    raise SystemExit("Could not recover --bitstream_target_module_keys from the legacy phase3 execution rows")


def _tuned_workers(ctx: ProgramContext, variant_id: str) -> int:
    row = ctx.phase3_efficiency_rows.get(variant_id) or {}
    raw = str(row.get("tuned_workers") or "").strip()
    return int(raw or 4)


def _tuned_eval_batch_size(ctx: ProgramContext, variant_id: str) -> int:
    row = ctx.phase3_efficiency_rows.get(variant_id) or {}
    raw = str(row.get("tuned_eval_batch_size") or "").strip()
    return int(raw or 48)


def _lane_budget(ctx: ProgramContext, variant_id: str) -> dict[str, Any]:
    budgets = _engineering_smoke_budget_rows(ctx)
    if variant_id not in budgets:
        raise SystemExit(f"Missing engineering smoke budget for {variant_id}")
    return budgets[variant_id]


def _lane_sample_budget(ctx: ProgramContext, variant_id: str) -> int:
    return int(_lane_budget(ctx, variant_id)["sample_budget"])


def _lane_pass_mode(ctx: ProgramContext, variant_id: str) -> str:
    return str(_lane_budget(ctx, variant_id)["pass_mode"]).strip()


def _lane_baseline_cache_policy(ctx: ProgramContext, variant_id: str) -> str:
    return str(_lane_budget(ctx, variant_id)["baseline_cache_policy"]).strip()


def _analysis_grade_sample_budget(ctx: ProgramContext) -> int:
    return int(_analysis_grade_cfg(ctx).get("sample_budget") or 45000)


def _analysis_grade_pass_mode(ctx: ProgramContext) -> str:
    return str(_analysis_grade_cfg(ctx).get("pass_mode") or PASS_MODE_PAIRED).strip()


def _analysis_grade_lane_budget(ctx: ProgramContext, variant_id: str) -> dict[str, Any]:
    budgets = _analysis_grade_budget_rows(ctx)
    if variant_id not in budgets:
        raise SystemExit(f"Missing analysis-grade lane budget for {variant_id}")
    return budgets[variant_id]


def _analysis_grade_lane_sample_budget(ctx: ProgramContext, variant_id: str) -> int:
    return int(_analysis_grade_lane_budget(ctx, variant_id)["sample_budget"])


def _analysis_grade_lane_pass_mode(ctx: ProgramContext, variant_id: str) -> str:
    return str(_analysis_grade_lane_budget(ctx, variant_id)["pass_mode"]).strip()


def _analysis_grade_lane_baseline_cache_policy(ctx: ProgramContext, variant_id: str) -> str:
    return str(_analysis_grade_lane_budget(ctx, variant_id)["baseline_cache_policy"]).strip()


def _phase4_eligible_flag(value: Any) -> bool:
    return bool(value)


def _lane_required_outputs(ctx: ProgramContext, variant_id: str) -> list[str]:
    lane = _lane_lookup(ctx)[variant_id]
    return [str(item) for item in lane.get("phase4_required_outputs") or []]


def _lane_required_summary_fields(ctx: ProgramContext, variant_id: str) -> list[str]:
    return list(ctx.phase2_handoff_rows[variant_id]["required_summary_fields"])


def _lane_module_specific_fields(ctx: ProgramContext, variant_id: str) -> list[str]:
    lane = _lane_lookup(ctx)[variant_id]
    return [str(item) for item in lane.get("module_specific_fields") or []]


def _lane_claim_boundary(ctx: ProgramContext, variant_id: str) -> str:
    lane = _lane_lookup(ctx)[variant_id]
    return str(lane.get("claim_boundary") or "").strip()


def _lane_forbidden_promotions(ctx: ProgramContext, variant_id: str) -> list[str]:
    lane = _lane_lookup(ctx)[variant_id]
    return [str(item) for item in lane.get("forbidden_promotions") or []]


def _engineering_smoke_reference_manifest_root(ctx: ProgramContext, variant_id: str) -> Path:
    return _lane_output_root(ctx, variant_id) / "progress"


def _engineering_smoke_results_csv(ctx: ProgramContext, variant_id: str) -> Path:
    return _lane_output_root(ctx, variant_id) / "raw_accuracy.csv"


def _engineering_smoke_annotated_csv(ctx: ProgramContext, variant_id: str) -> Path:
    return _lane_output_root(ctx, variant_id) / "annotated_accuracy.csv"


def _engineering_smoke_progress_manifest(ctx: ProgramContext, variant_id: str) -> Path:
    return _engineering_smoke_reference_manifest_root(ctx, variant_id) / "manifest.json"


def _engineering_smoke_prepared_config_root(ctx: ProgramContext, variant_id: str) -> Path:
    return _lane_output_root(ctx, variant_id) / "prepared_phase1_configs"


def _engineering_smoke_prepared_eligibility_root(ctx: ProgramContext, variant_id: str) -> Path:
    return _lane_output_root(ctx, variant_id) / "prepared_eligibility"


def _engineering_smoke_authorization_root(ctx: ProgramContext, variant_id: str) -> Path:
    return _lane_output_root(ctx, variant_id) / "authorization"


def _baseline_reference_run_id(ctx: ProgramContext) -> str:
    return f"{_phase1_run_id(ctx, 'ASTRA')}_acc_s0"


def _baseline_reference_csv(ctx: ProgramContext) -> Path:
    return _engineering_smoke_results_csv(ctx, "ASTRA")


def _baseline_source_step_id() -> str:
    return "anchor_validation__ASTRA__engineering_smoke_current"


def _analysis_grade_baseline_source_step_id() -> str:
    return "analysis_grade_replay__ASTRA__current"


def _analysis_grade_baseline_reference_run_id(ctx: ProgramContext) -> str:
    lane_id = _analysis_grade_canonical_baseline_lane(ctx)
    seed = _analysis_grade_baseline_reference_seed(ctx)
    return f"{_phase1_run_id(ctx, lane_id)}_acc_s{seed}"


def _manifest_rows(path: Path, limit: int) -> list[dict[str, str]]:
    rows = _load_csv(path)
    if len(rows) < int(limit):
        raise SystemExit(f"Manifest {path} only has {len(rows)} rows; expected at least {limit}")
    return rows[: int(limit)]


def materialize_engineering_smoke_slice_manifests(ctx: ProgramContext, *, root_dir: Path = ROOT) -> dict[str, str]:
    full_manifest = _full_eval_manifest_path(ctx)
    if not full_manifest.exists():
        raise SystemExit(f"Missing engineering-smoke source manifest: {full_manifest}")
    outputs: dict[str, str] = {}
    header = list((_load_csv(full_manifest)[:1] or [{}])[0].keys())
    if not header:
        raise SystemExit(f"Engineering-smoke source manifest is empty: {full_manifest}")
    for sample_budget in (512, 256):
        path = _slice_manifest_path(ctx, sample_budget=sample_budget)
        rows = _manifest_rows(full_manifest, sample_budget)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        outputs[str(sample_budget)] = str(path.resolve())
    return outputs


def build_experiment_matrix_rows(ctx: ProgramContext) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    family_lookup = _family_lookup(ctx)
    budgets = _engineering_smoke_budget_rows(ctx)
    slice_512 = str(_slice_manifest_path(ctx, sample_budget=512))
    slice_256 = str(_slice_manifest_path(ctx, sample_budget=256))
    full_manifest = str(_full_eval_manifest_path(ctx))
    required_device = str(_governance(ctx).get("required_device") or "").strip()
    for family_id in EXPERIMENT_FAMILY_ORDER:
        family = family_lookup[family_id]
        sample_budget: Any = ""
        slice_manifest_path: Any = ""
        pass_mode: Any = ""
        baseline_cache_policy: Any = ""
        claim_tier = str(family.get("claim_tier") or CLAIM_TIER_SUPPORT).strip()
        phase4_eligible = bool(family.get("phase4_eligible"))
        legacy_replacement_policy = str(
            family.get("legacy_replacement_policy") or LEGACY_REPLACEMENT_POLICY
        ).strip()
        execution_policy = "contract_first_program_family"
        if family_id == "anchor_validation":
            sample_budget = int(budgets["ASTRA"]["sample_budget"])
            slice_manifest_path = slice_512
            pass_mode = str(budgets["ASTRA"]["pass_mode"])
            baseline_cache_policy = str(budgets["ASTRA"]["baseline_cache_policy"])
            execution_policy = "contract_first_engineering_smoke"
        elif family_id == "lane_isolation_runtime_smoke":
            sample_budget = {
                lane_id: int(budgets[lane_id]["sample_budget"])
                for lane_id in LANE_ORDER
            }
            slice_manifest_path = {
                lane_id: slice_256 if lane_id == "FULLER" else slice_512
                for lane_id in LANE_ORDER
            }
            pass_mode = {
                lane_id: str(budgets[lane_id]["pass_mode"])
                for lane_id in LANE_ORDER
            }
            baseline_cache_policy = {
                lane_id: str(budgets[lane_id]["baseline_cache_policy"])
                for lane_id in LANE_ORDER
            }
            execution_policy = "contract_first_engineering_smoke"
        elif family_id == "analysis_grade_replay":
            claim_lanes = _analysis_grade_claim_lane_scope(ctx)
            sample_budget = {
                lane_id: _analysis_grade_lane_sample_budget(ctx, lane_id)
                for lane_id in claim_lanes
            }
            slice_manifest_path = {lane_id: full_manifest for lane_id in claim_lanes}
            pass_mode = {
                lane_id: _analysis_grade_lane_pass_mode(ctx, lane_id)
                for lane_id in claim_lanes
            }
            baseline_cache_policy = {
                lane_id: _analysis_grade_lane_baseline_cache_policy(ctx, lane_id)
                for lane_id in claim_lanes
            }
            execution_policy = "canonical_baseline_then_quantized_only_full_manifest_replay"
            claim_tier = CLAIM_TIER_ANALYSIS
            phase4_eligible = True
        elif family_id == "realism_calibration_support":
            sample_budget = _analysis_grade_sample_budget(ctx)
            slice_manifest_path = full_manifest
            pass_mode = PASS_MODE_QUANTIZED_ONLY
            baseline_cache_policy = "reuse_optional_from_astra_analysis_grade"
            execution_policy = "support_audit_family_contract"
        rows.append(
            {
                "experiment_family_id": family_id,
                "research_question_id": str(family.get("research_question_id") or "").strip(),
                "tier": str(family.get("tier") or "").strip(),
                "lane_scope": [str(item) for item in family.get("lane_scope") or []],
                "comparison_role": str(family.get("comparison_role") or "").strip(),
                "parameter_axes_json": [str(item) for item in family.get("parameter_axes") or []],
                "required_artifacts_json": [str(item) for item in family.get("required_artifacts") or []],
                "required_metrics_json": [str(item) for item in family.get("required_metrics") or []],
                "device_policy": required_device,
                "execution_policy": execution_policy,
                "stop_rules_json": [str(item) for item in family.get("stop_rules") or []],
                "report_targets_json": [str(item) for item in family.get("report_targets") or []],
                "sample_budget": sample_budget,
                "slice_manifest_path": slice_manifest_path,
                "pass_mode": pass_mode,
                "baseline_cache_policy": baseline_cache_policy,
                "claim_tier": claim_tier,
                "phase4_eligible": phase4_eligible,
                "legacy_replacement_policy": legacy_replacement_policy,
                "status": str(family.get("status") or "").strip(),
            }
        )
    return rows


def build_data_contract_rows(ctx: ProgramContext) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane_id in ["ASTRA"]:
        summary_fields = _lane_required_summary_fields(ctx, lane_id)
        module_fields = _lane_module_specific_fields(ctx, lane_id)
        for artifact_kind, artifact_grain, required_fields in (
            ("run_manifest", "run", GENERIC_MANIFEST_FIELDS),
            ("summary_surface", "lane", summary_fields),
            ("master_metrics", "run", GENERIC_MASTER_FIELDS),
            ("per_layer_surface", "layer", GENERIC_PER_LAYER_FIELDS),
            ("timeline_surface", "stage", GENERIC_TIMELINE_FIELDS),
            ("module_specific_surface", "lane", module_fields),
            ("paired_comparison_surface", "lane", GENERIC_PAIRED_COMPARISON_FIELDS),
            ("phase4_intake_surface", "lane", summary_fields),
            ("report_ready_surface", "lane", summary_fields),
        ):
            rows.append(
                {
                    "artifact_id": f"anchor_validation:{lane_id}:{artifact_kind}",
                    "experiment_family_id": "anchor_validation",
                    "artifact_kind": artifact_kind,
                    "artifact_grain": artifact_grain,
                    "producer_surface": "fuller_experiment_program_v2_engineering_smoke",
                    "required_fields_json": required_fields,
                    "phase4_gate": "engineering_smoke_ineligible_for_claim_intake",
                    "release_scope": "engineering_smoke_current",
                    "notes": f"{lane_id} / baseline cache producer",
                }
            )
    for lane_id in LANE_ORDER:
        summary_fields = _lane_required_summary_fields(ctx, lane_id)
        module_fields = _lane_module_specific_fields(ctx, lane_id)
        for artifact_kind, artifact_grain, required_fields in (
            ("run_manifest", "run", GENERIC_MANIFEST_FIELDS),
            ("summary_surface", "lane", summary_fields),
            ("master_metrics", "run", GENERIC_MASTER_FIELDS),
            ("per_layer_surface", "layer", GENERIC_PER_LAYER_FIELDS),
            ("timeline_surface", "stage", GENERIC_TIMELINE_FIELDS),
            ("module_specific_surface", "lane", module_fields),
            ("paired_comparison_surface", "lane", GENERIC_PAIRED_COMPARISON_FIELDS),
            ("phase4_intake_surface", "lane", summary_fields),
            ("report_ready_surface", "lane", summary_fields),
        ):
            rows.append(
                {
                    "artifact_id": f"lane_isolation_runtime_smoke:{lane_id}:{artifact_kind}",
                    "experiment_family_id": "lane_isolation_runtime_smoke",
                    "artifact_kind": artifact_kind,
                    "artifact_grain": artifact_grain,
                    "producer_surface": "fuller_experiment_program_v2_engineering_smoke",
                    "required_fields_json": required_fields,
                    "phase4_gate": "engineering_smoke_ineligible_for_claim_intake",
                    "release_scope": "engineering_smoke_current",
                    "notes": f"{lane_id} / engineering smoke",
                }
            )
    for lane_id in ANALYSIS_GRADE_LANES:
        summary_fields = _lane_required_summary_fields(ctx, lane_id)
        module_fields = _lane_module_specific_fields(ctx, lane_id)
        lane_note = (
            f"{lane_id} / canonical paired baseline"
            if lane_id == _analysis_grade_canonical_baseline_lane(ctx)
            else f"{lane_id} / full-manifest quantized_only replay vs ASTRA canonical baseline"
        )
        for artifact_kind, artifact_grain, required_fields in (
            ("run_manifest", "run", GENERIC_MANIFEST_FIELDS),
            ("summary_surface", "lane", summary_fields),
            ("master_metrics", "run", GENERIC_MASTER_FIELDS),
            ("per_layer_surface", "layer", GENERIC_PER_LAYER_FIELDS),
            ("timeline_surface", "stage", GENERIC_TIMELINE_FIELDS),
            ("module_specific_surface", "lane", module_fields),
            ("paired_comparison_surface", "lane", GENERIC_PAIRED_COMPARISON_FIELDS),
            ("phase4_intake_surface", "lane", summary_fields),
            ("report_ready_surface", "lane", summary_fields),
        ):
            rows.append(
                {
                    "artifact_id": f"analysis_grade_replay:{lane_id}:{artifact_kind}",
                    "experiment_family_id": "analysis_grade_replay",
                    "artifact_kind": artifact_kind,
                    "artifact_grain": artifact_grain,
                    "producer_surface": "future_analysis_grade_replay_surface",
                    "required_fields_json": required_fields,
                    "phase4_gate": "analysis_grade_claim_intake_required",
                    "release_scope": "future_analysis_grade_current",
                    "notes": lane_note,
                }
            )
    for lane_id in REALISM_CALIBRATION_LANES:
        summary_fields = _lane_required_summary_fields(ctx, lane_id)
        module_fields = _lane_module_specific_fields(ctx, lane_id)
        for artifact_kind, artifact_grain, required_fields in (
            ("run_manifest", "run", GENERIC_MANIFEST_FIELDS),
            ("summary_surface", "lane", summary_fields),
            ("master_metrics", "run", GENERIC_MASTER_FIELDS),
            ("per_layer_surface", "layer", GENERIC_PER_LAYER_FIELDS),
            ("timeline_surface", "stage", GENERIC_TIMELINE_FIELDS),
            ("module_specific_surface", "lane", module_fields),
            ("paired_comparison_surface", "lane", GENERIC_PAIRED_COMPARISON_FIELDS),
            ("phase4_intake_surface", "lane", summary_fields),
            ("report_ready_surface", "lane", summary_fields),
        ):
            rows.append(
                {
                    "artifact_id": f"realism_calibration_support:{lane_id}:{artifact_kind}",
                    "experiment_family_id": "realism_calibration_support",
                    "artifact_kind": artifact_kind,
                    "artifact_grain": artifact_grain,
                    "producer_surface": "future_realism_calibration_support_surface",
                    "required_fields_json": required_fields,
                    "phase4_gate": "support_family_gate_required",
                    "release_scope": "future_support_family_current",
                    "notes": f"{lane_id} / realism-calibration support audit",
                }
            )
    family_specific_rows = {
        "noise_robustness": [
            ("summary_surface", "profile", ["profile", "acc_top1", "acc_drop_pp", "latency_ms", "energy_j"]),
            ("paired_comparison_surface", "profile", ["profile", "latency_ms", "energy_j"]),
            ("phase4_intake_surface", "family", ["profile", "acc_top1", "acc_drop_pp", "latency_ms", "energy_j"]),
            ("report_ready_surface", "family", ["profile", "acc_top1", "acc_drop_pp", "latency_ms", "energy_j"]),
        ],
        "scaling_support": [
            ("summary_surface", "scale_point", ["batch_size", "sequence_length", "latency_ms", "throughput_images_s", "flow_buffer_peak_frac"]),
            ("paired_comparison_surface", "scale_point", ["batch_size", "sequence_length", "latency_ms", "throughput_images_s"]),
            ("phase4_intake_surface", "family", ["batch_size", "sequence_length", "latency_ms", "throughput_images_s", "flow_buffer_peak_frac"]),
            ("report_ready_surface", "family", ["batch_size", "sequence_length", "latency_ms", "throughput_images_s", "flow_buffer_peak_frac"]),
        ],
        "device_compare": [
            ("paired_comparison_surface", "device", ["host_name", "device_model", "framework", "precision_mode", "latency_ms", "avg_power_w", "energy_j", "comparison_boundary"]),
            ("phase4_intake_surface", "family", ["host_name", "device_model", "framework", "precision_mode", "latency_ms", "avg_power_w", "energy_j", "comparison_boundary"]),
            ("report_ready_surface", "family", ["host_name", "device_model", "framework", "precision_mode", "latency_ms", "avg_power_w", "energy_j", "comparison_boundary"]),
        ],
        "holdout_audit": [
            ("paired_comparison_surface", "audit", ["claim_report", "claim_summary", "blocking_truth"]),
            ("phase4_intake_surface", "audit", ["claim_report", "claim_summary", "blocking_truth"]),
            ("report_ready_surface", "audit", ["claim_report", "claim_summary", "blocking_truth"]),
        ],
        "report_pack": [
            ("report_ready_surface", "deliverable", ["deliverable_readiness", "source_family_traceability"]),
        ],
    }
    for family_id, specs in family_specific_rows.items():
        release_scope = "future_program_family"
        for artifact_kind, artifact_grain, required_fields in specs:
            rows.append(
                {
                    "artifact_id": f"{family_id}:{artifact_kind}",
                    "experiment_family_id": family_id,
                    "artifact_kind": artifact_kind,
                    "artifact_grain": artifact_grain,
                    "producer_surface": "fuller_experiment_program_contract_v2",
                    "required_fields_json": required_fields,
                    "phase4_gate": "family_gate_required",
                    "release_scope": release_scope,
                    "notes": family_id,
                }
            )
    return rows


def build_execution_plan_rows(ctx: ProgramContext) -> list[dict[str, Any]]:
    governance = _governance(ctx)
    outputs = _outputs(ctx)
    required_device = str(governance.get("required_device") or "").strip()
    launch_wrapper = [str(item) for item in governance.get("launch_wrapper") or []]
    python_bin = ROOT / ".venv311-mps" / "bin" / "python"
    runner = ROOT / "experiments" / "tools" / "run_config_conditioned_accuracy_matrix.py"
    status_checker = (
        ROOT / "experiments" / "tools" / "check_fuller_v2_runtime_smoke_result_surface.py"
    )
    target_module_keys = _active_target_module_keys(ctx)
    baseline_reference_csv = _baseline_reference_csv(ctx)
    baseline_reference_summary_json = _baseline_reference_summary_path(ctx)

    def _base_command(variant_id: str) -> list[str]:
        sample_budget = _lane_sample_budget(ctx, variant_id)
        pass_mode = _lane_pass_mode(ctx, variant_id)
        run_id = _phase1_run_id(ctx, variant_id)
        results_csv = _engineering_smoke_results_csv(ctx, variant_id)
        annotated_csv = _engineering_smoke_annotated_csv(ctx, variant_id)
        progress_root = _engineering_smoke_reference_manifest_root(ctx, variant_id)
        prepared_phase1_root = _engineering_smoke_prepared_config_root(ctx, variant_id)
        prepared_eligibility_root = _engineering_smoke_prepared_eligibility_root(ctx, variant_id)
        command = [
            str(python_bin),
            str(runner),
            "--run_ids",
            run_id,
            "--results_csv",
            str(results_csv),
            "--annotated_results_csv",
            str(annotated_csv),
            "--prepared_phase1_config_root",
            str(prepared_phase1_root),
            "--prepared_eligibility_report_root",
            str(prepared_eligibility_root),
            "--progress_root",
            str(progress_root),
            "--progress_heartbeat_interval_seconds",
            "15.0",
            "--stall_timeout_seconds",
            "180.0",
            "--prelaunch_runtime_smoke_samples",
            "16",
            "--prelaunch_min_samples_per_hour",
            "30.0",
            "--prelaunch_max_seconds_per_sample",
            "120.0",
            "--pathological_min_samples_per_hour",
            "12.0",
            "--pathological_max_seconds_per_sample",
            "300.0",
            "--pathological_max_eta_current_rate_seconds",
            "86400.0",
            "--pathological_min_processed_samples",
            "4",
            "--pathological_min_elapsed_seconds",
            "300.0",
            "--accuracy_backend",
            "mlx",
            "--models",
            "mobilevit_s",
            "--device",
            required_device,
            "--workers",
            str(_tuned_workers(ctx, variant_id)),
            "--eval_batch_size",
            str(_tuned_eval_batch_size(ctx, variant_id)),
            "--seeds",
            "0",
            "--evidence_tier",
            "runtime_smoke",
            "--manifest_override",
            str(_slice_manifest_path(ctx, sample_budget=sample_budget)),
            "--pass_mode",
            pass_mode,
            "--annotation_measurement_truth_class",
            "bridge_only_nonbitstream_measured",
            "--bitstream_measurement_truth_class",
            "bitstream_limited_surface_pilot",
            "--annotation_contract_note",
            "engineering_smoke_validation_only_not_claim_tier",
            "--bitstream_contract_note",
            "engineering_smoke_validation_only_not_claim_tier",
            "--enable_bitstream_pilot",
            "--bitstream_surface_scope",
            str(_engineering_smoke_cfg(ctx).get("limited_runtime_surface_scope") or "limited_linear_attention_pilot"),
            "--bitstream_target_module_keys",
            target_module_keys,
            "--resume",
        ]
        if variant_id == "ASTRA":
            command.extend(
                [
                    "--baseline_reference_summary_json",
                    str(baseline_reference_summary_json),
                ]
            )
        else:
            command.extend(
                [
                    "--baseline_reference_csv",
                    str(baseline_reference_csv),
                    "--baseline_reference_run_id",
                    _baseline_reference_run_id(ctx),
                ]
            )
        return command

    def _checker_command(variant_id: str) -> list[str]:
        eval_run_id = f"{_phase1_run_id(ctx, variant_id)}_acc_s0"
        return [
            str(python_bin),
            str(status_checker),
            "--mode",
            "postrun",
            "--lane_id",
            variant_id,
            "--annotated_csv",
            str(_engineering_smoke_annotated_csv(ctx, variant_id)),
            "--eval_run_id",
            eval_run_id,
        ]

    rows: list[dict[str, Any]] = [
        {
            "step_id": _baseline_source_step_id(),
            "experiment_family_id": "anchor_validation",
            "tier": "runtime_smoke",
            "package_kind": "engineering_smoke_paired",
            "lane_id": "ASTRA",
            "command_json": _base_command("ASTRA"),
            "checker_kind": "v2_result_surface_contract",
            "checker_command_json": _checker_command("ASTRA"),
            "progress_root": str(_engineering_smoke_reference_manifest_root(ctx, "ASTRA")),
            "required_device": required_device,
            "launch_wrapper_json": launch_wrapper,
            "serial_group": "fuller_experiment_program_v2_engineering_smoke",
            "resume_enabled": True,
            "stop_on_failure": True,
            "current_run_policy": "active_engineering_smoke_v2",
            "sample_budget": _lane_sample_budget(ctx, "ASTRA"),
            "slice_manifest_path": str(_slice_manifest_path(ctx, sample_budget=_lane_sample_budget(ctx, "ASTRA"))),
            "pass_mode": _lane_pass_mode(ctx, "ASTRA"),
            "baseline_source_step_id": "",
            "baseline_reference_csv": str(_engineering_smoke_results_csv(ctx, "ASTRA")),
            "baseline_reference_summary_json": str(baseline_reference_summary_json),
            "phase4_eligible": False,
            "legacy_replacement_policy": LEGACY_REPLACEMENT_POLICY,
        }
    ]
    for lane_id in ENGINEERING_SMOKE_LANES:
        rows.append(
            {
                "step_id": f"lane_isolation_runtime_smoke__{lane_id}__engineering_smoke_current",
                "experiment_family_id": "lane_isolation_runtime_smoke",
                "tier": "runtime_smoke",
                "package_kind": "engineering_smoke_quantized_only",
                "lane_id": lane_id,
                "command_json": _base_command(lane_id),
                "checker_kind": "v2_result_surface_contract",
                "checker_command_json": _checker_command(lane_id),
                "progress_root": str(_engineering_smoke_reference_manifest_root(ctx, lane_id)),
                "required_device": required_device,
                "launch_wrapper_json": launch_wrapper,
                "serial_group": "fuller_experiment_program_v2_engineering_smoke",
                "resume_enabled": True,
                "stop_on_failure": True,
                "current_run_policy": "active_engineering_smoke_v2",
                "sample_budget": _lane_sample_budget(ctx, lane_id),
                "slice_manifest_path": str(_slice_manifest_path(ctx, sample_budget=_lane_sample_budget(ctx, lane_id))),
                "pass_mode": _lane_pass_mode(ctx, lane_id),
                "baseline_source_step_id": _baseline_source_step_id(),
                "baseline_reference_csv": str(baseline_reference_csv),
                "baseline_reference_summary_json": str(baseline_reference_summary_json),
                "phase4_eligible": False,
                "legacy_replacement_policy": LEGACY_REPLACEMENT_POLICY,
            }
        )
    return rows


def build_phase4_intake_rows(ctx: ProgramContext) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = ctx.contract.get("sources") or {}
    common_provenance = [
        str(_resolve_path(ROOT, sources["phase1_runtime_smoke_bundle"])),
        str(_resolve_path(ROOT, sources["phase2_handoff_contract_csv"])),
        str(_resolve_path(ROOT, sources["phase3_execution_packet_csv"])),
        str(_resolve_path(ROOT, sources["phase3_serial_launch_status_json"])),
        str(_host_tuning_profile_path(ctx)),
    ]

    def _phase4_row(
        *,
        experiment_family_id: str,
        required_outputs_json: list[str],
        required_manifest_fields_json: list[str],
        required_summary_fields_json: list[str],
        required_provenance_inputs_json: list[str],
        claim_boundary: str,
        forbidden_promotions_json: list[str],
        claim_tier: str,
        phase4_eligible: bool,
        baseline_dependency: str,
        full_dataset_required: bool,
        intake_status: str,
        host_tuning_provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "experiment_family_id": experiment_family_id,
            "required_outputs_json": required_outputs_json,
            "required_manifest_fields_json": required_manifest_fields_json,
            "required_summary_fields_json": required_summary_fields_json,
            "required_provenance_inputs_json": required_provenance_inputs_json,
            **(host_tuning_provenance or empty_host_tuning_provenance()),
            "claim_boundary": claim_boundary,
            "forbidden_promotions_json": forbidden_promotions_json,
            "claim_tier": claim_tier,
            "phase4_eligible": phase4_eligible,
            "baseline_dependency": baseline_dependency,
            "full_dataset_required": full_dataset_required,
            "intake_status": intake_status,
        }

    for family_id in EXPERIMENT_FAMILY_ORDER:
        family = _family_lookup(ctx)[family_id]
        if family_id == "anchor_validation":
            lane_ids = ["ASTRA"]
            rows.append(
                _phase4_row(
                    experiment_family_id=family_id,
                    required_outputs_json=list(
                        dict.fromkeys(sum((_lane_required_outputs(ctx, lane_id) for lane_id in lane_ids), []))
                    ),
                    required_manifest_fields_json=GENERIC_MANIFEST_FIELDS,
                    required_summary_fields_json=list(
                        dict.fromkeys(sum((_lane_required_summary_fields(ctx, lane_id) for lane_id in lane_ids), []))
                    ),
                    required_provenance_inputs_json=common_provenance,
                    host_tuning_provenance=family_host_tuning_provenance(
                        ctx,
                        experiment_family_id=family_id,
                        lane_ids=lane_ids,
                        pass_mode_by_lane={"ASTRA": _lane_pass_mode(ctx, "ASTRA")},
                    ),
                    claim_boundary="engineering smoke anchor only; baseline cache and minimal paired output are not claim-tier evidence",
                    forbidden_promotions_json=[
                        "bitstream_model_level_measured_for_claim",
                        "benchmark_claim_ready",
                        "proxy_promotion_ready",
                    ],
                    claim_tier=CLAIM_TIER_ENGINEERING,
                    phase4_eligible=False,
                    baseline_dependency="",
                    full_dataset_required=False,
                    intake_status="engineering_smoke_not_phase4_eligible",
                )
            )
            continue
        if family_id == "lane_isolation_runtime_smoke":
            required_outputs: list[str] = []
            required_summary_fields: list[str] = []
            claim_boundaries: list[str] = []
            forbidden: list[str] = [
                "bitstream_model_level_measured_for_claim",
                "benchmark_claim_ready",
                "proxy_promotion_ready",
            ]
            for lane_id in LANE_ORDER:
                required_outputs.extend(_lane_required_outputs(ctx, lane_id))
                required_summary_fields.extend(_lane_required_summary_fields(ctx, lane_id))
                claim_boundaries.append(_lane_claim_boundary(ctx, lane_id))
                forbidden.extend(_lane_forbidden_promotions(ctx, lane_id))
            rows.append(
                _phase4_row(
                    experiment_family_id=family_id,
                    required_outputs_json=list(dict.fromkeys(required_outputs)),
                    required_manifest_fields_json=GENERIC_MANIFEST_FIELDS,
                    required_summary_fields_json=list(dict.fromkeys(required_summary_fields)),
                    required_provenance_inputs_json=common_provenance,
                    host_tuning_provenance=family_host_tuning_provenance(
                        ctx,
                        experiment_family_id=family_id,
                        lane_ids=LANE_ORDER,
                        pass_mode_by_lane={lane_id: _lane_pass_mode(ctx, lane_id) for lane_id in LANE_ORDER},
                    ),
                    claim_boundary="engineering smoke only; module fields and health checks may not be promoted as claim-tier evidence",
                    forbidden_promotions_json=list(dict.fromkeys(forbidden)),
                    claim_tier=CLAIM_TIER_ENGINEERING,
                    phase4_eligible=False,
                    baseline_dependency=_baseline_source_step_id(),
                    full_dataset_required=False,
                    intake_status="engineering_smoke_not_phase4_eligible",
                )
            )
            continue
        if family_id == "analysis_grade_replay":
            required_outputs: list[str] = []
            required_summary_fields: list[str] = []
            forbidden: list[str] = []
            for lane_id in _analysis_grade_claim_lane_scope(ctx):
                required_outputs.extend(_lane_required_outputs(ctx, lane_id))
                required_summary_fields.extend(_lane_required_summary_fields(ctx, lane_id))
                forbidden.extend(_lane_forbidden_promotions(ctx, lane_id))
            rows.append(
                _phase4_row(
                    experiment_family_id=family_id,
                    required_outputs_json=list(dict.fromkeys(required_outputs)),
                    required_manifest_fields_json=GENERIC_MANIFEST_FIELDS,
                    required_summary_fields_json=list(dict.fromkeys(required_summary_fields)),
                    required_provenance_inputs_json=common_provenance + [str(_full_eval_manifest_path(ctx))],
                    host_tuning_provenance=family_host_tuning_provenance(
                        ctx,
                        experiment_family_id=family_id,
                        lane_ids=_analysis_grade_claim_lane_scope(ctx),
                        pass_mode_by_lane={
                            lane_id: _analysis_grade_lane_pass_mode(ctx, lane_id)
                            for lane_id in _analysis_grade_claim_lane_scope(ctx)
                        },
                    ),
                    claim_boundary="ASTRA paired canonical baseline plus full-manifest quantized-only replay for MESO/HOPS/DET/SPARSE/FULLER is the only claim-eligible FULLER lane family",
                    forbidden_promotions_json=list(
                        dict.fromkeys(forbidden + ["engineering_smoke_substitution_for_claim"])
                    ),
                    claim_tier=CLAIM_TIER_ANALYSIS,
                    phase4_eligible=True,
                    baseline_dependency=_analysis_grade_baseline_source_step_id(),
                    full_dataset_required=True,
                    intake_status="astra_canonical_then_mainline_quantized_only_replay",
                )
            )
            continue
        if family_id == "realism_calibration_support":
            lane_id = REALISM_CALIBRATION_LANES[0]
            rows.append(
                _phase4_row(
                    experiment_family_id=family_id,
                    required_outputs_json=_lane_required_outputs(ctx, lane_id),
                    required_manifest_fields_json=GENERIC_MANIFEST_FIELDS,
                    required_summary_fields_json=_lane_required_summary_fields(ctx, lane_id),
                    required_provenance_inputs_json=common_provenance + [str(_full_eval_manifest_path(ctx))],
                    host_tuning_provenance=family_host_tuning_provenance(
                        ctx,
                        experiment_family_id=family_id,
                        lane_ids=[lane_id],
                        pass_mode_by_lane={lane_id: PASS_MODE_QUANTIZED_ONLY},
                    ),
                    claim_boundary="PHY realism/calibration remains support-audit evidence only and is outside the paper's main claim-tier lane family",
                    forbidden_promotions_json=[
                        "claim_lane_substitution_from_phy",
                        "engineering_smoke_substitution_for_claim",
                        "hardware_claim_from_support_only_lane",
                        "archived_row_relabel",
                    ],
                    claim_tier=CLAIM_TIER_SUPPORT,
                    phase4_eligible=False,
                    baseline_dependency=_analysis_grade_baseline_source_step_id(),
                    full_dataset_required=True,
                    intake_status="support_audit_family_contract_current",
                )
            )
            continue
        if family_id == "noise_robustness":
            outputs = ["noise_accuracy_summary.csv", "noise_paired_metrics.csv"]
            summary_fields = ["profile", "acc_top1", "acc_drop_pp", "latency_ms", "energy_j"]
            provenance = [str(_resolve_path(ROOT, sources["legacy_implementation_design_contract"]))]
            claim_boundary = "noise robustness remains support evidence only"
            forbidden = ["engineering_smoke_substitution_for_claim", "archived_row_relabel"]
            claim_tier = CLAIM_TIER_SUPPORT
            phase4_eligible = False
            baseline_dependency = ""
            full_dataset_required = False
            intake_status = "support_family_gate_required"
        elif family_id == "scaling_support":
            outputs = ["scaling_summary.csv"]
            summary_fields = ["batch_size", "sequence_length", "latency_ms", "throughput_images_s", "flow_buffer_peak_frac"]
            provenance = [str(_resolve_path(ROOT, sources["legacy_implementation_execution_bundle"]))]
            claim_boundary = "scaling support remains bounded support evidence"
            forbidden = ["engineering_smoke_substitution_for_claim", "archived_row_relabel"]
            claim_tier = CLAIM_TIER_SUPPORT
            phase4_eligible = False
            baseline_dependency = ""
            full_dataset_required = False
            intake_status = "support_family_gate_required"
        elif family_id == "device_compare":
            outputs = ["device_compare_metrics.csv"]
            summary_fields = ["host_name", "device_model", "framework", "precision_mode", "latency_ms", "avg_power_w", "energy_j", "comparison_boundary"]
            provenance = [str(_resolve_path(ROOT, sources["legacy_implementation_execution_bundle"]))]
            claim_boundary = "device compare remains contextual and not benchmark-equivalence evidence"
            forbidden = ["engineering_smoke_substitution_for_claim", "archived_row_relabel"]
            claim_tier = CLAIM_TIER_SUPPORT
            phase4_eligible = False
            baseline_dependency = ""
            full_dataset_required = False
            intake_status = "support_family_gate_required"
        elif family_id == "holdout_audit":
            outputs = ["holdout_claim_report.md", "holdout_claim_summary.csv"]
            summary_fields = ["claim_report", "claim_summary", "blocking_truth"]
            provenance = [str(_resolve_path(ROOT, sources["legacy_implementation_design_contract"]))]
            claim_boundary = "holdout audit is the only claim-audit family and cannot be substituted by smoke outputs"
            forbidden = ["smoke_substitution_for_holdout", "engineering_smoke_substitution_for_claim", "archived_row_relabel"]
            claim_tier = CLAIM_TIER_HOLDOUT
            phase4_eligible = False
            baseline_dependency = ""
            full_dataset_required = False
            intake_status = "holdout_family_gate_required"
        else:
            outputs = ["report_pack_manifest.json"]
            summary_fields = ["deliverable_readiness", "source_family_traceability"]
            provenance = [str(_resolve_path(ROOT, sources["phase2_modeling_contract"]))]
            claim_boundary = "report pack may only cite declared claim-tier families"
            forbidden = ["engineering_smoke_as_benchmark_source", "undeclared_upstream_inputs", "archived_row_relabel"]
            claim_tier = CLAIM_TIER_SUPPORT
            phase4_eligible = False
            baseline_dependency = ""
            full_dataset_required = False
            intake_status = "report_pack_contract_current"
        rows.append(
            _phase4_row(
                experiment_family_id=family_id,
                required_outputs_json=outputs,
                required_manifest_fields_json=GENERIC_MANIFEST_FIELDS,
                required_summary_fields_json=summary_fields,
                required_provenance_inputs_json=provenance,
                host_tuning_provenance=empty_host_tuning_provenance(),
                claim_boundary=claim_boundary,
                forbidden_promotions_json=forbidden,
                claim_tier=claim_tier,
                phase4_eligible=phase4_eligible,
                baseline_dependency=baseline_dependency,
                full_dataset_required=full_dataset_required,
                intake_status=intake_status,
            )
        )
    return rows


def build_report_contract_rows(ctx: ProgramContext) -> list[dict[str, Any]]:
    deliverables = {
        str(item["deliverable_id"]).strip(): item
        for item in ctx.contract.get("report_deliverables") or []
    }
    rows: list[dict[str, Any]] = []
    for deliverable_id in REPORT_DELIVERABLE_ORDER:
        item = deliverables[deliverable_id]
        rows.append(
            {
                "deliverable_id": deliverable_id,
                "deliverable_kind": str(item.get("deliverable_kind") or "").strip(),
                "source_family_ids_json": [str(value) for value in item.get("source_family_ids") or []],
                "required_artifacts_json": [str(value) for value in item.get("required_artifacts") or []],
                "required_metrics_json": [str(value) for value in item.get("required_metrics") or []],
                "readiness_gate": str(item.get("readiness_gate") or "").strip(),
                "release_scope": str(item.get("release_scope") or "").strip(),
                "status": str(item.get("status") or "").strip(),
            }
        )
    return rows
