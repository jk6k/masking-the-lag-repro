"""Shared conv-native fidelity semantics for bitstream MobileViT workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from accuracy.bitstream_semantics import BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT_DIR / "experiments" / "results" / "accuracy" / "bitstream_slices"

CONV_NATIVE_CLASS_DEPTHWISE_GROUPED_PATCH_DOMINANT = (
    "depthwise_grouped_patch_dominant"
)
CONV_NATIVE_CLASS_STANDARD_SPATIAL_FILTER_BANK_VISIBLE = (
    "standard_spatial_filter_bank_visible"
)
CONV_NATIVE_CLASS_POINTWISE_GEMM_ALIGNED = "pointwise_gemm_aligned"

CONV_FIDELITY_STAGE_RUNTIME_MODELED = "runtime_modeled"
CONV_FIDELITY_STAGE_HARDWARE_EVIDENCE_CLOSED = "hardware_evidence_closed"
CONV_FIDELITY_STAGE_MEASURED_CLOSED = "measured_closed"

CONV_HARDWARE_EVIDENCE_UNCLOSED = "conv_hardware_evidence_unclosed"

CONV_EVIDENCE_MANIFEST_SCHEMA_VERSION = "true_sc_conv_evidence_manifest.v1"
CONV_FOCUSED_MEASURED_PACKAGE_SCHEMA_VERSION = "true_sc_conv_focused_measured_package.v1"
CONV_FOCUSED_MEASURED_PACKAGE_STATUS_DRY_RUN = (
    "dry_run_only_no_measured_run_authorized"
)
CONV_FOCUSED_MEASURED_PACKAGE_STATUS_AUTHORIZED = "authorized_conv_focused_slice"
CONV_FOCUSED_CLAIM_SURFACE_STATUS = "conv_focused_claim_surface_runtime"
CONV_MEASURED_CLOSURE_STATUS_RUNTIME_MODELED = "runtime_modeled"
CONV_MEASURED_CLOSURE_STATUS_UNCLOSED = "hardware_evidence_unclosed"
CONV_MEASURED_CLOSURE_STATUS_MEASURED_CLOSED = "measured_closed"
LEGACY_ALL_TARGET_MEASURED_ANCHOR_RUN_ID = "20260415_sc_default_dark_launch_candidate_acc_s0"

DEFAULT_CONV_EVIDENCE_MANIFEST_PATH = (
    RESULTS_ROOT / "true_sc_conv_evidence_manifest_mobilevit_s_postopt7_20260418.json"
)
DEFAULT_CONV_FOCUSED_MEASURED_PACKAGE_PATH = (
    RESULTS_ROOT / "true_sc_conv_focused_measured_package_mobilevit_s_postopt7_20260418.json"
)

DEFAULT_MOBILEVIT_S_CONV_ARTIFACTS = {
    "shape_risk_json": RESULTS_ROOT
    / "true_sc_t6_conv_lowering_shape_risk_mobilevit_s_postopt7_20260418.json",
    "overhead_band_json": RESULTS_ROOT
    / "true_sc_t6_conv_lowering_overhead_band_mobilevit_s_postopt7_20260418.json",
    "memory_layout_reuse_json": RESULTS_ROOT
    / "true_sc_t6_conv_memory_layout_reuse_provenance_mobilevit_s_postopt7_20260418.json",
    "pipeline_cycle_model_json": RESULTS_ROOT
    / "true_sc_t6_conv_pipeline_cycle_model_mobilevit_s_postopt7_20260418.json",
    "psum_accumulator_scope_json": RESULTS_ROOT
    / "true_sc_t6_conv_psum_accumulator_scope_mobilevit_s_postopt7_20260418.json",
}

PROVENANCE_CLASS_TO_PIPELINE_CLASS = {
    "depthwise_patch_dominant_operand_reuse": "patch_generate_then_multiply",
    "filter_bank_residency_visible_operand_reuse": (
        "filter_preload_or_patch_feed_then_multiply_accumulate_writeback"
    ),
    "patch_generation_dominant_with_channel_fanout_reuse": (
        "patch_generate_broadcast_multiply_accumulate"
    ),
}


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_conv_target_module_key_for_runtime(module_key: str) -> str:
    normalized = str(module_key or "").strip()
    for suffix in (".block.conv", ".block.norm", ".block.act"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)] + suffix.replace(".block", "")
    return normalized


def conv_focused_runtime_target_module_keys(
    manifest_payload: dict[str, Any],
) -> list[str]:
    target_module_keys = (
        ((manifest_payload.get("conv_focused_target_set") or {}).get("target_module_keys"))
        or []
    )
    normalized = [
        normalize_conv_target_module_key_for_runtime(str(module_key))
        for module_key in target_module_keys
        if str(module_key).strip()
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for module_key in normalized:
        if module_key in seen:
            continue
        seen.add(module_key)
        deduped.append(module_key)
    return deduped


def _pair(value: object, default: tuple[int, int] = (1, 1)) -> list[int]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [max(1, _safe_int(value[0], default[0])), max(1, _safe_int(value[1], default[1]))]
    scalar = max(1, _safe_int(value, default[0]))
    return [scalar, scalar]


def classify_conv_native_class(op: dict[str, object]) -> str:
    kernel = _pair(op.get("kernel"))
    groups = max(1, _safe_int(op.get("groups"), 1))
    if groups > 1:
        return CONV_NATIVE_CLASS_DEPTHWISE_GROUPED_PATCH_DOMINANT
    if kernel != [1, 1]:
        return CONV_NATIVE_CLASS_STANDARD_SPATIAL_FILTER_BANK_VISIBLE
    return CONV_NATIVE_CLASS_POINTWISE_GEMM_ALIGNED


def _default_provenance_class(conv_native_class: str) -> str:
    if conv_native_class == CONV_NATIVE_CLASS_DEPTHWISE_GROUPED_PATCH_DOMINANT:
        return "depthwise_patch_dominant_operand_reuse"
    if conv_native_class == CONV_NATIVE_CLASS_STANDARD_SPATIAL_FILTER_BANK_VISIBLE:
        return "filter_bank_residency_visible_operand_reuse"
    return "patch_generation_dominant_with_channel_fanout_reuse"


def build_conv_runtime_semantics(
    op: dict[str, object],
    *,
    runtime_stream_reuse_policy: str,
    provenance_class: str | None = None,
) -> dict[str, object]:
    conv_native_class = classify_conv_native_class(op)
    resolved_provenance_class = provenance_class or _default_provenance_class(conv_native_class)
    pipeline_class = PROVENANCE_CLASS_TO_PIPELINE_CLASS.get(
        resolved_provenance_class,
        PROVENANCE_CLASS_TO_PIPELINE_CLASS[_default_provenance_class(conv_native_class)],
    )
    if conv_native_class == CONV_NATIVE_CLASS_DEPTHWISE_GROUPED_PATCH_DOMINANT:
        patch_materialization_scope = "per_output_patch_group_bundle"
        filter_materialization_scope = "per_out_channel_filter_bundle"
    elif conv_native_class == CONV_NATIVE_CLASS_STANDARD_SPATIAL_FILTER_BANK_VISIBLE:
        patch_materialization_scope = "per_output_spatial_patch_window"
        filter_materialization_scope = "filter_bank_epoch_visible_residency"
    else:
        patch_materialization_scope = "per_output_channel_bundle"
        filter_materialization_scope = "pointwise_filter_bank_residency"
    return {
        "kernel": _pair(op.get("kernel")),
        "stride": _pair(op.get("stride")),
        "groups": max(1, _safe_int(op.get("groups"), 1)),
        "dilation": _pair(op.get("dilation"), (1, 1)),
        "reuse_policy": str(runtime_stream_reuse_policy or "").strip(),
        "patch_materialization_scope": patch_materialization_scope,
        "filter_materialization_scope": filter_materialization_scope,
        "pipeline_class": pipeline_class,
    }


def conv_fidelity_blockers_for_stage(stage: str) -> list[str]:
    return [] if stage == CONV_FIDELITY_STAGE_MEASURED_CLOSED else [CONV_HARDWARE_EVIDENCE_UNCLOSED]


def _resolve_mobilevit_s_artifacts(
    *,
    model_key: str | None,
    ops_path: Path | None,
) -> dict[str, Path] | None:
    model_name = str(model_key or "").strip().lower()
    ops_name = "" if ops_path is None else ops_path.name.strip().lower()
    if model_name == "mobilevit_s" or ops_name == "ops_mobilevit_s.json":
        return dict(DEFAULT_MOBILEVIT_S_CONV_ARTIFACTS)
    return None


def _read_ops_payload(ops_path: Path) -> list[dict[str, object]]:
    payload = _load_json(ops_path)
    if not isinstance(payload, dict):
        return []
    ops = payload.get("ops")
    if not isinstance(ops, list):
        return []
    return [row for row in ops if isinstance(row, dict)]


def _select_conv_focused_target_set(rows: list[dict[str, Any]]) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()

    def _extend(names: list[str]) -> None:
        for name in names:
            if not name or name in seen:
                continue
            seen.add(name)
            selected.append(name)

    high_and_medium_high = [
        str(row["module_key"])
        for row in rows
        if str(row.get("risk_class") or "") in {"high", "medium_high"}
    ]
    _extend(high_and_medium_high)

    standard_spatial = sorted(
        [
            row
            for row in rows
            if row.get("conv_native_class")
            == CONV_NATIVE_CLASS_STANDARD_SPATIAL_FILTER_BANK_VISIBLE
        ],
        key=lambda row: float(row.get("nominal_weighted_cost_proxy") or 0.0),
        reverse=True,
    )
    _extend([str(row["module_key"]) for row in standard_spatial[:5]])

    pointwise = sorted(
        [
            row
            for row in rows
            if row.get("conv_native_class") == CONV_NATIVE_CLASS_POINTWISE_GEMM_ALIGNED
        ],
        key=lambda row: float(row.get("baseline_lowered_ops_proxy") or 0.0),
        reverse=True,
    )
    _extend([str(row["module_key"]) for row in pointwise[:3]])
    return selected


def build_conv_evidence_manifest(
    *,
    model_key: str,
    ops_path: Path,
) -> dict[str, Any]:
    ops = _read_ops_payload(ops_path)
    conv_ops = [op for op in ops if str(op.get("type") or "").strip().lower() == "conv2d"]
    artifacts = _resolve_mobilevit_s_artifacts(model_key=model_key, ops_path=ops_path) or {}
    shape_payload = _load_json(artifacts["shape_risk_json"]) if "shape_risk_json" in artifacts else None
    overhead_payload = _load_json(artifacts["overhead_band_json"]) if "overhead_band_json" in artifacts else None
    reuse_payload = _load_json(artifacts["memory_layout_reuse_json"]) if "memory_layout_reuse_json" in artifacts else None
    pipeline_payload = _load_json(artifacts["pipeline_cycle_model_json"]) if "pipeline_cycle_model_json" in artifacts else None
    psum_payload = _load_json(artifacts["psum_accumulator_scope_json"]) if "psum_accumulator_scope_json" in artifacts else None

    shape_rows = {
        str(row.get("name") or ""): row
        for row in (shape_payload.get("rows") or [] if isinstance(shape_payload, dict) else [])
        if isinstance(row, dict) and str(row.get("name") or "").strip()
    }
    overhead_rows = {
        str(row.get("name") or ""): row
        for row in (overhead_payload.get("rows") or [] if isinstance(overhead_payload, dict) else [])
        if isinstance(row, dict) and str(row.get("name") or "").strip()
    }
    reuse_rows = {
        str(row.get("name") or ""): row
        for row in (reuse_payload.get("rows") or [] if isinstance(reuse_payload, dict) else [])
        if isinstance(row, dict) and str(row.get("name") or "").strip()
    }
    pipeline_rows = {
        str(row.get("memory_layout_provenance_class") or ""): row
        for row in (pipeline_payload.get("rows") or [] if isinstance(pipeline_payload, dict) else [])
        if isinstance(row, dict) and str(row.get("memory_layout_provenance_class") or "").strip()
    }
    psum_rows = {
        str(row.get("memory_layout_provenance_class") or ""): row
        for row in (psum_payload.get("rows") or [] if isinstance(psum_payload, dict) else [])
        if isinstance(row, dict) and str(row.get("memory_layout_provenance_class") or "").strip()
    }

    manifest_rows: list[dict[str, Any]] = []
    for op in conv_ops:
        module_key = str(op.get("name") or "").strip()
        conv_native_class = classify_conv_native_class(op)
        shape_row = shape_rows.get(module_key, {})
        overhead_row = overhead_rows.get(module_key, {})
        reuse_row = reuse_rows.get(module_key, {})
        provenance_class = str(
            reuse_row.get("memory_layout_provenance_class")
            or _default_provenance_class(conv_native_class)
        )
        pipeline_row = pipeline_rows.get(provenance_class, {})
        psum_row = psum_rows.get(provenance_class, {})
        required_artifacts = [
            artifact_name
            for artifact_name, source_row in (
                ("shape_risk", shape_row),
                ("overhead_band", overhead_row),
                ("memory_layout_reuse_provenance", reuse_row),
                ("pipeline_cycle_model", pipeline_row),
                ("psum_accumulator_scope", psum_row),
            )
            if source_row
        ]
        premeasurement_evidence_status = (
            "bound_existing_provenance_artifacts"
            if len(required_artifacts) == 5
            else "partial_existing_provenance_artifacts"
        )
        manifest_rows.append(
            {
                "module_key": module_key,
                "conv_native_class": conv_native_class,
                "risk_class": str(shape_row.get("risk_class") or "unclassified"),
                "weighted_cost_rank": 0,
                "requires_measured_closure": bool(
                    overhead_row.get("requires_measured_validation", True)
                ),
                "required_artifacts": required_artifacts,
                "premeasurement_evidence_status": premeasurement_evidence_status,
                "measured_evidence_status": "unclosed",
                "shape_risk": {
                    "risk_class": shape_row.get("risk_class"),
                    "lowering_class": shape_row.get("lowering_class"),
                    "risk_tags": shape_row.get("risk_tags"),
                },
                "overhead_band": {
                    "rule_id": overhead_row.get("rule_id"),
                    "nominal_factor": overhead_row.get("nominal_factor"),
                    "nominal_adjusted_ops_proxy": overhead_row.get(
                        "nominal_adjusted_ops_proxy"
                    ),
                    "nominal_extra_ops_proxy": overhead_row.get(
                        "nominal_extra_ops_proxy"
                    ),
                },
                "memory_layout_reuse_provenance": {
                    "memory_layout_provenance_class": provenance_class,
                    "patch_generated_uint8_cells_proxy": reuse_row.get(
                        "patch_generated_uint8_cells_proxy"
                    ),
                    "filter_resident_uint8_cells_proxy": reuse_row.get(
                        "filter_resident_uint8_cells_proxy"
                    ),
                    "generated_reduction_factor_vs_cell_unique": reuse_row.get(
                        "generated_reduction_factor_vs_cell_unique"
                    ),
                },
                "pipeline_cycle_model": {
                    "dominant_stage_driver": pipeline_row.get("dominant_stage_driver"),
                    "pipeline_overlap_assumption": pipeline_row.get(
                        "pipeline_overlap_assumption"
                    ),
                    "evidence_status": pipeline_row.get("evidence_status"),
                },
                "psum_accumulator_scope": {
                    "proposed_psum_locality": psum_row.get("proposed_psum_locality"),
                    "evidence_status": psum_row.get("evidence_status"),
                },
                "baseline_lowered_ops_proxy": float(
                    shape_row.get("lowered_mdn_ops") or (_safe_int(op.get("m"), 0) * _safe_int(op.get("d"), 0) * _safe_int(op.get("n"), 0))
                ),
                "nominal_weighted_cost_proxy": float(
                    overhead_row.get("nominal_adjusted_ops_proxy")
                    or shape_row.get("lowered_mdn_ops")
                    or (_safe_int(op.get("m"), 0) * _safe_int(op.get("d"), 0) * _safe_int(op.get("n"), 0))
                ),
            }
        )

    manifest_rows.sort(
        key=lambda row: float(row.get("nominal_weighted_cost_proxy") or 0.0),
        reverse=True,
    )
    for rank, row in enumerate(manifest_rows, start=1):
        row["weighted_cost_rank"] = rank

    target_set = _select_conv_focused_target_set(manifest_rows)
    target_set_sha256 = hashlib.sha256(
        json.dumps(target_set, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return {
        "schema_version": CONV_EVIDENCE_MANIFEST_SCHEMA_VERSION,
        "created_date": "2026-04-18",
        "model": model_key,
        "ops_manifest_path": str(ops_path.resolve()),
        "artifacts": {key: str(path.resolve()) for key, path in artifacts.items()},
        "rows": manifest_rows,
        "conv_focused_target_set": {
            "selection_rule_version": "conv_focused_target_selection_v1",
            "target_module_keys": target_set,
            "target_module_key_count": len(target_set),
            "target_set_sha256": target_set_sha256,
            "legacy_run_ids_not_authorized_for_strong_conv_closure": [
                LEGACY_ALL_TARGET_MEASURED_ANCHOR_RUN_ID
            ],
        },
    }


def resolve_conv_evidence_manifest(
    *,
    model_key: str,
    ops_path: str | Path,
) -> dict[str, Any]:
    ops_manifest_path = Path(ops_path).expanduser()
    if DEFAULT_CONV_EVIDENCE_MANIFEST_PATH.is_file() and (
        str(model_key).strip().lower() == "mobilevit_s"
        or ops_manifest_path.name.strip().lower() == "ops_mobilevit_s.json"
    ):
        payload = _load_json(DEFAULT_CONV_EVIDENCE_MANIFEST_PATH)
        if payload is not None:
            return {
                "manifest_path": str(DEFAULT_CONV_EVIDENCE_MANIFEST_PATH.resolve()),
                "manifest_sha256": _sha256_path(DEFAULT_CONV_EVIDENCE_MANIFEST_PATH),
                "manifest": payload,
            }
    payload = build_conv_evidence_manifest(model_key=model_key, ops_path=ops_manifest_path)
    return {
        "manifest_path": "",
        "manifest_sha256": hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "manifest": payload,
    }


def build_conv_focused_measured_package_skeleton(
    manifest_payload: dict[str, Any],
) -> dict[str, Any]:
    target_set = (
        ((manifest_payload.get("conv_focused_target_set") or {}).get("target_module_keys"))
        or []
    )
    target_set_sha256 = str(
        ((manifest_payload.get("conv_focused_target_set") or {}).get("target_set_sha256"))
        or ""
    )
    return {
        "schema_version": CONV_FOCUSED_MEASURED_PACKAGE_SCHEMA_VERSION,
        "created_date": "2026-04-18",
        "status": CONV_FOCUSED_MEASURED_PACKAGE_STATUS_DRY_RUN,
        "scope": {
            "model": str(manifest_payload.get("model") or ""),
            "closure_target": "strong_conv_hardware_style_closure",
            "selection_rule_version": "conv_focused_target_selection_v1",
        },
        "governance": {
            "authorizes_future_measured_run": False,
            "benchmark_claim_ready": False,
            "creates_measured_evidence": False,
            "pilot_bridge_restamp_promotable": False,
            "promotes_claim": False,
        },
        "future_package_requirements": {
            "requires_new_run_id": True,
            "requires_new_authorization_note": True,
            "requires_new_eligibility_check": True,
            "requires_new_phase1_replay": True,
            "requires_new_measured_source_csv": True,
            "legacy_all_target_row_forbidden": True,
        },
        "target_set": {
            "target_module_keys": target_set,
            "target_module_key_count": len(target_set),
            "target_set_sha256": target_set_sha256,
        },
        "evidence_bindings": {
            "conv_evidence_manifest_path": str(DEFAULT_CONV_EVIDENCE_MANIFEST_PATH.resolve()),
            "conv_evidence_manifest_sha256": (
                _sha256_path(DEFAULT_CONV_EVIDENCE_MANIFEST_PATH)
                if DEFAULT_CONV_EVIDENCE_MANIFEST_PATH.is_file()
                else ""
            ),
        },
    }


def resolve_conv_focused_measured_package() -> dict[str, Any]:
    if DEFAULT_CONV_FOCUSED_MEASURED_PACKAGE_PATH.is_file():
        payload = _load_json(DEFAULT_CONV_FOCUSED_MEASURED_PACKAGE_PATH)
        if payload is not None:
            return {
                "package_path": str(DEFAULT_CONV_FOCUSED_MEASURED_PACKAGE_PATH.resolve()),
                "package_sha256": _sha256_path(DEFAULT_CONV_FOCUSED_MEASURED_PACKAGE_PATH),
                "package": payload,
            }
    manifest = resolve_conv_evidence_manifest(
        model_key="mobilevit_s",
        ops_path=ROOT_DIR / "experiments" / "mtl_model" / "ops" / "ops_mobilevit_s.json",
    )
    payload = build_conv_focused_measured_package_skeleton(manifest["manifest"])
    return {
        "package_path": "",
        "package_sha256": hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "package": payload,
    }


def validate_conv_focused_measured_package(
    *,
    conv_evidence_manifest: dict[str, Any],
    conv_focused_measured_package: dict[str, Any],
) -> dict[str, Any]:
    manifest_payload = conv_evidence_manifest.get("manifest") or {}
    package_payload = conv_focused_measured_package.get("package") or {}
    package_target = package_payload.get("target_set") or {}
    package_bindings = package_payload.get("evidence_bindings") or {}
    measured_evidence = package_payload.get("measured_evidence") or {}

    expected_target_set_sha256 = str(
        ((manifest_payload.get("conv_focused_target_set") or {}).get("target_set_sha256")) or ""
    ).strip()
    expected_runtime_target_keys = conv_focused_runtime_target_module_keys(manifest_payload)
    expected_manifest_path = str(conv_evidence_manifest.get("manifest_path") or "").strip()
    expected_manifest_sha256 = str(conv_evidence_manifest.get("manifest_sha256") or "").strip()
    declared_status = str(package_payload.get("status") or "").strip()
    declared_target_set_sha256 = str(package_target.get("target_set_sha256") or "").strip()
    declared_manifest_path = str(
        package_bindings.get("conv_evidence_manifest_path") or ""
    ).strip()
    declared_manifest_sha256 = str(
        package_bindings.get("conv_evidence_manifest_sha256") or ""
    ).strip()

    blockers: list[str] = []
    if (
        str(package_payload.get("schema_version") or "").strip()
        != CONV_FOCUSED_MEASURED_PACKAGE_SCHEMA_VERSION
    ):
        blockers.append("conv_measured_package_schema_invalid")
    if expected_manifest_path and declared_manifest_path != expected_manifest_path:
        blockers.append("conv_measured_package_manifest_binding_mismatch")
    if expected_manifest_sha256 and declared_manifest_sha256 != expected_manifest_sha256:
        blockers.append("conv_measured_package_manifest_sha_mismatch")
    if expected_target_set_sha256 and declared_target_set_sha256 != expected_target_set_sha256:
        blockers.append("conv_measured_package_target_set_mismatch")
    if declared_status != CONV_FOCUSED_MEASURED_PACKAGE_STATUS_AUTHORIZED:
        blockers.append("conv_measured_package_not_authorized")
    measured_run_id = str(measured_evidence.get("run_id") or "").strip()
    measured_truth_class = str(
        measured_evidence.get("bitstream_measurement_truth_class") or ""
    ).strip()
    measured_claim_surface_status = str(
        measured_evidence.get("bitstream_runtime_claim_surface_status") or ""
    ).strip()
    measured_authorization_status = str(
        measured_evidence.get("bitstream_truth_class_authorization_status") or ""
    ).strip()
    measured_results_csv_path = str(measured_evidence.get("results_csv_path") or "").strip()
    measured_active_target_keys = measured_evidence.get(
        "bitstream_runtime_active_target_module_keys"
    )
    if not isinstance(measured_active_target_keys, list):
        measured_active_target_keys = []
    measured_active_target_keys = [
        str(item).strip() for item in measured_active_target_keys if str(item).strip()
    ]
    measured_active_target_count = _safe_int(
        measured_evidence.get("bitstream_runtime_active_target_module_count"),
        len(measured_active_target_keys),
    )
    if declared_status == CONV_FOCUSED_MEASURED_PACKAGE_STATUS_AUTHORIZED:
        if not measured_results_csv_path or not Path(measured_results_csv_path).is_file():
            blockers.append("conv_measured_package_results_csv_missing")
        if not measured_run_id:
            blockers.append("conv_measured_package_run_id_missing")
        elif measured_run_id == LEGACY_ALL_TARGET_MEASURED_ANCHOR_RUN_ID:
            blockers.append("legacy_all_target_row_not_authorized_for_conv_closure")
        if measured_truth_class != BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS:
            blockers.append("conv_measured_package_truth_class_unsatisfied")
        if measured_claim_surface_status != CONV_FOCUSED_CLAIM_SURFACE_STATUS:
            blockers.append("conv_measured_package_claim_surface_unsatisfied")
        if measured_authorization_status != "authorized":
            blockers.append("conv_measured_package_authorization_unsatisfied")
        if measured_active_target_count != len(expected_runtime_target_keys):
            blockers.append("conv_measured_package_active_target_count_mismatch")
        if sorted(measured_active_target_keys) != sorted(expected_runtime_target_keys):
            blockers.append("conv_measured_package_active_target_set_mismatch")

    deduped_blockers: list[str] = []
    for blocker in blockers:
        if blocker not in deduped_blockers:
            deduped_blockers.append(blocker)
    return {
        "package_ready": not deduped_blockers,
        "package_status": declared_status,
        "package_path": str(conv_focused_measured_package.get("package_path") or ""),
        "package_sha256": str(conv_focused_measured_package.get("package_sha256") or ""),
        "measured_run_id": measured_run_id,
        "package_blockers": deduped_blockers,
    }


__all__ = [
    "CONV_EVIDENCE_MANIFEST_SCHEMA_VERSION",
    "CONV_FIDELITY_STAGE_HARDWARE_EVIDENCE_CLOSED",
    "CONV_FIDELITY_STAGE_MEASURED_CLOSED",
    "CONV_FIDELITY_STAGE_RUNTIME_MODELED",
    "CONV_FOCUSED_CLAIM_SURFACE_STATUS",
    "CONV_FOCUSED_MEASURED_PACKAGE_SCHEMA_VERSION",
    "CONV_FOCUSED_MEASURED_PACKAGE_STATUS_AUTHORIZED",
    "CONV_FOCUSED_MEASURED_PACKAGE_STATUS_DRY_RUN",
    "CONV_HARDWARE_EVIDENCE_UNCLOSED",
    "CONV_MEASURED_CLOSURE_STATUS_MEASURED_CLOSED",
    "CONV_MEASURED_CLOSURE_STATUS_RUNTIME_MODELED",
    "CONV_MEASURED_CLOSURE_STATUS_UNCLOSED",
    "CONV_NATIVE_CLASS_DEPTHWISE_GROUPED_PATCH_DOMINANT",
    "CONV_NATIVE_CLASS_POINTWISE_GEMM_ALIGNED",
    "CONV_NATIVE_CLASS_STANDARD_SPATIAL_FILTER_BANK_VISIBLE",
    "DEFAULT_CONV_EVIDENCE_MANIFEST_PATH",
    "DEFAULT_CONV_FOCUSED_MEASURED_PACKAGE_PATH",
    "LEGACY_ALL_TARGET_MEASURED_ANCHOR_RUN_ID",
    "build_conv_evidence_manifest",
    "build_conv_focused_measured_package_skeleton",
    "build_conv_runtime_semantics",
    "classify_conv_native_class",
    "conv_focused_runtime_target_module_keys",
    "conv_fidelity_blockers_for_stage",
    "normalize_conv_target_module_key_for_runtime",
    "resolve_conv_evidence_manifest",
    "resolve_conv_focused_measured_package",
    "validate_conv_focused_measured_package",
]
