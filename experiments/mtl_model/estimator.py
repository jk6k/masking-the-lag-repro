"""MTL energy/latency estimator for photonic GEMM and electronic ops.

This module is intentionally self-contained for MTL experiments.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

from accuracy.bitstream_conv_semantics import (
    CONV_FIDELITY_STAGE_MEASURED_CLOSED,
    CONV_FIDELITY_STAGE_RUNTIME_MODELED,
    CONV_NATIVE_CLASS_DEPTHWISE_GROUPED_PATCH_DOMINANT,
    CONV_NATIVE_CLASS_POINTWISE_GEMM_ALIGNED,
    CONV_NATIVE_CLASS_STANDARD_SPATIAL_FILTER_BANK_VISIBLE,
    build_conv_runtime_semantics,
    classify_conv_native_class,
    conv_fidelity_blockers_for_stage,
)
from accuracy.mlx_mobilevit import DEFAULT_BITSTREAM_RUNTIME_STREAM_REUSE_POLICY
from sc_bitstream.generators import resolve_generator_stream_state_policy

ELEMENTWISE_TYPES = {"softmax", "norm", "layer_norm", "batch_norm", "group_norm", "activation"}
BITSTREAM_NATIVE_NORMALIZATION_TYPES = {"norm", "layer_norm", "batch_norm", "group_norm"}
BITSTREAM_NATIVE_ELEMENTWISE_TYPES = BITSTREAM_NATIVE_NORMALIZATION_TYPES | {"activation"}
BITSTREAM_WORKLOAD_SUPPORT_QUALIFIED_TYPES = {"softmax"}

OP_FAMILY_ELEMENTWISE = "elementwise"
OP_FAMILY_GEMM_LIKE = "gemm_like"

MODEL_ABSTRACTION_KIND_CONV2D_GEMM_LOWERED_APPROXIMATION = (
    "conv2d_gemm_lowered_approximation"
)
MODEL_ABSTRACTION_KIND_CONV2D_NATIVE_RUNTIME_PATH_MODELED = (
    "conv2d_native_runtime_path_modeled"
)
MODEL_ABSTRACTION_KIND_GEMM_LIKE_SHAPE_PROXY = "gemm_like_shape_proxy"
MODEL_ABSTRACTION_KIND_NATIVE_ELECTRONIC_BITSTREAM = "native_electronic_bitstream"
MODEL_ABSTRACTION_KIND_GOVERNED_ELECTRONIC_SUPPORT_PROXY = (
    "governed_electronic_support_proxy"
)
MODEL_ABSTRACTION_KIND_OUT_OF_SURFACE_PROXY = "out_of_surface_proxy"

MODEL_ABSTRACTION_STATUS_APPROXIMATE = "approximate"
MODEL_ABSTRACTION_STATUS_NATIVE = "native"
MODEL_ABSTRACTION_STATUS_NATIVE_RUNTIME_MODELED = "native_runtime_modeled"
MODEL_ABSTRACTION_STATUS_NATIVE_MEASURED_CLOSED = "native_measured_closed"
MODEL_ABSTRACTION_STATUS_SUPPORT_ONLY = "support_only"
MODEL_ABSTRACTION_STATUS_OUT_OF_SURFACE = "out_of_surface"

SC_NATIVE_BITSTREAM = "sc_native_bitstream"
SC_GOVERNED_ELECTRONIC_SUPPORT = "sc_governed_electronic_support"
SC_DEFAULT_UNSUPPORTED = "sc_default_unsupported"

TRUE_SC_NATIVE = "true_sc_native"
TRUE_SC_OUT_OF_SURFACE = "true_sc_out_of_surface"
GOVERNED_SUPPORT_NOT_TRUE_SC = "governed_support_not_true_sc"
TRUE_SC_NATIVE_BLOCKED_BY_WORKLOAD_FIDELITY = "true_sc_native_blocked_by_workload_fidelity"

TRUE_SC_CLAIM_SURFACE_IN = "in_claim_surface"
TRUE_SC_CLAIM_SURFACE_SUPPORT_OUT = "support_out_of_claim_surface"
TRUE_SC_CLAIM_SURFACE_OUT = "out_of_claim_surface"
TRUE_SC_CLAIM_SURFACE_FULL = "full_true_sc_claim_surface"
TRUE_SC_CLAIM_SURFACE_LIMITED = "limited_true_sc_surface_with_out_of_surface_support"
TRUE_SC_CLAIM_SURFACE_BLOCKED = "claim_surface_blocked_by_governed_support"
TRUE_SC_CLAIM_SURFACE_EMPTY = "no_true_sc_claim_surface"
TRUE_SC_CLAIM_SURFACE_FIDELITY_BLOCKED = "claim_surface_blocked_by_workload_fidelity"

TRUSTED_DEFAULT = "trusted_default"
DEFAULT_WITH_SUPPORTING_ASSUMPTIONS = "default_with_supporting_assumptions"
OUT_OF_BAND = "out_of_band"

ESTIMATION_MODEL_COMPLETE = "complete_estimation_model"
ESTIMATION_MODEL_INCOMPLETE = "incomplete_estimation_model"
ESTIMATION_SUPPORT_BOUNDARY_NATIVE_ONLY = "native_only"
ESTIMATION_SUPPORT_BOUNDARY_NATIVE_PLUS_GOVERNED = "native_plus_governed_support"

WORKLOAD_FIDELITY_CLASS_NATIVE = "native_workload_fidelity"
WORKLOAD_FIDELITY_CLASS_APPROXIMATE = "approximate_workload_fidelity"
WORKLOAD_FIDELITY_CLASS_SUPPORT_QUALIFIED = "support_qualified_workload_fidelity"
WORKLOAD_FIDELITY_CLASS_OUT_OF_SURFACE = "out_of_surface_workload_fidelity"

WORKLOAD_FIDELITY_STATUS_NATIVE_READY = "native_ready"
WORKLOAD_FIDELITY_STATUS_APPROXIMATE = "approximate"
WORKLOAD_FIDELITY_STATUS_SUPPORT_ONLY = "support_only"
WORKLOAD_FIDELITY_STATUS_SUPPORT_QUALIFIED_READY = "support_qualified_ready"
WORKLOAD_FIDELITY_STATUS_OUT_OF_SURFACE = "out_of_surface"

WORKLOAD_CLAIM_ROLE_NATIVE = "workload_native"
WORKLOAD_CLAIM_ROLE_SUPPORT_QUALIFIED = "workload_support_qualified"
WORKLOAD_CLAIM_ROLE_BLOCKED_BY_SUPPORT = "workload_support_blocked_by_support"
WORKLOAD_CLAIM_ROLE_OUT = "workload_out_of_surface"


@dataclass(frozen=True)
class BitstreamEstimatorConfig:
    enabled: bool
    execution_semantics: str
    encoding_mode: str
    multiplier_mode: str
    accumulator_mode: str
    stream_length: int
    generator: str
    calibration_source: str | None
    capture_manifest_csv: str | None
    sample_rate_gsps: float | None
    parallel_outputs: int
    cycles_per_stream_bit: int
    accumulator_energy_pj: float
    effective_stream_length_scale: float
    calibration_applied: bool
    calibration_summary_json: str | None
    calibration_capture_row_count: int
    calibration_replay_row_count: int
    calibration_module_count: int
    calibration_median_abs_error: float | None
    calibration_max_abs_error: float | None
    calibration_median_relative_error: float | None
    calibration_reason: str | None
    stream_state_policy: dict[str, object]
    parallel_outputs_provenance: str
    cycles_per_stream_bit_provenance: str
    accumulator_energy_pj_provenance: str
    effective_stream_length_scale_provenance: str
    norm_parallelism: int
    norm_parallelism_provenance: str
    activation_parallelism: int
    activation_parallelism_provenance: str


def _ceil_div(a: int, b: int) -> int:
    if b == 0:
        return 0
    return int(math.ceil(a / b))


def _power_to_energy_j(power_mw: float | None, rate_gsps: float | None) -> float | None:
    if power_mw is None or rate_gsps is None or rate_gsps == 0:
        return None
    power_w = power_mw / 1000.0
    rate_hz = rate_gsps * 1e9
    return power_w / rate_hz


def _get_with_fallback(cfg: dict, key: str, fallback: object) -> object:
    value = cfg.get(key)
    if value is None:
        return fallback
    return value


def _to_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_int_scalar_with_provenance(
    *,
    primary_cfg: dict[str, object],
    primary_key: str,
    primary_default: int,
    primary_provenance: str,
    secondary_cfg: dict[str, object] | None = None,
    secondary_key: str | None = None,
    secondary_default: int | None = None,
    secondary_provenance: str | None = None,
) -> tuple[int, str]:
    if primary_cfg.get(primary_key) not in {None, ""}:
        return max(1, _safe_int(primary_cfg.get(primary_key), primary_default)), primary_provenance
    if (
        secondary_cfg is not None
        and secondary_key is not None
        and secondary_cfg.get(secondary_key) not in {None, ""}
    ):
        fallback_default = primary_default if secondary_default is None else secondary_default
        provenance = secondary_provenance or "implicit_default"
        return max(1, _safe_int(secondary_cfg.get(secondary_key), fallback_default)), provenance
    return max(1, int(primary_default)), "implicit_default"


def _resolve_float_scalar_with_provenance(
    *,
    primary_cfg: dict[str, object],
    primary_key: str,
    primary_default: float,
    primary_provenance: str,
    secondary_cfg: dict[str, object] | None = None,
    secondary_key: str | None = None,
    secondary_default: float | None = None,
    secondary_provenance: str | None = None,
) -> tuple[float, str]:
    if primary_cfg.get(primary_key) not in {None, ""}:
        return _safe_float(primary_cfg.get(primary_key), primary_default), primary_provenance
    if (
        secondary_cfg is not None
        and secondary_key is not None
        and secondary_cfg.get(secondary_key) not in {None, ""}
    ):
        fallback_default = primary_default if secondary_default is None else secondary_default
        provenance = secondary_provenance or "implicit_default"
        return _safe_float(secondary_cfg.get(secondary_key), fallback_default), provenance
    return float(primary_default), "implicit_default"


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _op_family(op_type: str) -> str:
    return OP_FAMILY_ELEMENTWISE if op_type in ELEMENTWISE_TYPES else OP_FAMILY_GEMM_LIKE


def _is_bitstream_native_elementwise(op_type: str, estimator_mode: str) -> bool:
    return estimator_mode == "bitstream" and op_type in BITSTREAM_NATIVE_ELEMENTWISE_TYPES


def _bitstream_native_elementwise_claim_reason(op_type: str) -> str:
    if op_type in BITSTREAM_NATIVE_NORMALIZATION_TYPES:
        return "normalization_family_bitstream_native"
    return "activation_family_bitstream_native"


def _is_workload_support_qualified(op_type: str, estimator_mode: str) -> bool:
    return estimator_mode == "bitstream" and op_type in BITSTREAM_WORKLOAD_SUPPORT_QUALIFIED_TYPES


def _support_class(op_type: str, estimator_mode: str) -> str:
    if op_type in ELEMENTWISE_TYPES:
        if _is_bitstream_native_elementwise(op_type, estimator_mode):
            return SC_NATIVE_BITSTREAM
        return SC_GOVERNED_ELECTRONIC_SUPPORT
    if estimator_mode == "bitstream":
        return SC_NATIVE_BITSTREAM
    return SC_DEFAULT_UNSUPPORTED


def _trust_posture(estimator_mode: str, support_class: str) -> str:
    if estimator_mode != "bitstream":
        return OUT_OF_BAND
    if support_class == SC_NATIVE_BITSTREAM:
        return TRUSTED_DEFAULT
    if support_class == SC_GOVERNED_ELECTRONIC_SUPPORT:
        return DEFAULT_WITH_SUPPORTING_ASSUMPTIONS
    return OUT_OF_BAND


def _true_sc_claim_state(op_type: str, estimator_mode: str) -> str:
    if op_type in ELEMENTWISE_TYPES:
        if _is_bitstream_native_elementwise(op_type, estimator_mode):
            return TRUE_SC_NATIVE
        return GOVERNED_SUPPORT_NOT_TRUE_SC
    if estimator_mode == "bitstream":
        return TRUE_SC_NATIVE
    return TRUE_SC_OUT_OF_SURFACE


def _true_sc_claim_surface_metadata(
    op_type: str,
    estimator_mode: str,
    support_class: str,
) -> dict[str, str]:
    if op_type in ELEMENTWISE_TYPES:
        if _is_bitstream_native_elementwise(op_type, estimator_mode):
            return {
                "true_sc_claim_surface_role": TRUE_SC_CLAIM_SURFACE_IN,
                "true_sc_claim_surface_reason": _bitstream_native_elementwise_claim_reason(
                    op_type
                ),
            }
        return {
            "true_sc_claim_surface_role": TRUE_SC_CLAIM_SURFACE_SUPPORT_OUT,
            "true_sc_claim_surface_reason": "blocked_pending_native_model",
        }
    if estimator_mode == "bitstream" and support_class == SC_NATIVE_BITSTREAM:
        return {
            "true_sc_claim_surface_role": TRUE_SC_CLAIM_SURFACE_IN,
            "true_sc_claim_surface_reason": "gemm_like_bitstream_native",
        }
    return {
        "true_sc_claim_surface_role": TRUE_SC_CLAIM_SURFACE_OUT,
        "true_sc_claim_surface_reason": "proxy_or_unsupported_not_claimed_true_sc",
    }


def _workload_claim_metadata(
    op_type: str,
    estimator_mode: str,
    support_class: str,
) -> dict[str, object]:
    if support_class == SC_NATIVE_BITSTREAM:
        return {
            "workload_claim_role": WORKLOAD_CLAIM_ROLE_NATIVE,
            "workload_claim_reason": "native_bitstream_workload_model",
            "workload_claim_blocks_native": False,
        }
    if _is_workload_support_qualified(op_type, estimator_mode):
        return {
            "workload_claim_role": WORKLOAD_CLAIM_ROLE_SUPPORT_QUALIFIED,
            "workload_claim_reason": "declared_runtime_support_policy",
            "workload_claim_blocks_native": False,
        }
    if support_class == SC_GOVERNED_ELECTRONIC_SUPPORT:
        return {
            "workload_claim_role": WORKLOAD_CLAIM_ROLE_BLOCKED_BY_SUPPORT,
            "workload_claim_reason": (
                "governed_support_requires_bitstream_runtime_policy"
                if estimator_mode != "bitstream"
                else "governed_support_not_workload_native"
            ),
            "workload_claim_blocks_native": True,
        }
    return {
        "workload_claim_role": WORKLOAD_CLAIM_ROLE_OUT,
        "workload_claim_reason": "operator_model_out_of_surface",
        "workload_claim_blocks_native": True,
    }


def _new_model_abstraction_boundary_inventory() -> dict[str, int]:
    return {
        MODEL_ABSTRACTION_KIND_CONV2D_GEMM_LOWERED_APPROXIMATION: 0,
        MODEL_ABSTRACTION_KIND_CONV2D_NATIVE_RUNTIME_PATH_MODELED: 0,
        MODEL_ABSTRACTION_KIND_GEMM_LIKE_SHAPE_PROXY: 0,
        MODEL_ABSTRACTION_KIND_NATIVE_ELECTRONIC_BITSTREAM: 0,
        MODEL_ABSTRACTION_KIND_GOVERNED_ELECTRONIC_SUPPORT_PROXY: 0,
        MODEL_ABSTRACTION_KIND_OUT_OF_SURFACE_PROXY: 0,
    }


def _model_abstraction_boundary_metadata(
    op: dict[str, object],
    estimator_mode: str,
) -> dict[str, object]:
    op_type = str(op.get("type") or "gemm").strip().lower() or "gemm"
    boundary: dict[str, object]

    if op_type == "conv2d":
        kernel = op.get("kernel")
        stride = op.get("stride")
        groups = op.get("groups")
        conv_native_class = classify_conv_native_class(op)
        conv_runtime_semantics = build_conv_runtime_semantics(
            op,
            runtime_stream_reuse_policy=DEFAULT_BITSTREAM_RUNTIME_STREAM_REUSE_POLICY,
        )
        boundary = {
            "op_type": op_type,
            "model_abstraction_kind": MODEL_ABSTRACTION_KIND_CONV2D_NATIVE_RUNTIME_PATH_MODELED,
            "model_abstraction_status": MODEL_ABSTRACTION_STATUS_NATIVE_RUNTIME_MODELED,
            "modeled_as": "conv2d_native_runtime_path",
            "fidelity_claim": "conv_native_runtime_modeled_hardware_evidence_unclosed",
            "consumed_cost_fields": ["m", "d", "n"],
            "captured_manifest_fields": {
                "kernel": kernel,
                "stride": stride,
                "groups": groups,
            },
            "captured_but_not_fully_consumed_fields": [],
            "omitted_cost_terms": [],
            "conv_native_class": conv_native_class,
            "conv_fidelity_stage": CONV_FIDELITY_STAGE_RUNTIME_MODELED,
            "conv_fidelity_blockers": conv_fidelity_blockers_for_stage(
                CONV_FIDELITY_STAGE_RUNTIME_MODELED
            ),
            "conv_runtime_semantics": conv_runtime_semantics,
            "cost_model_backend": (
                "bitstream_conv2d_native_estimator"
                if estimator_mode == "bitstream"
                else "proxy_conv2d_reference_cost"
            ),
            "boundary_reason": (
                "conv2d is classified with a native runtime-path contract and "
                "explicit runtime semantics, but hardware evidence remains "
                "unclosed until a separate conv-focused measured package "
                "satisfies the fidelity contract"
            ),
        }
        return {
            "model_abstraction_kind": boundary["model_abstraction_kind"],
            "model_abstraction_status": boundary["model_abstraction_status"],
            "model_abstraction_reason": boundary["boundary_reason"],
            "conv_lowering_kernel": kernel,
            "conv_lowering_stride": stride,
            "conv_lowering_groups": groups,
            "conv_native_class": conv_native_class,
            "conv_fidelity_stage": CONV_FIDELITY_STAGE_RUNTIME_MODELED,
            "conv_fidelity_blockers": json.dumps(
                conv_fidelity_blockers_for_stage(CONV_FIDELITY_STAGE_RUNTIME_MODELED),
                ensure_ascii=False,
            ),
            "conv_runtime_semantics_json": json.dumps(
                conv_runtime_semantics,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "model_abstraction_boundary_json": json.dumps(
                boundary,
                ensure_ascii=False,
                sort_keys=True,
            ),
        }

    if op_type in {"gemm", "linear"}:
        boundary = {
            "op_type": op_type,
            "model_abstraction_kind": MODEL_ABSTRACTION_KIND_GEMM_LIKE_SHAPE_PROXY,
            "model_abstraction_status": MODEL_ABSTRACTION_STATUS_APPROXIMATE,
            "modeled_as": "gemm_like",
            "fidelity_claim": "shape_only_proxy",
            "consumed_cost_fields": ["m", "d", "n"],
            "captured_manifest_fields": {},
            "captured_but_not_fully_consumed_fields": [],
            "omitted_cost_terms": [
                "lowering_overhead",
                "dataflow_overhead",
            ],
            "boundary_reason": (
                "linear/gemm-like operator is represented by a shape-only GEMM proxy"
            ),
        }
        return {
            "model_abstraction_kind": boundary["model_abstraction_kind"],
            "model_abstraction_status": boundary["model_abstraction_status"],
            "model_abstraction_reason": boundary["boundary_reason"],
            "conv_lowering_kernel": None,
            "conv_lowering_stride": None,
            "conv_lowering_groups": None,
            "conv_native_class": None,
            "conv_fidelity_stage": None,
            "conv_fidelity_blockers": json.dumps([], ensure_ascii=False),
            "conv_runtime_semantics_json": "",
            "model_abstraction_boundary_json": json.dumps(
                boundary,
                ensure_ascii=False,
                sort_keys=True,
            ),
        }

    if op_type in ELEMENTWISE_TYPES:
        if _is_bitstream_native_elementwise(op_type, estimator_mode):
            kind = MODEL_ABSTRACTION_KIND_NATIVE_ELECTRONIC_BITSTREAM
            status = MODEL_ABSTRACTION_STATUS_NATIVE
            reason = _bitstream_native_elementwise_claim_reason(op_type)
        else:
            kind = MODEL_ABSTRACTION_KIND_GOVERNED_ELECTRONIC_SUPPORT_PROXY
            status = MODEL_ABSTRACTION_STATUS_SUPPORT_ONLY
            reason = "governed_electronic_support_only"
        boundary = {
            "op_type": op_type,
            "model_abstraction_kind": kind,
            "model_abstraction_status": status,
            "modeled_as": "electronic_elementwise",
            "fidelity_claim": (
                "native_bitstream_electronic"
                if status == MODEL_ABSTRACTION_STATUS_NATIVE
                else "support_only_proxy"
            ),
            "consumed_cost_fields": ["elements"],
            "captured_manifest_fields": {},
            "captured_but_not_fully_consumed_fields": [],
            "omitted_cost_terms": [],
            "boundary_reason": reason,
        }
        return {
            "model_abstraction_kind": boundary["model_abstraction_kind"],
            "model_abstraction_status": boundary["model_abstraction_status"],
            "model_abstraction_reason": boundary["boundary_reason"],
            "conv_lowering_kernel": None,
            "conv_lowering_stride": None,
            "conv_lowering_groups": None,
            "conv_native_class": None,
            "conv_fidelity_stage": None,
            "conv_fidelity_blockers": json.dumps([], ensure_ascii=False),
            "conv_runtime_semantics_json": "",
            "model_abstraction_boundary_json": json.dumps(
                boundary,
                ensure_ascii=False,
                sort_keys=True,
            ),
        }

    boundary = {
        "op_type": op_type,
        "model_abstraction_kind": MODEL_ABSTRACTION_KIND_OUT_OF_SURFACE_PROXY,
        "model_abstraction_status": MODEL_ABSTRACTION_STATUS_OUT_OF_SURFACE,
        "modeled_as": "unsupported_proxy",
        "fidelity_claim": "out_of_surface",
        "consumed_cost_fields": [],
        "captured_manifest_fields": {},
        "captured_but_not_fully_consumed_fields": [],
        "omitted_cost_terms": [],
        "boundary_reason": "operator is outside the supported estimator surface",
    }
    return {
        "model_abstraction_kind": boundary["model_abstraction_kind"],
        "model_abstraction_status": boundary["model_abstraction_status"],
        "model_abstraction_reason": boundary["boundary_reason"],
        "conv_lowering_kernel": None,
        "conv_lowering_stride": None,
        "conv_lowering_groups": None,
        "conv_native_class": None,
        "conv_fidelity_stage": None,
        "conv_fidelity_blockers": json.dumps([], ensure_ascii=False),
        "conv_runtime_semantics_json": "",
        "model_abstraction_boundary_json": json.dumps(
            boundary,
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def _summarize_model_abstraction_boundary(
    ops: list[dict[str, object]],
    *,
    estimator_mode: str,
) -> dict[str, object]:
    inventory = _new_model_abstraction_boundary_inventory()
    boundary_kinds: list[str] = []
    conv2d_count = 0
    conv2d_native_count = 0
    gemm_like_count = 0

    for op in ops:
        metadata = _model_abstraction_boundary_metadata(op, estimator_mode)
        kind = str(metadata.get("model_abstraction_kind") or MODEL_ABSTRACTION_KIND_OUT_OF_SURFACE_PROXY)
        boundary_kinds.append(kind)
        if kind in inventory:
            inventory[kind] += 1
        if kind == MODEL_ABSTRACTION_KIND_CONV2D_GEMM_LOWERED_APPROXIMATION:
            conv2d_count += 1
        elif kind == MODEL_ABSTRACTION_KIND_CONV2D_NATIVE_RUNTIME_PATH_MODELED:
            conv2d_native_count += 1
        elif kind == MODEL_ABSTRACTION_KIND_GEMM_LIKE_SHAPE_PROXY:
            gemm_like_count += 1

    if conv2d_native_count > 0:
        surface_kind = MODEL_ABSTRACTION_KIND_CONV2D_NATIVE_RUNTIME_PATH_MODELED
        surface_status = MODEL_ABSTRACTION_STATUS_NATIVE_RUNTIME_MODELED
        surface_reason = (
            "conv2d ops are classified as native runtime-path modeled with "
            "explicit runtime semantics, but hardware evidence remains unclosed"
        )
    elif conv2d_count > 0:
        surface_kind = MODEL_ABSTRACTION_KIND_CONV2D_GEMM_LOWERED_APPROXIMATION
        surface_status = MODEL_ABSTRACTION_STATUS_APPROXIMATE
        surface_reason = "legacy conv2d gemm-lowered approximation metadata present"
    elif gemm_like_count > 0:
        surface_kind = MODEL_ABSTRACTION_KIND_GEMM_LIKE_SHAPE_PROXY
        surface_status = MODEL_ABSTRACTION_STATUS_APPROXIMATE
        surface_reason = "workload is represented by a shape-only GEMM proxy"
    elif inventory[MODEL_ABSTRACTION_KIND_NATIVE_ELECTRONIC_BITSTREAM] > 0:
        surface_kind = MODEL_ABSTRACTION_KIND_NATIVE_ELECTRONIC_BITSTREAM
        surface_status = MODEL_ABSTRACTION_STATUS_NATIVE
        surface_reason = "native bitstream electronic support surface only"
    elif inventory[MODEL_ABSTRACTION_KIND_GOVERNED_ELECTRONIC_SUPPORT_PROXY] > 0:
        surface_kind = MODEL_ABSTRACTION_KIND_GOVERNED_ELECTRONIC_SUPPORT_PROXY
        surface_status = MODEL_ABSTRACTION_STATUS_SUPPORT_ONLY
        surface_reason = "governed electronic support only"
    else:
        surface_kind = MODEL_ABSTRACTION_KIND_OUT_OF_SURFACE_PROXY
        surface_status = MODEL_ABSTRACTION_STATUS_OUT_OF_SURFACE
        surface_reason = "no modeled surface present"

    summary_payload = {
        "model_abstraction_boundary_kind": surface_kind,
        "model_abstraction_boundary_status": surface_status,
        "model_abstraction_boundary_reason": surface_reason,
        "model_abstraction_boundary_inventory": inventory,
        "model_abstraction_boundary_kinds": boundary_kinds,
        "conv2d_gemm_lowered_approximation_op_count": conv2d_count,
        "conv2d_native_runtime_modeled_op_count": conv2d_native_count,
        "gemm_like_shape_proxy_op_count": gemm_like_count,
        "model_abstraction_boundary_json": json.dumps(
            {
                "model_abstraction_boundary_kind": surface_kind,
                "model_abstraction_boundary_status": surface_status,
                "model_abstraction_boundary_reason": surface_reason,
                "model_abstraction_boundary_inventory": inventory,
                "conv2d_gemm_lowered_approximation_op_count": conv2d_count,
                "conv2d_native_runtime_modeled_op_count": conv2d_native_count,
                "gemm_like_shape_proxy_op_count": gemm_like_count,
                "model_abstraction_boundary_kinds": boundary_kinds,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    return summary_payload


def _new_support_class_inventory() -> dict[str, int]:
    return {
        SC_NATIVE_BITSTREAM: 0,
        SC_GOVERNED_ELECTRONIC_SUPPORT: 0,
        SC_DEFAULT_UNSUPPORTED: 0,
    }


def _new_trust_posture_inventory() -> dict[str, int]:
    return {
        TRUSTED_DEFAULT: 0,
        DEFAULT_WITH_SUPPORTING_ASSUMPTIONS: 0,
        OUT_OF_BAND: 0,
    }


def _new_op_family_inventory() -> dict[str, int]:
    return {
        OP_FAMILY_GEMM_LIKE: 0,
        OP_FAMILY_ELEMENTWISE: 0,
    }


def _new_workload_claim_inventory() -> dict[str, int]:
    return {
        WORKLOAD_CLAIM_ROLE_NATIVE: 0,
        WORKLOAD_CLAIM_ROLE_SUPPORT_QUALIFIED: 0,
        WORKLOAD_CLAIM_ROLE_BLOCKED_BY_SUPPORT: 0,
        WORKLOAD_CLAIM_ROLE_OUT: 0,
    }


def _new_true_sc_claim_state_inventory() -> dict[str, int]:
    return {
        TRUE_SC_NATIVE: 0,
        TRUE_SC_OUT_OF_SURFACE: 0,
        GOVERNED_SUPPORT_NOT_TRUE_SC: 0,
    }


def _new_true_sc_claim_surface_inventory() -> dict[str, int]:
    return {
        TRUE_SC_CLAIM_SURFACE_IN: 0,
        TRUE_SC_CLAIM_SURFACE_SUPPORT_OUT: 0,
        TRUE_SC_CLAIM_SURFACE_OUT: 0,
    }


def _new_bitstream_datapath_stage_summary() -> dict[str, dict[str, float]]:
    return {
        "stream_generation_load": {"events": 0.0, "cycles": 0.0, "energy_j": 0.0},
        "stochastic_multiply": {"events": 0.0, "cycles": 0.0, "energy_j": 0.0},
        "accumulation_pca": {"events": 0.0, "cycles": 0.0, "energy_j": 0.0},
        "laser_clocking": {"events": 0.0, "cycles": 0.0, "energy_j": 0.0},
        "memory": {"events": 0.0, "cycles": 0.0, "energy_j": 0.0},
        "patch_generation_load": {"events": 0.0, "cycles": 0.0, "energy_j": 0.0},
        "filter_materialization_or_residency": {
            "events": 0.0,
            "cycles": 0.0,
            "energy_j": 0.0,
        },
        "psum_or_accumulation": {"events": 0.0, "cycles": 0.0, "energy_j": 0.0},
        "writeback_or_decode": {"events": 0.0, "cycles": 0.0, "energy_j": 0.0},
        "static": {"events": 0.0, "cycles": 0.0, "energy_j": 0.0},
        "serialization_passes": {"events": 0.0, "cycles": 0.0, "energy_j": 0.0},
    }


def _merge_bitstream_datapath_stage_summary(
    accumulator: dict[str, dict[str, float]],
    stages: dict[str, dict[str, float]],
) -> None:
    for stage_name, stage_payload in stages.items():
        target = accumulator.setdefault(
            stage_name,
            {"events": 0.0, "cycles": 0.0, "energy_j": 0.0},
        )
        for key in ("events", "cycles", "energy_j"):
            target[key] = float(target.get(key, 0.0)) + float(stage_payload.get(key, 0.0))


def _bump_inventory(inventory: dict[str, int], key: str) -> None:
    inventory[key] = int(inventory.get(key, 0)) + 1


def _row_support_metadata(op_type: str, estimator_mode: str) -> dict[str, object]:
    support_class = _support_class(op_type, estimator_mode)
    true_sc_claim_state = _true_sc_claim_state(op_type, estimator_mode)
    return {
        "op_family": _op_family(op_type),
        "support_class": support_class,
        "trust_posture": _trust_posture(estimator_mode, support_class),
        "true_sc_claim_state": true_sc_claim_state,
        **_true_sc_claim_surface_metadata(op_type, estimator_mode, support_class),
        **_workload_claim_metadata(op_type, estimator_mode, support_class),
    }


def _summary_trust_posture(trust_posture_inventory: dict[str, int]) -> str:
    if trust_posture_inventory.get(OUT_OF_BAND, 0) > 0:
        return OUT_OF_BAND
    if trust_posture_inventory.get(DEFAULT_WITH_SUPPORTING_ASSUMPTIONS, 0) > 0:
        return DEFAULT_WITH_SUPPORTING_ASSUMPTIONS
    return TRUSTED_DEFAULT


def _derive_workload_fidelity(
    *,
    model_abstraction_boundary: dict[str, object],
    support_class_inventory: dict[str, int],
    workload_claim_inventory: dict[str, int],
) -> dict[str, object]:
    boundary_kind = str(
        model_abstraction_boundary.get("model_abstraction_boundary_kind")
        or MODEL_ABSTRACTION_KIND_OUT_OF_SURFACE_PROXY
    )
    boundary_status = str(
        model_abstraction_boundary.get("model_abstraction_boundary_status")
        or MODEL_ABSTRACTION_STATUS_OUT_OF_SURFACE
    )
    blockers: list[str] = []
    workload_fidelity_class = WORKLOAD_FIDELITY_CLASS_NATIVE
    workload_fidelity_status = WORKLOAD_FIDELITY_STATUS_NATIVE_READY
    workload_fidelity_reason = "all_required_workload_models_are_native"

    if boundary_kind == MODEL_ABSTRACTION_KIND_CONV2D_NATIVE_RUNTIME_PATH_MODELED:
        workload_fidelity_class = WORKLOAD_FIDELITY_CLASS_APPROXIMATE
        workload_fidelity_status = WORKLOAD_FIDELITY_STATUS_APPROXIMATE
        blockers.append("conv_hardware_evidence_unclosed")
        workload_fidelity_reason = str(
            model_abstraction_boundary.get("model_abstraction_boundary_reason")
            or "conv_native_runtime_modeled_hardware_evidence_unclosed"
        )
    elif boundary_status == MODEL_ABSTRACTION_STATUS_APPROXIMATE:
        workload_fidelity_class = WORKLOAD_FIDELITY_CLASS_APPROXIMATE
        workload_fidelity_status = WORKLOAD_FIDELITY_STATUS_APPROXIMATE
        if boundary_kind == MODEL_ABSTRACTION_KIND_CONV2D_GEMM_LOWERED_APPROXIMATION:
            blockers.append("conv_model_not_native")
        elif boundary_kind == MODEL_ABSTRACTION_KIND_GEMM_LIKE_SHAPE_PROXY:
            blockers.append("gemm_shape_proxy_not_native")
        else:
            blockers.append("approximate_model_abstraction_present")
        workload_fidelity_reason = str(
            model_abstraction_boundary.get("model_abstraction_boundary_reason")
            or "approximate_model_abstraction_present"
        )

    blocking_support_count = int(
        workload_claim_inventory.get(WORKLOAD_CLAIM_ROLE_BLOCKED_BY_SUPPORT, 0)
    )
    support_qualified_count = int(
        workload_claim_inventory.get(WORKLOAD_CLAIM_ROLE_SUPPORT_QUALIFIED, 0)
    )
    if blocking_support_count > 0 and not blockers:
        workload_fidelity_class = WORKLOAD_FIDELITY_CLASS_SUPPORT_QUALIFIED
        workload_fidelity_status = WORKLOAD_FIDELITY_STATUS_SUPPORT_ONLY
        blockers.append("governed_support_not_workload_native")
        workload_fidelity_reason = "governed_support_ops_present"
    elif blocking_support_count > 0:
        blockers.append("governed_support_not_workload_native")
    elif support_qualified_count > 0 and not blockers:
        workload_fidelity_class = WORKLOAD_FIDELITY_CLASS_SUPPORT_QUALIFIED
        workload_fidelity_status = WORKLOAD_FIDELITY_STATUS_SUPPORT_QUALIFIED_READY
        workload_fidelity_reason = (
            "all_required_workload_models_are_native_or_runtime_support_qualified"
        )

    if boundary_status == MODEL_ABSTRACTION_STATUS_OUT_OF_SURFACE:
        workload_fidelity_class = WORKLOAD_FIDELITY_CLASS_OUT_OF_SURFACE
        workload_fidelity_status = WORKLOAD_FIDELITY_STATUS_OUT_OF_SURFACE
        blockers.append("operator_model_out_of_surface")
        workload_fidelity_reason = "required_operator_model_out_of_surface"

    deduped_blockers: list[str] = []
    seen: set[str] = set()
    for blocker in blockers:
        if blocker in seen:
            continue
        seen.add(blocker)
        deduped_blockers.append(blocker)

    return {
        "workload_fidelity_class": workload_fidelity_class,
        "workload_fidelity_status": workload_fidelity_status,
        "workload_fidelity_reason": workload_fidelity_reason,
        "workload_fidelity_blockers": deduped_blockers,
        "workload_native_claim_eligible": not deduped_blockers,
    }


def _summary_true_sc_claim_state(
    true_sc_claim_state_inventory: dict[str, int],
    *,
    workload_fidelity: dict[str, object] | None = None,
) -> str:
    if true_sc_claim_state_inventory.get(GOVERNED_SUPPORT_NOT_TRUE_SC, 0) > 0:
        return GOVERNED_SUPPORT_NOT_TRUE_SC
    if true_sc_claim_state_inventory.get(TRUE_SC_OUT_OF_SURFACE, 0) > 0:
        return TRUE_SC_OUT_OF_SURFACE
    if workload_fidelity and not bool(workload_fidelity.get("workload_native_claim_eligible")):
        return TRUE_SC_NATIVE_BLOCKED_BY_WORKLOAD_FIDELITY
    return TRUE_SC_NATIVE


def _summary_true_sc_claim_surface_status(
    *,
    native_op_count: int,
    governed_support_op_count: int,
    support_out_of_surface_op_count: int,
    out_of_claim_surface_op_count: int,
    workload_fidelity: dict[str, object] | None = None,
) -> str:
    if governed_support_op_count > 0:
        return TRUE_SC_CLAIM_SURFACE_BLOCKED
    if native_op_count <= 0:
        return TRUE_SC_CLAIM_SURFACE_EMPTY
    if support_out_of_surface_op_count > 0 or out_of_claim_surface_op_count > 0:
        return TRUE_SC_CLAIM_SURFACE_LIMITED
    if workload_fidelity and not bool(workload_fidelity.get("workload_native_claim_eligible")):
        return TRUE_SC_CLAIM_SURFACE_FIDELITY_BLOCKED
    return TRUE_SC_CLAIM_SURFACE_FULL


def _estimation_model_coverage_status(
    support_class_inventory: dict[str, int],
) -> str:
    if int(support_class_inventory.get(SC_DEFAULT_UNSUPPORTED, 0)) > 0:
        return ESTIMATION_MODEL_INCOMPLETE
    return ESTIMATION_MODEL_COMPLETE


def _estimation_model_support_boundary(
    support_class_inventory: dict[str, int],
) -> str:
    if int(support_class_inventory.get(SC_GOVERNED_ELECTRONIC_SUPPORT, 0)) > 0:
        return ESTIMATION_SUPPORT_BOUNDARY_NATIVE_PLUS_GOVERNED
    return ESTIMATION_SUPPORT_BOUNDARY_NATIVE_ONLY


def _update_true_sc_claim_surface_counts(
    *,
    metadata: dict[str, str],
    inventory: dict[str, int],
    counts: dict[str, int],
) -> None:
    role = metadata["true_sc_claim_surface_role"]
    _bump_inventory(inventory, role)
    if role == TRUE_SC_CLAIM_SURFACE_IN:
        if metadata["support_class"] == SC_NATIVE_BITSTREAM:
            counts["native"] += 1
        elif metadata["support_class"] == SC_GOVERNED_ELECTRONIC_SUPPORT:
            counts["governed_support"] += 1
        else:
            counts["out_of_surface"] += 1
    elif role == TRUE_SC_CLAIM_SURFACE_SUPPORT_OUT:
        counts["support_out_of_surface"] += 1
    else:
        counts["out_of_surface"] += 1


def _resolve_summary_json_path(
    *,
    capture_manifest_csv: str | None,
    calibration_source: str | None,
) -> Path | None:
    if calibration_source:
        path = Path(calibration_source).expanduser()
        return path if path.suffix.lower() == ".json" else None
    if not capture_manifest_csv:
        return None
    capture_path = Path(capture_manifest_csv).expanduser()
    if capture_path.suffix.lower() != ".csv":
        return None
    summary_name = capture_path.name.replace("_capture", "_summary", 1)
    if summary_name == capture_path.name:
        summary_name = f"{capture_path.stem}_summary.json"
    else:
        summary_name = Path(summary_name).with_suffix(".json").name
    return capture_path.with_name(summary_name)


def _module_role_complete(rows: list[dict[str, str]]) -> bool:
    roles = {str(row.get("operand_role") or "") for row in rows}
    has_lhs = any(role.startswith("lhs_") for role in roles)
    has_rhs = any(role.startswith("rhs_") for role in roles)
    has_output = any(role.startswith("output_") for role in roles)
    return has_lhs and has_rhs and has_output


def _read_capture_manifest_rows(path: Path) -> list[dict[str, str]] | None:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except OSError:
        return None
    required_fields = {
        "module_key",
        "operand_role",
        "stream_length",
        "generator",
        "encoding_mode",
        "multiplier_mode",
        "accumulator_mode",
    }
    fieldnames = set(reader.fieldnames or [])
    if not required_fields.issubset(fieldnames):
        return None
    return rows


def _resolve_bitstream_calibration(
    *,
    config: dict,
    resolved_cfg: BitstreamEstimatorConfig,
) -> BitstreamEstimatorConfig:
    capture_manifest_csv = resolved_cfg.capture_manifest_csv
    if not capture_manifest_csv:
        return replace(resolved_cfg, calibration_reason="capture_manifest_missing")

    capture_path = Path(capture_manifest_csv).expanduser()
    if not capture_path.is_file():
        return replace(resolved_cfg, calibration_reason="capture_manifest_unreadable")

    capture_rows = _read_capture_manifest_rows(capture_path)
    if capture_rows is None:
        return replace(resolved_cfg, calibration_reason="capture_manifest_invalid")

    matching_capture_rows = [
        row
        for row in capture_rows
        if str(row.get("generator") or "").strip().lower() == resolved_cfg.generator
        and _safe_int(row.get("stream_length"), 0) == resolved_cfg.stream_length
        and str(row.get("encoding_mode") or "").strip().lower() == resolved_cfg.encoding_mode
        and str(row.get("multiplier_mode") or "").strip().lower() == resolved_cfg.multiplier_mode
        and str(row.get("accumulator_mode") or "").strip().lower() == resolved_cfg.accumulator_mode
    ]
    if not matching_capture_rows:
        return replace(
            resolved_cfg,
            calibration_capture_row_count=len(capture_rows),
            calibration_reason="capture_manifest_mismatch",
        )

    grouped_capture_rows: dict[str, list[dict[str, str]]] = {}
    for row in matching_capture_rows:
        grouped_capture_rows.setdefault(str(row.get("module_key") or ""), []).append(row)
    complete_capture_modules = {
        module_key
        for module_key, module_rows in grouped_capture_rows.items()
        if module_key and _module_role_complete(module_rows)
    }
    if not complete_capture_modules:
        return replace(
            resolved_cfg,
            calibration_capture_row_count=len(matching_capture_rows),
            calibration_reason="capture_manifest_incomplete",
        )

    summary_path = _resolve_summary_json_path(
        capture_manifest_csv=capture_manifest_csv,
        calibration_source=resolved_cfg.calibration_source,
    )
    if summary_path is None or not summary_path.is_file():
        return replace(
            resolved_cfg,
            calibration_capture_row_count=len(matching_capture_rows),
            calibration_module_count=len(complete_capture_modules),
            calibration_reason="summary_json_missing",
        )

    try:
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return replace(
            resolved_cfg,
            calibration_capture_row_count=len(matching_capture_rows),
            calibration_module_count=len(complete_capture_modules),
            calibration_reason="summary_json_invalid",
        )

    summary_capture_csv = str(summary_payload.get("capture_csv") or "").strip()
    if summary_capture_csv and Path(summary_capture_csv).expanduser() != capture_path:
        return replace(
            resolved_cfg,
            calibration_capture_row_count=len(matching_capture_rows),
            calibration_module_count=len(complete_capture_modules),
            calibration_reason="summary_capture_mismatch",
        )

    semantics = summary_payload.get("semantics") or {}
    if (
        str(semantics.get("generator") or "").strip().lower() != resolved_cfg.generator
        or _safe_int(semantics.get("stream_length"), 0) != resolved_cfg.stream_length
        or str(semantics.get("encoding_mode") or "").strip().lower() != resolved_cfg.encoding_mode
        or str(semantics.get("multiplier_mode") or "").strip().lower() != resolved_cfg.multiplier_mode
        or str(semantics.get("accumulator_mode") or "").strip().lower()
        != resolved_cfg.accumulator_mode
    ):
        return replace(
            resolved_cfg,
            calibration_capture_row_count=len(matching_capture_rows),
            calibration_module_count=len(complete_capture_modules),
            calibration_reason="summary_semantics_mismatch",
        )

    replay_rows = summary_payload.get("replay")
    if not isinstance(replay_rows, list):
        return replace(
            resolved_cfg,
            calibration_capture_row_count=len(matching_capture_rows),
            calibration_module_count=len(complete_capture_modules),
            calibration_reason="summary_replay_missing",
        )

    matched_replay_rows = [
        row
        for row in replay_rows
        if isinstance(row, dict)
        and str(row.get("status") or "").strip().lower() == "replayed"
        and str(row.get("generator") or "").strip().lower() == resolved_cfg.generator
        and _safe_int(row.get("stream_length"), 0) == resolved_cfg.stream_length
        and str(row.get("module_key") or "") in complete_capture_modules
        and row.get("abs_error_vs_raw_exact") is not None
        and row.get("raw_exact_dot_product") is not None
    ]
    if not matched_replay_rows:
        return replace(
            resolved_cfg,
            calibration_summary_json=str(summary_path),
            calibration_capture_row_count=len(matching_capture_rows),
            calibration_module_count=len(complete_capture_modules),
            calibration_reason="summary_replay_mismatch",
        )

    abs_errors = [abs(_safe_float(row["abs_error_vs_raw_exact"], 0.0)) for row in matched_replay_rows]
    relative_errors = [
        abs(_safe_float(row["abs_error_vs_raw_exact"], 0.0))
        / max(1.0, abs(_safe_float(row["raw_exact_dot_product"], 0.0)))
        for row in matched_replay_rows
    ]
    median_relative_error = _median(relative_errors) or 0.0
    max_calibration_scale = float(config.get("bitstream", {}).get("max_calibration_scale", 8.0) or 8.0)
    effective_stream_length_scale = min(max(1.0 + median_relative_error, 1.0), max_calibration_scale)

    return replace(
        resolved_cfg,
        effective_stream_length_scale=effective_stream_length_scale,
        effective_stream_length_scale_provenance="derived_from_calibration_replay",
        calibration_applied=True,
        calibration_summary_json=str(summary_path),
        calibration_capture_row_count=len(matching_capture_rows),
        calibration_replay_row_count=len(matched_replay_rows),
        calibration_module_count=len(
            {str(row.get("module_key") or "") for row in matched_replay_rows}
        ),
        calibration_median_abs_error=_median(abs_errors),
        calibration_max_abs_error=max(abs_errors) if abs_errors else None,
        calibration_median_relative_error=median_relative_error,
        calibration_reason="applied",
    )


def _resolve_bitstream_estimator_config(config: dict) -> BitstreamEstimatorConfig:
    bitstream = config.get("bitstream") or {}
    photonic = config.get("photonic") or {}
    electronic = config.get("electronic") or {}

    requested_semantics = str(
        bitstream.get("execution_semantics") or config.get("execution_semantics") or "proxy"
    ).strip().lower() or "proxy"
    enabled = _to_bool(bitstream.get("enabled"), False) or requested_semantics == "bitstream"
    stream_length = int(bitstream.get("stream_length") or 0)
    sample_rate_gsps = bitstream.get("sample_rate_gsps") or photonic.get("sample_rate_gsps")
    tile_parallel_outputs = max(
        1,
        int(int(photonic.get("tile_k", 1)) * int(photonic.get("tile_k_prime", 1)) or 1),
    )
    if bitstream.get("parallel_outputs") not in {None, ""}:
        parallel_outputs = max(1, _safe_int(bitstream.get("parallel_outputs"), tile_parallel_outputs))
        parallel_outputs_provenance = "explicit_config"
    else:
        parallel_outputs = tile_parallel_outputs
        parallel_outputs_provenance = "derived_from_tile_shape"
    cycles_per_stream_bit, cycles_per_stream_bit_provenance = _resolve_int_scalar_with_provenance(
        primary_cfg=bitstream,
        primary_key="cycles_per_stream_bit",
        primary_default=1,
        primary_provenance="explicit_config",
    )
    accumulator_energy_pj, accumulator_energy_pj_provenance = _resolve_float_scalar_with_provenance(
        primary_cfg=bitstream,
        primary_key="accumulator_energy_pj",
        primary_default=0.0,
        primary_provenance="explicit_config",
        secondary_cfg=electronic,
        secondary_key="norm_energy_pj",
        secondary_default=0.0,
        secondary_provenance="aliased_from_electronic_norm",
    )
    effective_stream_length_scale, effective_stream_length_scale_provenance = _resolve_float_scalar_with_provenance(
        primary_cfg=bitstream,
        primary_key="effective_stream_length_scale",
        primary_default=1.0,
        primary_provenance="explicit_config",
    )
    norm_parallelism, norm_parallelism_provenance = _resolve_int_scalar_with_provenance(
        primary_cfg=bitstream,
        primary_key="norm_parallelism",
        primary_default=32,
        primary_provenance="explicit_config",
        secondary_cfg=electronic,
        secondary_key="parallelism_factor",
        secondary_default=32,
        secondary_provenance="aliased_from_electronic_norm",
    )
    activation_parallelism, activation_parallelism_provenance = _resolve_int_scalar_with_provenance(
        primary_cfg=bitstream,
        primary_key="activation_parallelism",
        primary_default=32,
        primary_provenance="explicit_config",
        secondary_cfg=electronic,
        secondary_key="parallelism_factor",
        secondary_default=32,
        secondary_provenance="aliased_from_electronic_norm",
    )

    if not enabled or stream_length <= 0:
        return BitstreamEstimatorConfig(
            enabled=False,
            execution_semantics="proxy",
            encoding_mode="bipolar",
            multiplier_mode="xnor",
            accumulator_mode="bitcount",
            stream_length=0,
            generator="bernoulli",
            calibration_source=None,
            capture_manifest_csv=None,
            sample_rate_gsps=sample_rate_gsps,
            parallel_outputs=parallel_outputs,
            cycles_per_stream_bit=cycles_per_stream_bit,
            accumulator_energy_pj=0.0,
            effective_stream_length_scale=1.0,
            calibration_applied=False,
            calibration_summary_json=None,
            calibration_capture_row_count=0,
            calibration_replay_row_count=0,
            calibration_module_count=0,
            calibration_median_abs_error=None,
            calibration_max_abs_error=None,
            calibration_median_relative_error=None,
            calibration_reason=None,
            stream_state_policy={},
            parallel_outputs_provenance=parallel_outputs_provenance,
            cycles_per_stream_bit_provenance=cycles_per_stream_bit_provenance,
            accumulator_energy_pj_provenance="implicit_default",
            effective_stream_length_scale_provenance="implicit_default",
            norm_parallelism=norm_parallelism,
            norm_parallelism_provenance=norm_parallelism_provenance,
            activation_parallelism=activation_parallelism,
            activation_parallelism_provenance=activation_parallelism_provenance,
        )
    generator_name = str(bitstream.get("generator") or "bernoulli").strip().lower()
    resolved = BitstreamEstimatorConfig(
        enabled=True,
        execution_semantics="bitstream",
        encoding_mode=str(bitstream.get("encoding_mode") or "bipolar").strip().lower(),
        multiplier_mode=str(bitstream.get("multiplier_mode") or "xnor").strip().lower(),
        accumulator_mode=str(bitstream.get("accumulator_mode") or "bitcount").strip().lower(),
        stream_length=stream_length,
        generator=generator_name,
        calibration_source=(
            str(bitstream.get("calibration_source")).strip()
            if bitstream.get("calibration_source") not in {None, ""}
            else None
        ),
        capture_manifest_csv=(
            str(bitstream.get("capture_manifest_csv")).strip()
            if bitstream.get("capture_manifest_csv") not in {None, ""}
            else None
        ),
        sample_rate_gsps=float(sample_rate_gsps) if sample_rate_gsps is not None else None,
        parallel_outputs=parallel_outputs,
        cycles_per_stream_bit=cycles_per_stream_bit,
        accumulator_energy_pj=accumulator_energy_pj,
        effective_stream_length_scale=effective_stream_length_scale,
        calibration_applied=False,
        calibration_summary_json=None,
        calibration_capture_row_count=0,
        calibration_replay_row_count=0,
        calibration_module_count=0,
        calibration_median_abs_error=None,
        calibration_max_abs_error=None,
        calibration_median_relative_error=None,
        calibration_reason=None,
        stream_state_policy=resolve_generator_stream_state_policy(
            generator_name,
            policy_config=bitstream,
        ),
        parallel_outputs_provenance=parallel_outputs_provenance,
        cycles_per_stream_bit_provenance=cycles_per_stream_bit_provenance,
        accumulator_energy_pj_provenance=accumulator_energy_pj_provenance,
        effective_stream_length_scale_provenance=effective_stream_length_scale_provenance,
        norm_parallelism=norm_parallelism,
        norm_parallelism_provenance=norm_parallelism_provenance,
        activation_parallelism=activation_parallelism,
        activation_parallelism_provenance=activation_parallelism_provenance,
    )
    return _resolve_bitstream_calibration(config=config, resolved_cfg=resolved)


def _effective_stream_length(bitstream_cfg: BitstreamEstimatorConfig) -> int:
    return max(
        int(bitstream_cfg.stream_length),
        int(
            math.ceil(
                float(bitstream_cfg.stream_length)
                * float(bitstream_cfg.effective_stream_length_scale)
            )
        ),
    )


def _bitstream_scalar_summary(bitstream_cfg: BitstreamEstimatorConfig) -> dict[str, object]:
    return {
        "bitstream_parallel_outputs": bitstream_cfg.parallel_outputs,
        "bitstream_parallel_outputs_provenance": bitstream_cfg.parallel_outputs_provenance,
        "bitstream_cycles_per_stream_bit": bitstream_cfg.cycles_per_stream_bit,
        "bitstream_cycles_per_stream_bit_provenance": (
            bitstream_cfg.cycles_per_stream_bit_provenance
        ),
        "bitstream_accumulator_energy_pj": bitstream_cfg.accumulator_energy_pj,
        "bitstream_accumulator_energy_pj_provenance": (
            bitstream_cfg.accumulator_energy_pj_provenance
        ),
        "bitstream_effective_stream_length_scale": bitstream_cfg.effective_stream_length_scale,
        "bitstream_effective_stream_length_scale_provenance": (
            bitstream_cfg.effective_stream_length_scale_provenance
        ),
        "bitstream_elementwise_parallelism_factor": {
            "activation": bitstream_cfg.activation_parallelism,
            "norm": bitstream_cfg.norm_parallelism,
        },
        "bitstream_elementwise_parallelism_provenance": {
            "activation": bitstream_cfg.activation_parallelism_provenance,
            "norm": bitstream_cfg.norm_parallelism_provenance,
        },
    }


def build_energy_params(config: dict) -> dict[str, float]:
    """Assemble photonic energy parameters (in joules per sample)."""
    photonic = config.get("photonic", {})
    energy_cfg = config.get("energy", {})

    sample_rate_gsps = photonic.get("sample_rate_gsps")

    dac_rate = _get_with_fallback(energy_cfg, "dac_sample_rate_gsps", sample_rate_gsps)
    adc_rate = _get_with_fallback(energy_cfg, "adc_sample_rate_gsps", sample_rate_gsps)
    mod_rate = _get_with_fallback(energy_cfg, "modulator_sample_rate_gsps", sample_rate_gsps)
    pd_rate = _get_with_fallback(energy_cfg, "pd_sample_rate_gsps", sample_rate_gsps)
    tia_rate = _get_with_fallback(energy_cfg, "tia_sample_rate_gsps", sample_rate_gsps)
    laser_rate = _get_with_fallback(energy_cfg, "laser_sample_rate_gsps", sample_rate_gsps)

    e_dac = _power_to_energy_j(energy_cfg.get("dac_power_mw"), dac_rate)
    e_adc = _power_to_energy_j(energy_cfg.get("adc_power_mw"), adc_rate)
    e_mod = _power_to_energy_j(energy_cfg.get("modulator_power_mw"), mod_rate)
    e_pd = _power_to_energy_j(energy_cfg.get("pd_power_mw"), pd_rate)
    e_tia = _power_to_energy_j(energy_cfg.get("tia_power_mw"), tia_rate)
    e_laser = _power_to_energy_j(energy_cfg.get("laser_power_mw"), laser_rate)

    return {
        "sample_rate_gsps": sample_rate_gsps,
        "e_dac": e_dac or 0.0,
        "e_adc": e_adc or 0.0,
        "e_mod": e_mod or 0.0,
        "e_pd": e_pd or 0.0,
        "e_tia": e_tia or 0.0,
        "e_laser": e_laser or 0.0,
        "static_power_mw": energy_cfg.get("static_power_mw", 0.0),
        "mem_energy_pj": energy_cfg.get("mem_energy_pj", 0.0),
    }


def build_electronic_params(config: dict) -> dict[str, float]:
    """Extract electronic per-element energy/latency params (pJ / ns)."""
    electronic = config.get("electronic", {})
    return {
        "softmax_energy_pj": electronic.get("softmax_energy_pj", 0.0),
        "softmax_latency_ns": electronic.get("softmax_latency_ns", 0.0),
        "norm_energy_pj": electronic.get("norm_energy_pj", 0.0),
        "norm_latency_ns": electronic.get("norm_latency_ns", 0.0),
        "activation_energy_pj": electronic.get("activation_energy_pj", 0.0),
        "activation_latency_ns": electronic.get("activation_latency_ns", 0.0),
    }


def estimate_elementwise_energy_latency(
    elements: int | None, op_type: str, config: dict
) -> dict[str, float]:
    """Estimate electronic op energy/latency for elementwise operations."""
    if elements is None:
        return {"latency_s": 0.0, "energy_j": 0.0}
    elements = int(elements)
    if elements <= 0:
        return {"latency_s": 0.0, "energy_j": 0.0}

    params = build_electronic_params(config)
    if op_type == "softmax":
        energy_pj = params["softmax_energy_pj"]
        latency_ns = params["softmax_latency_ns"]
    elif op_type in {"norm", "layer_norm", "batch_norm", "group_norm"}:
        energy_pj = params["norm_energy_pj"]
        latency_ns = params["norm_latency_ns"]
    elif op_type == "activation":
        energy_pj = params["activation_energy_pj"]
        latency_ns = params["activation_latency_ns"]
    else:
        energy_pj = 0.0
        latency_ns = 0.0

    # Energy scales with processed elements; latency is parallelism aware.
    electronic = config.get("electronic", {})
    parallelism = max(1, int(electronic.get("parallelism_factor", 32)))
    energy_j = elements * energy_pj * 1e-12
    latency_s = (elements / parallelism) * latency_ns * 1e-9
    return {"latency_s": latency_s, "energy_j": energy_j}


def estimate_bitstream_elementwise_energy_latency(
    elements: int | None,
    op_type: str,
    config: dict,
    *,
    bitstream_cfg: BitstreamEstimatorConfig | None = None,
) -> dict[str, object]:
    """Estimate bounded native bitstream cost for supported elementwise op families."""
    if op_type not in BITSTREAM_NATIVE_ELEMENTWISE_TYPES:
        raise ValueError(f"Unsupported native bitstream elementwise op: {op_type}")

    resolved = bitstream_cfg or _resolve_bitstream_estimator_config(config)
    if not resolved.enabled:
        raise ValueError("Bitstream estimator requested without enabled bitstream config.")
    if resolved.sample_rate_gsps is None:
        raise ValueError("sample_rate_gsps is required for bitstream latency estimation.")

    effective_stream_length = _effective_stream_length(resolved)
    empty_details = {
        "stream_length": resolved.stream_length,
        "effective_stream_length": effective_stream_length,
        "effective_stream_length_scale": resolved.effective_stream_length_scale,
        "effective_stream_length_scale_provenance": (
            resolved.effective_stream_length_scale_provenance
        ),
        "parallel_passes": 0,
        "parallel_outputs": 0,
        "parallel_outputs_provenance": "implicit_default",
        "cycles_per_stream_bit": resolved.cycles_per_stream_bit,
        "cycles_per_stream_bit_provenance": resolved.cycles_per_stream_bit_provenance,
        "accumulator_energy_pj": resolved.accumulator_energy_pj,
        "accumulator_energy_pj_provenance": resolved.accumulator_energy_pj_provenance,
        "generator": resolved.generator,
        "encoding_mode": resolved.encoding_mode,
        "multiplier_mode": resolved.multiplier_mode,
        "accumulator_mode": resolved.accumulator_mode,
        "stream_state_policy": resolved.stream_state_policy,
        "generator_stream_state_policy": resolved.stream_state_policy,
        "calibration_source": resolved.calibration_source,
        "capture_manifest_csv": resolved.capture_manifest_csv,
        "calibration_applied": resolved.calibration_applied,
        "calibration_summary_json": resolved.calibration_summary_json,
        "calibration_capture_row_count": resolved.calibration_capture_row_count,
        "calibration_replay_row_count": resolved.calibration_replay_row_count,
        "calibration_module_count": resolved.calibration_module_count,
        "calibration_median_abs_error": resolved.calibration_median_abs_error,
        "calibration_max_abs_error": resolved.calibration_max_abs_error,
        "calibration_median_relative_error": resolved.calibration_median_relative_error,
        "calibration_reason": resolved.calibration_reason,
        "native_elementwise_family": op_type,
        "unary_events": 0,
        "elementwise_parallelism_factor": 0,
        "elementwise_parallelism_provenance": "implicit_default",
    }
    if elements is None:
        return {
            "latency_s": 0.0,
            "energy_j": 0.0,
            "parallel_passes": 0,
            "energy_components_j": {
                "load_x": 0.0,
                "load_y": 0.0,
                "oe": 0.0,
                "adc_pca": 0.0,
                "detect": 0.0,
                "laser": 0.0,
                "mem": 0.0,
                "static": 0.0,
                "bitstream_accumulator": 0.0,
            },
            "bitstream_datapath_stages": _new_bitstream_datapath_stage_summary(),
            "bitstream_details": empty_details,
        }

    elements = int(elements)
    if elements <= 0:
        return {
            "latency_s": 0.0,
            "energy_j": 0.0,
            "parallel_passes": 0,
            "energy_components_j": {
                "load_x": 0.0,
                "load_y": 0.0,
                "oe": 0.0,
                "adc_pca": 0.0,
                "detect": 0.0,
                "laser": 0.0,
                "mem": 0.0,
                "static": 0.0,
                "bitstream_accumulator": 0.0,
            },
            "bitstream_datapath_stages": _new_bitstream_datapath_stage_summary(),
            "bitstream_details": empty_details,
        }

    bitstream = config.get("bitstream") or {}
    electronic = config.get("electronic") or {}
    energy_params = build_energy_params(config)
    if op_type in BITSTREAM_NATIVE_NORMALIZATION_TYPES:
        norm_parallelism = resolved.norm_parallelism
        stats_passes = max(1, int(bitstream.get("norm_stats_passes", 2) or 2))
        apply_passes = max(1, int(bitstream.get("norm_apply_passes", 1) or 1))
        total_passes = stats_passes + apply_passes
        parallel_passes = _ceil_div(elements, norm_parallelism)
        total_cycles = (
            parallel_passes
            * effective_stream_length
            * resolved.cycles_per_stream_bit
            * total_passes
        )
        latency_s = total_cycles / (resolved.sample_rate_gsps * 1e9)
        stats_events = elements * effective_stream_length * stats_passes
        apply_events = elements * effective_stream_length * apply_passes
        unary_events = stats_events + apply_events
        load_scale = float(bitstream.get("norm_load_scale", 1.0) or 1.0)
        stats_scale = float(bitstream.get("norm_stats_accumulator_scale", 2.0) or 2.0)
        apply_scale = float(bitstream.get("norm_apply_accumulator_scale", 1.0) or 1.0)
        laser_scale = float(bitstream.get("norm_output_laser_scale", 1.0) or 1.0)
        mem_scale = float(bitstream.get("norm_mem_scale", 1.0) or 1.0)
        load_x = unary_events * (energy_params["e_dac"] + energy_params["e_mod"]) * load_scale
        bitstream_accumulator = (
            (stats_events * stats_scale) + (apply_events * apply_scale)
        ) * (resolved.accumulator_energy_pj * 1e-12)
        laser = apply_events * energy_params["e_laser"] * laser_scale
        mem = unary_events * (energy_params["mem_energy_pj"] * 1e-12) * mem_scale
        laser_events = apply_events
        detail_updates = {
            "parallel_outputs": norm_parallelism,
            "parallel_outputs_provenance": resolved.norm_parallelism_provenance,
            "elementwise_parallelism_factor": norm_parallelism,
            "elementwise_parallelism_provenance": resolved.norm_parallelism_provenance,
            "stats_passes": stats_passes,
            "apply_passes": apply_passes,
            "stats_events": stats_events,
            "apply_events": apply_events,
        }
    else:
        activation_parallelism = resolved.activation_parallelism
        parallel_passes = _ceil_div(elements, activation_parallelism)
        total_cycles = parallel_passes * effective_stream_length * resolved.cycles_per_stream_bit
        latency_s = total_cycles / (resolved.sample_rate_gsps * 1e9)
        unary_events = elements * effective_stream_length
        load_scale = float(bitstream.get("activation_load_scale", 1.0) or 1.0)
        threshold_scale = float(bitstream.get("activation_threshold_scale", 1.0) or 1.0)
        laser_scale = float(bitstream.get("activation_output_laser_scale", 1.0) or 1.0)
        mem_scale = float(bitstream.get("activation_mem_scale", 1.0) or 1.0)
        load_x = unary_events * (energy_params["e_dac"] + energy_params["e_mod"]) * load_scale
        bitstream_accumulator = (
            unary_events * (resolved.accumulator_energy_pj * 1e-12) * threshold_scale
        )
        laser = unary_events * energy_params["e_laser"] * laser_scale
        mem = unary_events * (energy_params["mem_energy_pj"] * 1e-12) * mem_scale
        laser_events = unary_events
        detail_updates = {
            "parallel_outputs": activation_parallelism,
            "parallel_outputs_provenance": resolved.activation_parallelism_provenance,
            "elementwise_parallelism_factor": activation_parallelism,
            "elementwise_parallelism_provenance": resolved.activation_parallelism_provenance,
        }
    static = (energy_params["static_power_mw"] / 1000.0) * latency_s
    energy_j = load_x + bitstream_accumulator + laser + mem + static
    bitstream_datapath_stages = {
        "stream_generation_load": {
            "events": float(unary_events),
            "cycles": float(total_cycles),
            "energy_j": load_x,
        },
        "stochastic_multiply": {
            "events": 0.0,
            "cycles": 0.0,
            "energy_j": 0.0,
        },
        "accumulation_pca": {
            "events": float(unary_events),
            "cycles": float(total_cycles),
            "energy_j": bitstream_accumulator,
        },
        "laser_clocking": {
            "events": float(laser_events),
            "cycles": float(total_cycles),
            "energy_j": laser,
        },
        "memory": {
            "events": float(unary_events),
            "cycles": 0.0,
            "energy_j": mem,
        },
        "static": {
            "events": 0.0,
            "cycles": float(total_cycles),
            "energy_j": static,
        },
        "serialization_passes": {
            "events": float(parallel_passes),
            "cycles": float(total_cycles),
            "energy_j": 0.0,
        },
    }
    return {
        "latency_s": latency_s,
        "energy_j": energy_j,
        "parallel_passes": parallel_passes,
        "energy_components_j": {
            "load_x": load_x,
            "load_y": 0.0,
            "oe": 0.0,
            "adc_pca": 0.0,
            "detect": bitstream_accumulator,
            "laser": laser,
            "mem": mem,
            "static": static,
            "bitstream_accumulator": bitstream_accumulator,
        },
        "bitstream_datapath_stages": bitstream_datapath_stages,
        "bitstream_details": {
            **empty_details,
            "parallel_passes": parallel_passes,
            **detail_updates,
            "unary_events": unary_events,
        },
    }


def estimate_gemm_energy_latency(m: int, d: int, n: int, config: dict) -> dict[str, object]:
    """Estimate GEMM energy/latency with a tiled photonic model."""
    photonic = config.get("photonic", {})
    k = int(photonic.get("tile_k", 16))
    k_prime = int(photonic.get("tile_k_prime", 1))
    cycles_per_tile = int(photonic.get("cycles_per_tile", 1))

    energy_params = build_energy_params(config)
    sample_rate_gsps = energy_params["sample_rate_gsps"]
    if sample_rate_gsps is None:
        raise ValueError("photonic.sample_rate_gsps is required for latency.")

    tiles_m = _ceil_div(m, k)
    tiles_d = _ceil_div(d, k)
    tiles_n = _ceil_div(n, k_prime)
    tiles = tiles_m * tiles_d * tiles_n

    # Parameterized component model (J).
    e_load = energy_params["e_dac"] + energy_params["e_mod"]
    e_oe = energy_params["e_pd"] + energy_params["e_tia"]
    e_adc = energy_params["e_adc"]
    e_laser = energy_params["e_laser"]

    load_x = tiles * (k * k) * e_load
    load_y = tiles * (k * k_prime) * e_load
    oe = tiles * (k * k_prime) * e_oe
    adc_pca = tiles * (k * k_prime) * e_adc
    detect = oe + adc_pca
    laser = tiles * e_laser

    mem_energy_j = 0.0
    if energy_params["mem_energy_pj"]:
        mem_energy_j = (
            (tiles * (k * k) + tiles * (k * k_prime) + tiles * (k * k_prime))
            * (energy_params["mem_energy_pj"] * 1e-12)
        )

    total_cycles = tiles * cycles_per_tile
    latency_s = total_cycles / (sample_rate_gsps * 1e9)

    static_power_w = energy_params["static_power_mw"] / 1000.0
    static_energy = static_power_w * latency_s

    energy_j = load_x + load_y + detect + laser + mem_energy_j + static_energy

    return {
        "tiles": tiles,
        "latency_s": latency_s,
        "energy_j": energy_j,
        "energy_components_j": {
            "load_x": load_x,
            "load_y": load_y,
            "oe": oe,
            "adc_pca": adc_pca,
            "detect": detect,
            "laser": laser,
            "mem": mem_energy_j,
            "static": static_energy,
        },
    }


def estimate_bitstream_gemm_energy_latency(
    m: int,
    d: int,
    n: int,
    config: dict,
    *,
    bitstream_cfg: BitstreamEstimatorConfig | None = None,
) -> dict[str, object]:
    """Estimate GEMM-like work with a bounded bitstream cost model."""
    resolved = bitstream_cfg or _resolve_bitstream_estimator_config(config)
    if not resolved.enabled:
        raise ValueError("Bitstream estimator requested without enabled bitstream config.")
    if resolved.sample_rate_gsps is None:
        raise ValueError("sample_rate_gsps is required for bitstream latency estimation.")

    energy_params = build_energy_params(config)
    effective_stream_length = _effective_stream_length(resolved)
    stream_events_per_output = int(d) * effective_stream_length
    output_elements = int(m) * int(n)
    parallel_passes = _ceil_div(output_elements, resolved.parallel_outputs)
    total_cycles = parallel_passes * stream_events_per_output * resolved.cycles_per_stream_bit
    latency_s = total_cycles / (resolved.sample_rate_gsps * 1e9)

    load_scale = float(config.get("bitstream", {}).get("load_scale", 1.0) or 1.0)
    detect_scale = float(config.get("bitstream", {}).get("detect_scale", 1.0) or 1.0)
    mem_scale = float(config.get("bitstream", {}).get("mem_scale", 1.0) or 1.0)

    lhs_load_events = int(m) * int(d) * effective_stream_length
    rhs_load_events = int(d) * int(n) * effective_stream_length
    multiply_events = output_elements * stream_events_per_output

    load_x = lhs_load_events * (energy_params["e_dac"] + energy_params["e_mod"]) * load_scale
    load_y = rhs_load_events * (energy_params["e_dac"] + energy_params["e_mod"]) * load_scale
    oe = multiply_events * (energy_params["e_pd"] + energy_params["e_tia"]) * detect_scale
    adc_pca = multiply_events * energy_params["e_adc"] * detect_scale
    bitstream_accumulator = multiply_events * (resolved.accumulator_energy_pj * 1e-12)
    detect = oe + adc_pca + bitstream_accumulator
    laser = output_elements * effective_stream_length * energy_params["e_laser"]
    mem = multiply_events * (energy_params["mem_energy_pj"] * 1e-12) * mem_scale
    static = (energy_params["static_power_mw"] / 1000.0) * latency_s
    energy_j = load_x + load_y + detect + laser + mem + static
    bitstream_datapath_stages = {
        "stream_generation_load": {
            "events": float(lhs_load_events + rhs_load_events),
            "cycles": float(total_cycles),
            "energy_j": load_x + load_y,
        },
        "stochastic_multiply": {
            "events": float(multiply_events),
            "cycles": float(total_cycles),
            "energy_j": oe,
        },
        "accumulation_pca": {
            "events": float(multiply_events),
            "cycles": float(total_cycles),
            "energy_j": adc_pca + bitstream_accumulator,
        },
        "laser_clocking": {
            "events": float(output_elements * effective_stream_length),
            "cycles": float(total_cycles),
            "energy_j": laser,
        },
        "memory": {
            "events": float(multiply_events),
            "cycles": 0.0,
            "energy_j": mem,
        },
        "static": {
            "events": 0.0,
            "cycles": float(total_cycles),
            "energy_j": static,
        },
        "serialization_passes": {
            "events": float(parallel_passes),
            "cycles": float(total_cycles),
            "energy_j": 0.0,
        },
    }

    return {
        "tiles": output_elements,
        "latency_s": latency_s,
        "energy_j": energy_j,
        "parallel_passes": parallel_passes,
        "energy_components_j": {
            "load_x": load_x,
            "load_y": load_y,
            "oe": oe,
            "adc_pca": adc_pca,
            "detect": detect,
            "laser": laser,
            "mem": mem,
            "static": static,
            "bitstream_accumulator": bitstream_accumulator,
        },
        "bitstream_datapath_stages": bitstream_datapath_stages,
        "bitstream_details": {
            "stream_length": resolved.stream_length,
            "effective_stream_length": effective_stream_length,
            "effective_stream_length_scale": resolved.effective_stream_length_scale,
            "effective_stream_length_scale_provenance": (
                resolved.effective_stream_length_scale_provenance
            ),
            "stream_events_per_output": stream_events_per_output,
            "parallel_passes": parallel_passes,
            "multiply_events": multiply_events,
            "parallel_outputs": resolved.parallel_outputs,
            "parallel_outputs_provenance": resolved.parallel_outputs_provenance,
            "cycles_per_stream_bit": resolved.cycles_per_stream_bit,
            "cycles_per_stream_bit_provenance": resolved.cycles_per_stream_bit_provenance,
            "accumulator_energy_pj": resolved.accumulator_energy_pj,
            "accumulator_energy_pj_provenance": resolved.accumulator_energy_pj_provenance,
            "generator": resolved.generator,
            "encoding_mode": resolved.encoding_mode,
            "multiplier_mode": resolved.multiplier_mode,
            "accumulator_mode": resolved.accumulator_mode,
            "stream_state_policy": resolved.stream_state_policy,
            "generator_stream_state_policy": resolved.stream_state_policy,
            "elementwise_parallelism_factor": None,
            "elementwise_parallelism_provenance": "",
            "calibration_source": resolved.calibration_source,
            "capture_manifest_csv": resolved.capture_manifest_csv,
            "calibration_applied": resolved.calibration_applied,
            "calibration_summary_json": resolved.calibration_summary_json,
            "calibration_capture_row_count": resolved.calibration_capture_row_count,
            "calibration_replay_row_count": resolved.calibration_replay_row_count,
            "calibration_module_count": resolved.calibration_module_count,
            "calibration_median_abs_error": resolved.calibration_median_abs_error,
            "calibration_max_abs_error": resolved.calibration_max_abs_error,
            "calibration_median_relative_error": resolved.calibration_median_relative_error,
            "calibration_reason": resolved.calibration_reason,
        },
    }


def estimate_bitstream_conv2d_energy_latency(
    m: int,
    d: int,
    n: int,
    config: dict,
    *,
    kernel: object,
    stride: object,
    groups: object,
    dilation: object | None = None,
    runtime_stream_reuse_policy: str = DEFAULT_BITSTREAM_RUNTIME_STREAM_REUSE_POLICY,
    bitstream_cfg: BitstreamEstimatorConfig | None = None,
) -> dict[str, object]:
    """Estimate conv2d work with explicit runtime-path semantics."""
    resolved = bitstream_cfg or _resolve_bitstream_estimator_config(config)
    if not resolved.enabled:
        raise ValueError("Bitstream estimator requested without enabled bitstream config.")
    if resolved.sample_rate_gsps is None:
        raise ValueError("sample_rate_gsps is required for bitstream latency estimation.")

    op = {
        "type": "conv2d",
        "m": int(m),
        "d": int(d),
        "n": int(n),
        "kernel": kernel,
        "stride": stride,
        "groups": groups,
        "dilation": dilation or [1, 1],
    }
    conv_native_class = classify_conv_native_class(op)
    runtime_semantics = build_conv_runtime_semantics(
        op,
        runtime_stream_reuse_policy=runtime_stream_reuse_policy,
    )
    energy_params = build_energy_params(config)
    effective_stream_length = _effective_stream_length(resolved)
    output_elements = int(m) * int(n)
    groups_int = max(1, _safe_int(groups, 1))
    out_channels_per_group = max(1, int(n) // groups_int)
    stream_events_per_dot = int(d) * effective_stream_length
    output_write_events = output_elements * effective_stream_length
    parallel_passes = _ceil_div(output_elements, resolved.parallel_outputs)

    reuse_policy = str(runtime_stream_reuse_policy or "").strip()
    use_operand_reuse = reuse_policy == DEFAULT_BITSTREAM_RUNTIME_STREAM_REUSE_POLICY
    if use_operand_reuse:
        patch_bundle_count = int(m) * groups_int
        filter_bundle_count = int(n)
        patch_generated_events = patch_bundle_count * stream_events_per_dot
        filter_materialization_events = filter_bundle_count * stream_events_per_dot
        patch_reuse_fanout = out_channels_per_group
        filter_reuse_fanout = int(m)
    else:
        patch_bundle_count = output_elements
        filter_bundle_count = output_elements
        patch_generated_events = output_elements * stream_events_per_dot
        filter_materialization_events = output_elements * stream_events_per_dot
        patch_reuse_fanout = 1
        filter_reuse_fanout = 1

    multiply_events = output_elements * stream_events_per_dot
    total_event_volume = (
        patch_generated_events
        + filter_materialization_events
        + multiply_events
        + output_write_events
    )
    total_cycles = parallel_passes * total_event_volume * resolved.cycles_per_stream_bit / max(
        1, resolved.parallel_outputs
    )
    latency_s = float(total_cycles) / (resolved.sample_rate_gsps * 1e9)

    load_x = patch_generated_events * (energy_params["e_dac"] + energy_params["e_mod"])
    load_y = filter_materialization_events * (energy_params["e_dac"] + energy_params["e_mod"])
    oe = multiply_events * (energy_params["e_pd"] + energy_params["e_tia"])
    adc_pca = multiply_events * energy_params["e_adc"]
    bitstream_accumulator = multiply_events * (resolved.accumulator_energy_pj * 1e-12)
    laser = output_write_events * energy_params["e_laser"]
    mem_patch = patch_generated_events * (energy_params["mem_energy_pj"] * 1e-12)
    mem_filter = filter_materialization_events * (energy_params["mem_energy_pj"] * 1e-12)
    mem_writeback = output_write_events * (energy_params["mem_energy_pj"] * 1e-12)
    mem = mem_patch + mem_filter + mem_writeback
    static = (energy_params["static_power_mw"] / 1000.0) * latency_s
    detect = oe + adc_pca + bitstream_accumulator
    energy_j = load_x + load_y + detect + laser + mem + static

    patch_cycles = patch_generated_events * resolved.cycles_per_stream_bit
    filter_cycles = filter_materialization_events * resolved.cycles_per_stream_bit
    multiply_cycles = multiply_events * resolved.cycles_per_stream_bit / max(
        1, resolved.parallel_outputs
    )
    writeback_cycles = output_write_events * resolved.cycles_per_stream_bit
    bitstream_datapath_stages = {
        "patch_generation_load": {
            "events": float(patch_generated_events),
            "cycles": float(patch_cycles),
            "energy_j": load_x + mem_patch,
        },
        "filter_materialization_or_residency": {
            "events": float(filter_materialization_events),
            "cycles": float(filter_cycles),
            "energy_j": load_y + mem_filter,
        },
        "stochastic_multiply": {
            "events": float(multiply_events),
            "cycles": float(multiply_cycles),
            "energy_j": oe,
        },
        "psum_or_accumulation": {
            "events": float(multiply_events),
            "cycles": float(multiply_cycles),
            "energy_j": adc_pca + bitstream_accumulator,
        },
        "writeback_or_decode": {
            "events": float(output_write_events),
            "cycles": float(writeback_cycles),
            "energy_j": laser + mem_writeback,
        },
        "serialization_passes": {
            "events": float(parallel_passes),
            "cycles": float(total_cycles),
            "energy_j": 0.0,
        },
        "static": {
            "events": 0.0,
            "cycles": float(total_cycles),
            "energy_j": static,
        },
    }
    if conv_native_class == CONV_NATIVE_CLASS_DEPTHWISE_GROUPED_PATCH_DOMINANT:
        pipeline_provenance_class = "depthwise_patch_dominant_operand_reuse"
    elif conv_native_class == CONV_NATIVE_CLASS_STANDARD_SPATIAL_FILTER_BANK_VISIBLE:
        pipeline_provenance_class = "filter_bank_residency_visible_operand_reuse"
    else:
        pipeline_provenance_class = "patch_generation_dominant_with_channel_fanout_reuse"

    return {
        "tiles": output_elements,
        "latency_s": latency_s,
        "energy_j": energy_j,
        "parallel_passes": parallel_passes,
        "energy_components_j": {
            "load_x": load_x,
            "load_y": load_y,
            "oe": oe,
            "adc_pca": adc_pca,
            "detect": detect,
            "laser": laser,
            "mem": mem,
            "static": static,
            "bitstream_accumulator": bitstream_accumulator,
        },
        "bitstream_datapath_stages": bitstream_datapath_stages,
        "bitstream_details": {
            "stream_length": resolved.stream_length,
            "effective_stream_length": effective_stream_length,
            "effective_stream_length_scale": resolved.effective_stream_length_scale,
            "effective_stream_length_scale_provenance": (
                resolved.effective_stream_length_scale_provenance
            ),
            "stream_events_per_output": stream_events_per_dot,
            "parallel_passes": parallel_passes,
            "multiply_events": multiply_events,
            "parallel_outputs": resolved.parallel_outputs,
            "parallel_outputs_provenance": resolved.parallel_outputs_provenance,
            "cycles_per_stream_bit": resolved.cycles_per_stream_bit,
            "cycles_per_stream_bit_provenance": resolved.cycles_per_stream_bit_provenance,
            "accumulator_energy_pj": resolved.accumulator_energy_pj,
            "accumulator_energy_pj_provenance": resolved.accumulator_energy_pj_provenance,
            "generator": resolved.generator,
            "encoding_mode": resolved.encoding_mode,
            "multiplier_mode": resolved.multiplier_mode,
            "accumulator_mode": resolved.accumulator_mode,
            "stream_state_policy": resolved.stream_state_policy,
            "generator_stream_state_policy": resolved.stream_state_policy,
            "runtime_stream_reuse_policy": reuse_policy,
            "elementwise_parallelism_factor": None,
            "elementwise_parallelism_provenance": "",
            "calibration_source": resolved.calibration_source,
            "capture_manifest_csv": resolved.capture_manifest_csv,
            "calibration_applied": resolved.calibration_applied,
            "calibration_summary_json": resolved.calibration_summary_json,
            "calibration_capture_row_count": resolved.calibration_capture_row_count,
            "calibration_replay_row_count": resolved.calibration_replay_row_count,
            "calibration_module_count": resolved.calibration_module_count,
            "calibration_median_abs_error": resolved.calibration_median_abs_error,
            "calibration_max_abs_error": resolved.calibration_max_abs_error,
            "calibration_median_relative_error": resolved.calibration_median_relative_error,
            "calibration_reason": resolved.calibration_reason,
            "conv_native_class": conv_native_class,
            "conv_fidelity_stage": CONV_FIDELITY_STAGE_RUNTIME_MODELED,
            "conv_fidelity_blockers": conv_fidelity_blockers_for_stage(
                CONV_FIDELITY_STAGE_RUNTIME_MODELED
            ),
            "conv_runtime_semantics": runtime_semantics,
            "patch_bundle_count_per_module_call": patch_bundle_count,
            "filter_bundle_count_per_module_call": filter_bundle_count,
            "patch_reuse_fanout_under_operand_reuse": patch_reuse_fanout,
            "filter_reuse_fanout_under_operand_reuse": filter_reuse_fanout,
            "pipeline_provenance_class": pipeline_provenance_class,
        },
    }


def _summarize_proxy_ops(
    ops: list[dict],
    config: dict,
    *,
    bitstream_cfg: BitstreamEstimatorConfig,
) -> tuple[list[dict], dict[str, float | None]]:
    """Summarize per-op and total energy/latency using the proxy GEMM model."""
    results = []
    total_energy_j = 0.0
    total_latency_s = 0.0
    total_components_j = {
        "load_x": 0.0,
        "load_y": 0.0,
        "oe": 0.0,
        "adc_pca": 0.0,
        "detect": 0.0,
        "laser": 0.0,
        "mem": 0.0,
        "static": 0.0,
        "elementwise_electronic": 0.0,
        "bitstream_accumulator": 0.0,
    }
    support_class_inventory = _new_support_class_inventory()
    op_family_inventory = _new_op_family_inventory()
    workload_claim_inventory = _new_workload_claim_inventory()
    trust_posture_inventory = _new_trust_posture_inventory()
    true_sc_claim_state_inventory = _new_true_sc_claim_state_inventory()
    true_sc_claim_surface_inventory = _new_true_sc_claim_surface_inventory()
    true_sc_claim_surface_counts = {
        "native": 0,
        "governed_support": 0,
        "support_out_of_surface": 0,
        "out_of_surface": 0,
    }
    bitstream_datapath_stage_summary = _new_bitstream_datapath_stage_summary()
    model_abstraction_boundary = _summarize_model_abstraction_boundary(
        ops,
        estimator_mode="proxy",
    )
    bitstream_stream_state_policy = dict(bitstream_cfg.stream_state_policy)
    bitstream_stream_state_policy["model_abstraction_boundary"] = model_abstraction_boundary

    for op in ops:
        op_type = op.get("type", "gemm")
        elements = op.get("elements")
        if op_type in ELEMENTWISE_TYPES:
            est = estimate_elementwise_energy_latency(elements, op_type, config)
            latency_s = est["latency_s"]
            energy_j = est["energy_j"]
            power_w = (energy_j / latency_s) if latency_s > 0 else None
            total_components_j["elementwise_electronic"] += energy_j
            support_metadata = _row_support_metadata(op_type, "proxy")
            _bump_inventory(support_class_inventory, support_metadata["support_class"])
            _bump_inventory(op_family_inventory, support_metadata["op_family"])
            _bump_inventory(workload_claim_inventory, support_metadata["workload_claim_role"])
            _bump_inventory(trust_posture_inventory, support_metadata["trust_posture"])
            _bump_inventory(
                true_sc_claim_state_inventory,
                support_metadata["true_sc_claim_state"],
            )
            _update_true_sc_claim_surface_counts(
                metadata=support_metadata,
                inventory=true_sc_claim_surface_inventory,
                counts=true_sc_claim_surface_counts,
            )
            results.append(
                {
                    "name": op.get("name", "op"),
                    "type": op_type,
                    "estimator_mode": "proxy",
                    **support_metadata,
                    **_model_abstraction_boundary_metadata(op, "proxy"),
                    "m": None,
                    "d": None,
                    "n": None,
                    "elements": int(elements) if elements is not None else None,
                    "tiles": None,
                    "latency_ms": latency_s * 1e3,
                    "energy_mj": energy_j * 1e3,
                    "power_w": power_w,
                    "energy_mj_load_x": 0.0,
                    "energy_mj_load_y": 0.0,
                    "energy_mj_detect": 0.0,
                    "energy_mj_oe": 0.0,
                    "energy_mj_adc_pca": 0.0,
                    "energy_mj_laser": 0.0,
                    "energy_mj_mem": 0.0,
                    "energy_mj_static": 0.0,
                    "energy_mj_elementwise": energy_j * 1e3,
                    "energy_mj_bitstream_accumulator": 0.0,
                    "bitstream_effective_stream_length": _effective_stream_length(bitstream_cfg),
                    "generator_stream_state_policy_json": json.dumps(
                        bitstream_stream_state_policy,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "bitstream_calibration_applied": bitstream_cfg.calibration_applied,
                    "bitstream_calibration_summary_json": bitstream_cfg.calibration_summary_json,
                    "bitstream_calibration_reason": bitstream_cfg.calibration_reason,
                }
            )
        elif False and op_type == "conv2d":
            m = int(op["m"])
            d = int(op["d"])
            n = int(op["n"])
            est = estimate_bitstream_conv2d_energy_latency(
                m,
                d,
                n,
                config,
                kernel=op.get("kernel"),
                stride=op.get("stride"),
                groups=op.get("groups"),
                dilation=op.get("dilation"),
                runtime_stream_reuse_policy=DEFAULT_BITSTREAM_RUNTIME_STREAM_REUSE_POLICY,
                bitstream_cfg=bitstream_cfg,
            )
            latency_s = est["latency_s"]
            energy_j = est["energy_j"]
            power_w = (energy_j / latency_s) if latency_s > 0 else None
            components = est.get("energy_components_j") or {}
            total_components_j["load_x"] += float(components.get("load_x", 0.0))
            total_components_j["load_y"] += float(components.get("load_y", 0.0))
            total_components_j["oe"] += float(components.get("oe", 0.0))
            total_components_j["adc_pca"] += float(components.get("adc_pca", 0.0))
            total_components_j["detect"] += float(components.get("detect", 0.0))
            total_components_j["laser"] += float(components.get("laser", 0.0))
            total_components_j["mem"] += float(components.get("mem", 0.0))
            total_components_j["static"] += float(components.get("static", 0.0))
            total_components_j["bitstream_accumulator"] += float(
                components.get("bitstream_accumulator", 0.0)
            )
            bitstream_datapath_stages = est.get("bitstream_datapath_stages") or {}
            if isinstance(bitstream_datapath_stages, dict):
                _merge_bitstream_datapath_stage_summary(
                    bitstream_datapath_stage_summary,
                    bitstream_datapath_stages,
                )
            support_metadata = _row_support_metadata(op_type, "bitstream")
            _bump_inventory(support_class_inventory, support_metadata["support_class"])
            _bump_inventory(op_family_inventory, support_metadata["op_family"])
            _bump_inventory(workload_claim_inventory, support_metadata["workload_claim_role"])
            _bump_inventory(trust_posture_inventory, support_metadata["trust_posture"])
            _bump_inventory(
                true_sc_claim_state_inventory,
                support_metadata["true_sc_claim_state"],
            )
            _update_true_sc_claim_surface_counts(
                metadata=support_metadata,
                inventory=true_sc_claim_surface_inventory,
                counts=true_sc_claim_surface_counts,
            )
            bitstream_details = est.get("bitstream_details", {})
            results.append(
                {
                    "name": op.get("name", "op"),
                    "type": op_type,
                    "estimator_mode": "bitstream",
                    **support_metadata,
                    **_model_abstraction_boundary_metadata(op, "bitstream"),
                    "m": m,
                    "d": d,
                    "n": n,
                    "elements": None,
                    "tiles": est["tiles"],
                    "latency_ms": latency_s * 1e3,
                    "energy_mj": energy_j * 1e3,
                    "power_w": power_w,
                    "energy_mj_load_x": float(components.get("load_x", 0.0)) * 1e3,
                    "energy_mj_load_y": float(components.get("load_y", 0.0)) * 1e3,
                    "energy_mj_detect": float(components.get("detect", 0.0)) * 1e3,
                    "energy_mj_oe": float(components.get("oe", 0.0)) * 1e3,
                    "energy_mj_adc_pca": float(components.get("adc_pca", 0.0)) * 1e3,
                    "energy_mj_laser": float(components.get("laser", 0.0)) * 1e3,
                    "energy_mj_mem": float(components.get("mem", 0.0)) * 1e3,
                    "energy_mj_static": float(components.get("static", 0.0)) * 1e3,
                    "energy_mj_elementwise": 0.0,
                    "energy_mj_bitstream_accumulator": float(
                        components.get("bitstream_accumulator", 0.0)
                    )
                    * 1e3,
                    "bitstream_stream_length": bitstream_cfg.stream_length,
                    "bitstream_effective_stream_length": int(
                        bitstream_details.get("effective_stream_length")
                        or bitstream_cfg.stream_length
                    ),
                    "bitstream_effective_stream_length_scale": bitstream_details.get(
                        "effective_stream_length_scale"
                    ),
                    "bitstream_effective_stream_length_scale_provenance": (
                        bitstream_details.get("effective_stream_length_scale_provenance")
                    ),
                    "bitstream_parallel_passes": int(
                        bitstream_details.get("parallel_passes") or 0
                    ),
                    "bitstream_parallel_outputs": bitstream_details.get(
                        "parallel_outputs"
                    ),
                    "bitstream_parallel_outputs_provenance": bitstream_details.get(
                        "parallel_outputs_provenance"
                    ),
                    "bitstream_cycles_per_stream_bit": bitstream_details.get(
                        "cycles_per_stream_bit"
                    ),
                    "bitstream_cycles_per_stream_bit_provenance": bitstream_details.get(
                        "cycles_per_stream_bit_provenance"
                    ),
                    "bitstream_accumulator_energy_pj": bitstream_details.get(
                        "accumulator_energy_pj"
                    ),
                    "bitstream_accumulator_energy_pj_provenance": bitstream_details.get(
                        "accumulator_energy_pj_provenance"
                    ),
                    "bitstream_elementwise_parallelism_factor": bitstream_details.get(
                        "elementwise_parallelism_factor"
                    ),
                    "bitstream_elementwise_parallelism_provenance": bitstream_details.get(
                        "elementwise_parallelism_provenance"
                    ),
                    "bitstream_generator": bitstream_cfg.generator,
                    "generator_stream_state_policy_json": json.dumps(
                        bitstream_stream_state_policy,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "bitstream_capture_manifest_csv": bitstream_cfg.capture_manifest_csv,
                    "bitstream_calibration_applied": bitstream_cfg.calibration_applied,
                    "bitstream_calibration_summary_json": bitstream_cfg.calibration_summary_json,
                    "bitstream_calibration_reason": bitstream_cfg.calibration_reason,
                    "bitstream_datapath_stages_json": json.dumps(
                        bitstream_datapath_stages,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )
        else:
            m = int(op["m"])
            d = int(op["d"])
            n = int(op["n"])
            est = estimate_gemm_energy_latency(m, d, n, config)
            latency_s = est["latency_s"]
            energy_j = est["energy_j"]
            power_w = (energy_j / latency_s) if latency_s > 0 else None
            components = est.get("energy_components_j") or {}
            total_components_j["load_x"] += float(components.get("load_x", 0.0))
            total_components_j["load_y"] += float(components.get("load_y", 0.0))
            total_components_j["oe"] += float(components.get("oe", 0.0))
            total_components_j["adc_pca"] += float(components.get("adc_pca", 0.0))
            total_components_j["detect"] += float(components.get("detect", 0.0))
            total_components_j["laser"] += float(components.get("laser", 0.0))
            total_components_j["mem"] += float(components.get("mem", 0.0))
            total_components_j["static"] += float(components.get("static", 0.0))
            support_metadata = _row_support_metadata(op_type, "proxy")
            _bump_inventory(support_class_inventory, support_metadata["support_class"])
            _bump_inventory(op_family_inventory, support_metadata["op_family"])
            _bump_inventory(workload_claim_inventory, support_metadata["workload_claim_role"])
            _bump_inventory(trust_posture_inventory, support_metadata["trust_posture"])
            _bump_inventory(
                true_sc_claim_state_inventory,
                support_metadata["true_sc_claim_state"],
            )
            _update_true_sc_claim_surface_counts(
                metadata=support_metadata,
                inventory=true_sc_claim_surface_inventory,
                counts=true_sc_claim_surface_counts,
            )
            results.append(
                {
                    "name": op.get("name", "op"),
                    "type": op_type,
                    "estimator_mode": "proxy",
                    **support_metadata,
                    **_model_abstraction_boundary_metadata(op, "proxy"),
                    "m": m,
                    "d": d,
                    "n": n,
                    "elements": None,
                    "tiles": est["tiles"],
                    "latency_ms": latency_s * 1e3,
                    "energy_mj": energy_j * 1e3,
                    "power_w": power_w,
                    "energy_mj_load_x": float(components.get("load_x", 0.0)) * 1e3,
                    "energy_mj_load_y": float(components.get("load_y", 0.0)) * 1e3,
                    "energy_mj_detect": float(components.get("detect", 0.0)) * 1e3,
                    "energy_mj_oe": float(components.get("oe", 0.0)) * 1e3,
                    "energy_mj_adc_pca": float(components.get("adc_pca", 0.0)) * 1e3,
                    "energy_mj_laser": float(components.get("laser", 0.0)) * 1e3,
                    "energy_mj_mem": float(components.get("mem", 0.0)) * 1e3,
                    "energy_mj_static": float(components.get("static", 0.0)) * 1e3,
                    "energy_mj_elementwise": 0.0,
                    "energy_mj_bitstream_accumulator": 0.0,
                    "bitstream_effective_stream_length": _effective_stream_length(bitstream_cfg),
                    "generator_stream_state_policy_json": json.dumps(
                        bitstream_stream_state_policy,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )
        total_energy_j += energy_j
        total_latency_s += latency_s

    total_power_w = (total_energy_j / total_latency_s) if total_latency_s > 0 else None
    workload_fidelity = _derive_workload_fidelity(
        model_abstraction_boundary=model_abstraction_boundary,
        support_class_inventory=support_class_inventory,
        workload_claim_inventory=workload_claim_inventory,
    )
    summary = {
        "estimator_mode": "proxy",
        "total_latency_ms": total_latency_s * 1e3,
        "total_energy_mj": total_energy_j * 1e3,
        "total_power_w": total_power_w,
        "energy_mj_load_x": total_components_j["load_x"] * 1e3,
        "energy_mj_load_y": total_components_j["load_y"] * 1e3,
        "energy_mj_detect": total_components_j["detect"] * 1e3,
        "energy_mj_oe": total_components_j["oe"] * 1e3,
        "energy_mj_adc_pca": total_components_j["adc_pca"] * 1e3,
        "energy_mj_laser": total_components_j["laser"] * 1e3,
        "energy_mj_mem": total_components_j["mem"] * 1e3,
        "energy_mj_static": total_components_j["static"] * 1e3,
        "energy_mj_elementwise": total_components_j["elementwise_electronic"] * 1e3,
        "energy_mj_bitstream_accumulator": 0.0,
        "support_class_inventory": support_class_inventory,
        "workload_claim_inventory": workload_claim_inventory,
        "model_abstraction_boundary": model_abstraction_boundary,
        "conv2d_native_runtime_modeled_op_count": model_abstraction_boundary.get(
            "conv2d_native_runtime_modeled_op_count", 0
        ),
        "conv_fidelity_stage": (
            CONV_FIDELITY_STAGE_RUNTIME_MODELED
            if model_abstraction_boundary.get("conv2d_native_runtime_modeled_op_count", 0)
            else None
        ),
        "conv_fidelity_blockers": (
            conv_fidelity_blockers_for_stage(CONV_FIDELITY_STAGE_RUNTIME_MODELED)
            if model_abstraction_boundary.get("conv2d_native_runtime_modeled_op_count", 0)
            else []
        ),
        **workload_fidelity,
        "bitstream_stream_state_policy": bitstream_stream_state_policy,
        "generator_stream_state_policy": bitstream_stream_state_policy,
        "estimation_model_coverage_status": _estimation_model_coverage_status(
            support_class_inventory
        ),
        "estimation_model_coverage_reason": (
            "unsupported_operator_models_present"
            if int(support_class_inventory.get(SC_DEFAULT_UNSUPPORTED, 0)) > 0
            else "all_ops_have_native_or_governed_models"
        ),
        "estimation_model_support_boundary": _estimation_model_support_boundary(
            support_class_inventory
        ),
        "estimation_model_supported_op_count": int(
            support_class_inventory.get(SC_NATIVE_BITSTREAM, 0)
        )
        + int(support_class_inventory.get(SC_GOVERNED_ELECTRONIC_SUPPORT, 0)),
        "estimation_model_unsupported_op_count": int(
            support_class_inventory.get(SC_DEFAULT_UNSUPPORTED, 0)
        ),
        "workload_native_op_count": int(
            workload_claim_inventory.get(WORKLOAD_CLAIM_ROLE_NATIVE, 0)
        ),
        "workload_support_qualified_op_count": int(
            workload_claim_inventory.get(WORKLOAD_CLAIM_ROLE_SUPPORT_QUALIFIED, 0)
        ),
        "workload_support_blocking_op_count": int(
            workload_claim_inventory.get(WORKLOAD_CLAIM_ROLE_BLOCKED_BY_SUPPORT, 0)
        ),
        "workload_out_of_surface_op_count": int(
            workload_claim_inventory.get(WORKLOAD_CLAIM_ROLE_OUT, 0)
        ),
        "op_family_inventory": op_family_inventory,
        "trust_posture_inventory": trust_posture_inventory,
        "trust_posture": _summary_trust_posture(trust_posture_inventory),
        "true_sc_claim_state_inventory": true_sc_claim_state_inventory,
        "true_sc_summary_claim_state": _summary_true_sc_claim_state(
            true_sc_claim_state_inventory,
            workload_fidelity=workload_fidelity,
        ),
        "true_sc_claim_surface_inventory": true_sc_claim_surface_inventory,
        "true_sc_claim_surface_status": _summary_true_sc_claim_surface_status(
            native_op_count=true_sc_claim_surface_counts["native"],
            governed_support_op_count=true_sc_claim_surface_counts["governed_support"],
            support_out_of_surface_op_count=true_sc_claim_surface_counts[
                "support_out_of_surface"
            ],
            out_of_claim_surface_op_count=true_sc_claim_surface_counts["out_of_surface"],
            workload_fidelity=workload_fidelity,
        ),
        "true_sc_claim_surface_native_op_count": true_sc_claim_surface_counts["native"],
        "true_sc_claim_surface_governed_support_op_count": true_sc_claim_surface_counts[
            "governed_support"
        ],
        "true_sc_support_out_of_surface_op_count": true_sc_claim_surface_counts[
            "support_out_of_surface"
        ],
        "true_sc_out_of_claim_surface_op_count": true_sc_claim_surface_counts[
            "out_of_surface"
        ],
        "true_sc_native_op_count": true_sc_claim_state_inventory[TRUE_SC_NATIVE],
        "true_sc_governed_not_true_sc_op_count": true_sc_claim_state_inventory[
            GOVERNED_SUPPORT_NOT_TRUE_SC
        ],
        "true_sc_out_of_surface_op_count": true_sc_claim_state_inventory[
            TRUE_SC_OUT_OF_SURFACE
        ],
    }
    return results, summary


def _summarize_bitstream_ops(
    ops: list[dict],
    config: dict,
    *,
    bitstream_cfg: BitstreamEstimatorConfig,
) -> tuple[list[dict], dict[str, float | None]]:
    """Summarize per-op and total energy/latency using the bitstream model."""
    results = []
    total_energy_j = 0.0
    total_latency_s = 0.0
    total_components_j = {
        "load_x": 0.0,
        "load_y": 0.0,
        "oe": 0.0,
        "adc_pca": 0.0,
        "detect": 0.0,
        "laser": 0.0,
        "mem": 0.0,
        "static": 0.0,
        "elementwise_electronic": 0.0,
        "bitstream_accumulator": 0.0,
    }
    support_class_inventory = _new_support_class_inventory()
    op_family_inventory = _new_op_family_inventory()
    workload_claim_inventory = _new_workload_claim_inventory()
    trust_posture_inventory = _new_trust_posture_inventory()
    true_sc_claim_state_inventory = _new_true_sc_claim_state_inventory()
    true_sc_claim_surface_inventory = _new_true_sc_claim_surface_inventory()
    true_sc_claim_surface_counts = {
        "native": 0,
        "governed_support": 0,
        "support_out_of_surface": 0,
        "out_of_surface": 0,
    }
    bitstream_datapath_stage_summary = _new_bitstream_datapath_stage_summary()
    model_abstraction_boundary = _summarize_model_abstraction_boundary(
        ops,
        estimator_mode="bitstream",
    )
    bitstream_stream_state_policy = dict(bitstream_cfg.stream_state_policy)
    bitstream_stream_state_policy["model_abstraction_boundary"] = model_abstraction_boundary

    for op in ops:
        op_type = op.get("type", "gemm")
        elements = op.get("elements")
        if op_type in ELEMENTWISE_TYPES:
            support_metadata = _row_support_metadata(op_type, "bitstream")
            _bump_inventory(support_class_inventory, support_metadata["support_class"])
            _bump_inventory(op_family_inventory, support_metadata["op_family"])
            _bump_inventory(workload_claim_inventory, support_metadata["workload_claim_role"])
            _bump_inventory(trust_posture_inventory, support_metadata["trust_posture"])
            _bump_inventory(
                true_sc_claim_state_inventory,
                support_metadata["true_sc_claim_state"],
            )
            _update_true_sc_claim_surface_counts(
                metadata=support_metadata,
                inventory=true_sc_claim_surface_inventory,
                counts=true_sc_claim_surface_counts,
            )
            if support_metadata["support_class"] == SC_NATIVE_BITSTREAM:
                est = estimate_bitstream_elementwise_energy_latency(
                    elements,
                    op_type,
                    config,
                    bitstream_cfg=bitstream_cfg,
                )
                latency_s = float(est["latency_s"])
                energy_j = float(est["energy_j"])
                power_w = (energy_j / latency_s) if latency_s > 0 else None
                components = est.get("energy_components_j") or {}
                total_components_j["load_x"] += float(components.get("load_x", 0.0))
                total_components_j["load_y"] += float(components.get("load_y", 0.0))
                total_components_j["oe"] += float(components.get("oe", 0.0))
                total_components_j["adc_pca"] += float(components.get("adc_pca", 0.0))
                total_components_j["detect"] += float(components.get("detect", 0.0))
                total_components_j["laser"] += float(components.get("laser", 0.0))
                total_components_j["mem"] += float(components.get("mem", 0.0))
                total_components_j["static"] += float(components.get("static", 0.0))
                total_components_j["bitstream_accumulator"] += float(
                    components.get("bitstream_accumulator", 0.0)
                )
                bitstream_datapath_stages = est.get("bitstream_datapath_stages") or {}
                if isinstance(bitstream_datapath_stages, dict):
                    _merge_bitstream_datapath_stage_summary(
                        bitstream_datapath_stage_summary,
                        bitstream_datapath_stages,
                    )
                bitstream_details = est.get("bitstream_details", {})
                results.append(
                    {
                        "name": op.get("name", "op"),
                        "type": op_type,
                        "estimator_mode": "bitstream",
                        **support_metadata,
                        **_model_abstraction_boundary_metadata(op, "bitstream"),
                        "m": None,
                        "d": None,
                        "n": None,
                        "elements": int(elements) if elements is not None else None,
                        "tiles": None,
                        "latency_ms": latency_s * 1e3,
                        "energy_mj": energy_j * 1e3,
                        "power_w": power_w,
                        "energy_mj_load_x": float(components.get("load_x", 0.0)) * 1e3,
                        "energy_mj_load_y": float(components.get("load_y", 0.0)) * 1e3,
                        "energy_mj_detect": float(components.get("detect", 0.0)) * 1e3,
                        "energy_mj_oe": float(components.get("oe", 0.0)) * 1e3,
                        "energy_mj_adc_pca": float(components.get("adc_pca", 0.0)) * 1e3,
                        "energy_mj_laser": float(components.get("laser", 0.0)) * 1e3,
                        "energy_mj_mem": float(components.get("mem", 0.0)) * 1e3,
                        "energy_mj_static": float(components.get("static", 0.0)) * 1e3,
                        "energy_mj_elementwise": 0.0,
                        "energy_mj_bitstream_accumulator": float(
                            components.get("bitstream_accumulator", 0.0)
                        )
                        * 1e3,
                        "bitstream_stream_length": bitstream_cfg.stream_length,
                        "bitstream_effective_stream_length": int(
                            bitstream_details.get("effective_stream_length")
                            or bitstream_cfg.stream_length
                        ),
                        "bitstream_effective_stream_length_scale": bitstream_details.get(
                            "effective_stream_length_scale"
                        ),
                        "bitstream_effective_stream_length_scale_provenance": (
                            bitstream_details.get("effective_stream_length_scale_provenance")
                        ),
                        "bitstream_parallel_passes": int(
                            bitstream_details.get("parallel_passes") or 0
                        ),
                        "bitstream_parallel_outputs": bitstream_details.get(
                            "parallel_outputs"
                        ),
                        "bitstream_parallel_outputs_provenance": bitstream_details.get(
                            "parallel_outputs_provenance"
                        ),
                        "bitstream_cycles_per_stream_bit": bitstream_details.get(
                            "cycles_per_stream_bit"
                        ),
                        "bitstream_cycles_per_stream_bit_provenance": bitstream_details.get(
                            "cycles_per_stream_bit_provenance"
                        ),
                        "bitstream_accumulator_energy_pj": bitstream_details.get(
                            "accumulator_energy_pj"
                        ),
                        "bitstream_accumulator_energy_pj_provenance": bitstream_details.get(
                            "accumulator_energy_pj_provenance"
                        ),
                        "bitstream_elementwise_parallelism_factor": bitstream_details.get(
                            "elementwise_parallelism_factor"
                        ),
                        "bitstream_elementwise_parallelism_provenance": bitstream_details.get(
                            "elementwise_parallelism_provenance"
                        ),
                        "bitstream_generator": bitstream_cfg.generator,
                        "generator_stream_state_policy_json": json.dumps(
                            bitstream_stream_state_policy,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "bitstream_capture_manifest_csv": bitstream_cfg.capture_manifest_csv,
                        "bitstream_calibration_applied": bitstream_cfg.calibration_applied,
                        "bitstream_calibration_summary_json": bitstream_cfg.calibration_summary_json,
                        "bitstream_calibration_reason": bitstream_cfg.calibration_reason,
                        "bitstream_datapath_stages_json": json.dumps(
                            bitstream_datapath_stages,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    }
                )
            else:
                est = estimate_elementwise_energy_latency(elements, op_type, config)
                latency_s = est["latency_s"]
                energy_j = est["energy_j"]
                power_w = (energy_j / latency_s) if latency_s > 0 else None
                total_components_j["elementwise_electronic"] += energy_j
                results.append(
                    {
                        "name": op.get("name", "op"),
                        "type": op_type,
                        "estimator_mode": "bitstream",
                        **support_metadata,
                        **_model_abstraction_boundary_metadata(op, "bitstream"),
                        "m": None,
                        "d": None,
                        "n": None,
                        "elements": int(elements) if elements is not None else None,
                        "tiles": None,
                        "latency_ms": latency_s * 1e3,
                        "energy_mj": energy_j * 1e3,
                        "power_w": power_w,
                        "energy_mj_load_x": 0.0,
                        "energy_mj_load_y": 0.0,
                        "energy_mj_detect": 0.0,
                        "energy_mj_oe": 0.0,
                        "energy_mj_adc_pca": 0.0,
                        "energy_mj_laser": 0.0,
                        "energy_mj_mem": 0.0,
                        "energy_mj_static": 0.0,
                        "energy_mj_elementwise": energy_j * 1e3,
                        "energy_mj_bitstream_accumulator": 0.0,
                        "generator_stream_state_policy_json": json.dumps(
                            bitstream_stream_state_policy,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    }
                )
        else:
            m = int(op["m"])
            d = int(op["d"])
            n = int(op["n"])
            est = estimate_bitstream_gemm_energy_latency(m, d, n, config, bitstream_cfg=bitstream_cfg)
            latency_s = est["latency_s"]
            energy_j = est["energy_j"]
            power_w = (energy_j / latency_s) if latency_s > 0 else None
            components = est.get("energy_components_j") or {}
            total_components_j["load_x"] += float(components.get("load_x", 0.0))
            total_components_j["load_y"] += float(components.get("load_y", 0.0))
            total_components_j["oe"] += float(components.get("oe", 0.0))
            total_components_j["adc_pca"] += float(components.get("adc_pca", 0.0))
            total_components_j["detect"] += float(components.get("detect", 0.0))
            total_components_j["laser"] += float(components.get("laser", 0.0))
            total_components_j["mem"] += float(components.get("mem", 0.0))
            total_components_j["static"] += float(components.get("static", 0.0))
            total_components_j["bitstream_accumulator"] += float(
                components.get("bitstream_accumulator", 0.0)
            )
            bitstream_datapath_stages = est.get("bitstream_datapath_stages") or {}
            if isinstance(bitstream_datapath_stages, dict):
                _merge_bitstream_datapath_stage_summary(
                    bitstream_datapath_stage_summary,
                    bitstream_datapath_stages,
                )
            support_metadata = _row_support_metadata(op_type, "bitstream")
            _bump_inventory(support_class_inventory, support_metadata["support_class"])
            _bump_inventory(op_family_inventory, support_metadata["op_family"])
            _bump_inventory(workload_claim_inventory, support_metadata["workload_claim_role"])
            _bump_inventory(trust_posture_inventory, support_metadata["trust_posture"])
            _bump_inventory(
                true_sc_claim_state_inventory,
                support_metadata["true_sc_claim_state"],
            )
            _update_true_sc_claim_surface_counts(
                metadata=support_metadata,
                inventory=true_sc_claim_surface_inventory,
                counts=true_sc_claim_surface_counts,
            )
            results.append(
                {
                    "name": op.get("name", "op"),
                    "type": op_type,
                    "estimator_mode": "bitstream",
                    **support_metadata,
                    **_model_abstraction_boundary_metadata(op, "bitstream"),
                    "m": m,
                    "d": d,
                    "n": n,
                    "elements": None,
                    "tiles": est["tiles"],
                    "latency_ms": latency_s * 1e3,
                    "energy_mj": energy_j * 1e3,
                    "power_w": power_w,
                    "energy_mj_load_x": float(components.get("load_x", 0.0)) * 1e3,
                    "energy_mj_load_y": float(components.get("load_y", 0.0)) * 1e3,
                    "energy_mj_detect": float(components.get("detect", 0.0)) * 1e3,
                    "energy_mj_oe": float(components.get("oe", 0.0)) * 1e3,
                    "energy_mj_adc_pca": float(components.get("adc_pca", 0.0)) * 1e3,
                    "energy_mj_laser": float(components.get("laser", 0.0)) * 1e3,
                    "energy_mj_mem": float(components.get("mem", 0.0)) * 1e3,
                    "energy_mj_static": float(components.get("static", 0.0)) * 1e3,
                    "energy_mj_elementwise": 0.0,
                    "energy_mj_bitstream_accumulator": float(
                        components.get("bitstream_accumulator", 0.0)
                    )
                    * 1e3,
                    "bitstream_stream_length": bitstream_cfg.stream_length,
                    "bitstream_effective_stream_length": int(
                        est.get("bitstream_details", {}).get("effective_stream_length")
                        or bitstream_cfg.stream_length
                    ),
                    "bitstream_effective_stream_length_scale": est.get(
                        "bitstream_details", {}
                    ).get("effective_stream_length_scale"),
                    "bitstream_effective_stream_length_scale_provenance": est.get(
                        "bitstream_details", {}
                    ).get("effective_stream_length_scale_provenance"),
                    "bitstream_parallel_passes": int(
                        est.get("bitstream_details", {}).get("parallel_passes") or 0
                    ),
                    "bitstream_parallel_outputs": est.get("bitstream_details", {}).get(
                        "parallel_outputs"
                    ),
                    "bitstream_parallel_outputs_provenance": est.get(
                        "bitstream_details", {}
                    ).get("parallel_outputs_provenance"),
                    "bitstream_cycles_per_stream_bit": est.get("bitstream_details", {}).get(
                        "cycles_per_stream_bit"
                    ),
                    "bitstream_cycles_per_stream_bit_provenance": est.get(
                        "bitstream_details", {}
                    ).get("cycles_per_stream_bit_provenance"),
                    "bitstream_accumulator_energy_pj": est.get("bitstream_details", {}).get(
                        "accumulator_energy_pj"
                    ),
                    "bitstream_accumulator_energy_pj_provenance": est.get(
                        "bitstream_details", {}
                    ).get("accumulator_energy_pj_provenance"),
                    "bitstream_elementwise_parallelism_factor": est.get(
                        "bitstream_details", {}
                    ).get("elementwise_parallelism_factor"),
                    "bitstream_elementwise_parallelism_provenance": est.get(
                        "bitstream_details", {}
                    ).get("elementwise_parallelism_provenance"),
                    "bitstream_generator": bitstream_cfg.generator,
                    "generator_stream_state_policy_json": json.dumps(
                        bitstream_stream_state_policy,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "bitstream_capture_manifest_csv": bitstream_cfg.capture_manifest_csv,
                    "bitstream_calibration_applied": bitstream_cfg.calibration_applied,
                    "bitstream_calibration_summary_json": bitstream_cfg.calibration_summary_json,
                    "bitstream_calibration_reason": bitstream_cfg.calibration_reason,
                    "bitstream_datapath_stages_json": json.dumps(
                        bitstream_datapath_stages,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )
        total_energy_j += energy_j
        total_latency_s += latency_s

    total_power_w = (total_energy_j / total_latency_s) if total_latency_s > 0 else None
    workload_fidelity = _derive_workload_fidelity(
        model_abstraction_boundary=model_abstraction_boundary,
        support_class_inventory=support_class_inventory,
        workload_claim_inventory=workload_claim_inventory,
    )
    summary = {
        "estimator_mode": "bitstream",
        "total_latency_ms": total_latency_s * 1e3,
        "total_energy_mj": total_energy_j * 1e3,
        "total_power_w": total_power_w,
        "energy_mj_load_x": total_components_j["load_x"] * 1e3,
        "energy_mj_load_y": total_components_j["load_y"] * 1e3,
        "energy_mj_detect": total_components_j["detect"] * 1e3,
        "energy_mj_oe": total_components_j["oe"] * 1e3,
        "energy_mj_adc_pca": total_components_j["adc_pca"] * 1e3,
        "energy_mj_laser": total_components_j["laser"] * 1e3,
        "energy_mj_mem": total_components_j["mem"] * 1e3,
        "energy_mj_static": total_components_j["static"] * 1e3,
        "energy_mj_elementwise": total_components_j["elementwise_electronic"] * 1e3,
        "energy_mj_bitstream_accumulator": total_components_j["bitstream_accumulator"] * 1e3,
        "bitstream_stream_length": bitstream_cfg.stream_length,
        "bitstream_effective_stream_length": _effective_stream_length(bitstream_cfg),
        "bitstream_generator": bitstream_cfg.generator,
        "bitstream_stream_state_policy": bitstream_stream_state_policy,
        "generator_stream_state_policy": bitstream_stream_state_policy,
        "bitstream_capture_manifest_csv": bitstream_cfg.capture_manifest_csv,
        "bitstream_calibration_applied": bitstream_cfg.calibration_applied,
        "bitstream_calibration_summary_json": bitstream_cfg.calibration_summary_json,
        "bitstream_calibration_capture_row_count": bitstream_cfg.calibration_capture_row_count,
        "bitstream_calibration_replay_row_count": bitstream_cfg.calibration_replay_row_count,
        "bitstream_calibration_module_count": bitstream_cfg.calibration_module_count,
        "bitstream_calibration_median_abs_error": bitstream_cfg.calibration_median_abs_error,
        "bitstream_calibration_max_abs_error": bitstream_cfg.calibration_max_abs_error,
        "bitstream_calibration_median_relative_error": bitstream_cfg.calibration_median_relative_error,
        "bitstream_calibration_reason": bitstream_cfg.calibration_reason,
        "support_class_inventory": support_class_inventory,
        "workload_claim_inventory": workload_claim_inventory,
        "model_abstraction_boundary": model_abstraction_boundary,
        "conv2d_native_runtime_modeled_op_count": model_abstraction_boundary.get(
            "conv2d_native_runtime_modeled_op_count", 0
        ),
        "conv_fidelity_stage": (
            CONV_FIDELITY_STAGE_RUNTIME_MODELED
            if model_abstraction_boundary.get("conv2d_native_runtime_modeled_op_count", 0)
            else None
        ),
        "conv_fidelity_blockers": (
            conv_fidelity_blockers_for_stage(CONV_FIDELITY_STAGE_RUNTIME_MODELED)
            if model_abstraction_boundary.get("conv2d_native_runtime_modeled_op_count", 0)
            else []
        ),
        **_bitstream_scalar_summary(bitstream_cfg),
        **workload_fidelity,
        "estimation_model_coverage_status": _estimation_model_coverage_status(
            support_class_inventory
        ),
        "estimation_model_coverage_reason": (
            "unsupported_operator_models_present"
            if int(support_class_inventory.get(SC_DEFAULT_UNSUPPORTED, 0)) > 0
            else "all_ops_have_native_or_governed_models"
        ),
        "estimation_model_support_boundary": _estimation_model_support_boundary(
            support_class_inventory
        ),
        "estimation_model_supported_op_count": int(
            support_class_inventory.get(SC_NATIVE_BITSTREAM, 0)
        )
        + int(support_class_inventory.get(SC_GOVERNED_ELECTRONIC_SUPPORT, 0)),
        "estimation_model_unsupported_op_count": int(
            support_class_inventory.get(SC_DEFAULT_UNSUPPORTED, 0)
        ),
        "workload_native_op_count": int(
            workload_claim_inventory.get(WORKLOAD_CLAIM_ROLE_NATIVE, 0)
        ),
        "workload_support_qualified_op_count": int(
            workload_claim_inventory.get(WORKLOAD_CLAIM_ROLE_SUPPORT_QUALIFIED, 0)
        ),
        "workload_support_blocking_op_count": int(
            workload_claim_inventory.get(WORKLOAD_CLAIM_ROLE_BLOCKED_BY_SUPPORT, 0)
        ),
        "workload_out_of_surface_op_count": int(
            workload_claim_inventory.get(WORKLOAD_CLAIM_ROLE_OUT, 0)
        ),
        "op_family_inventory": op_family_inventory,
        "trust_posture_inventory": trust_posture_inventory,
        "trust_posture": _summary_trust_posture(trust_posture_inventory),
        "true_sc_claim_state_inventory": true_sc_claim_state_inventory,
        "true_sc_summary_claim_state": _summary_true_sc_claim_state(
            true_sc_claim_state_inventory,
            workload_fidelity=workload_fidelity,
        ),
        "true_sc_claim_surface_inventory": true_sc_claim_surface_inventory,
        "true_sc_claim_surface_status": _summary_true_sc_claim_surface_status(
            native_op_count=true_sc_claim_surface_counts["native"],
            governed_support_op_count=true_sc_claim_surface_counts["governed_support"],
            support_out_of_surface_op_count=true_sc_claim_surface_counts[
                "support_out_of_surface"
            ],
            out_of_claim_surface_op_count=true_sc_claim_surface_counts["out_of_surface"],
            workload_fidelity=workload_fidelity,
        ),
        "bitstream_datapath_stage_summary": bitstream_datapath_stage_summary,
        "true_sc_claim_surface_native_op_count": true_sc_claim_surface_counts["native"],
        "true_sc_claim_surface_governed_support_op_count": true_sc_claim_surface_counts[
            "governed_support"
        ],
        "true_sc_support_out_of_surface_op_count": true_sc_claim_surface_counts[
            "support_out_of_surface"
        ],
        "true_sc_out_of_claim_surface_op_count": true_sc_claim_surface_counts[
            "out_of_surface"
        ],
        "true_sc_native_op_count": true_sc_claim_state_inventory[TRUE_SC_NATIVE],
        "true_sc_governed_not_true_sc_op_count": true_sc_claim_state_inventory[
            GOVERNED_SUPPORT_NOT_TRUE_SC
        ],
        "true_sc_out_of_surface_op_count": true_sc_claim_state_inventory[
            TRUE_SC_OUT_OF_SURFACE
        ],
    }
    return results, summary


def summarize_ops(ops: list[dict], config: dict) -> tuple[list[dict], dict[str, float | None]]:
    """Summarize per-op and total energy/latency."""
    bitstream_cfg = _resolve_bitstream_estimator_config(config)
    if bitstream_cfg.enabled:
        return _summarize_bitstream_ops(ops, config, bitstream_cfg=bitstream_cfg)
    return _summarize_proxy_ops(ops, config, bitstream_cfg=bitstream_cfg)


__all__ = [
    "ELEMENTWISE_TYPES",
    "BITSTREAM_NATIVE_ELEMENTWISE_TYPES",
    "BITSTREAM_WORKLOAD_SUPPORT_QUALIFIED_TYPES",
    "build_electronic_params",
    "build_energy_params",
    "BitstreamEstimatorConfig",
    "DEFAULT_WITH_SUPPORTING_ASSUMPTIONS",
    "GOVERNED_SUPPORT_NOT_TRUE_SC",
    "OP_FAMILY_ELEMENTWISE",
    "OP_FAMILY_GEMM_LIKE",
    "OUT_OF_BAND",
    "SC_DEFAULT_UNSUPPORTED",
    "SC_GOVERNED_ELECTRONIC_SUPPORT",
    "SC_NATIVE_BITSTREAM",
    "TRUE_SC_CLAIM_SURFACE_BLOCKED",
    "TRUE_SC_CLAIM_SURFACE_EMPTY",
    "TRUE_SC_CLAIM_SURFACE_FULL",
    "TRUE_SC_CLAIM_SURFACE_IN",
    "TRUE_SC_CLAIM_SURFACE_LIMITED",
    "TRUE_SC_CLAIM_SURFACE_OUT",
    "TRUE_SC_CLAIM_SURFACE_SUPPORT_OUT",
    "TRUE_SC_NATIVE",
    "TRUE_SC_OUT_OF_SURFACE",
    "estimate_bitstream_elementwise_energy_latency",
    "estimate_bitstream_conv2d_energy_latency",
    "estimate_bitstream_gemm_energy_latency",
    "estimate_elementwise_energy_latency",
    "estimate_gemm_energy_latency",
    "summarize_ops",
    "TRUSTED_DEFAULT",
    "WORKLOAD_FIDELITY_STATUS_SUPPORT_QUALIFIED_READY",
]
