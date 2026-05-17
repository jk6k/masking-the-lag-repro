"""Lightweight HPAT energy/latency estimator for photonic GEMM and electronic ops."""

from __future__ import annotations

import json
import math

from accuracy.bitstream_conv_semantics import (
    CONV_FIDELITY_STAGE_RUNTIME_MODELED,
    build_conv_runtime_semantics,
    classify_conv_native_class,
    conv_fidelity_blockers_for_stage,
)
from accuracy.mlx_mobilevit import DEFAULT_BITSTREAM_RUNTIME_STREAM_REUSE_POLICY

ELEMENTWISE_TYPES = {"softmax", "norm", "layer_norm", "batch_norm", "group_norm", "activation"}

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
MODEL_ABSTRACTION_STATUS_SUPPORT_ONLY = "support_only"
MODEL_ABSTRACTION_STATUS_OUT_OF_SURFACE = "out_of_surface"

WORKLOAD_FIDELITY_CLASS_NATIVE = "native_workload_fidelity"
WORKLOAD_FIDELITY_CLASS_APPROXIMATE = "approximate_workload_fidelity"
WORKLOAD_FIDELITY_CLASS_OUT_OF_SURFACE = "out_of_surface_workload_fidelity"
WORKLOAD_FIDELITY_STATUS_NATIVE_READY = "native_ready"
WORKLOAD_FIDELITY_STATUS_APPROXIMATE = "approximate"
WORKLOAD_FIDELITY_STATUS_OUT_OF_SURFACE = "out_of_surface"


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
            "boundary_reason": (
                "conv2d is classified with a native runtime-path contract and "
                "explicit runtime semantics, but hardware evidence remains "
                "unclosed"
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
        if estimator_mode == "bitstream" and op_type in {"norm", "layer_norm", "batch_norm", "group_norm", "activation"}:
            kind = MODEL_ABSTRACTION_KIND_NATIVE_ELECTRONIC_BITSTREAM
            status = MODEL_ABSTRACTION_STATUS_NATIVE
            reason = "electronic bitstream elementwise support surface"
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

    return {
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


def _derive_workload_fidelity(
    *,
    model_abstraction_boundary: dict[str, object],
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
    elif boundary_status == MODEL_ABSTRACTION_STATUS_OUT_OF_SURFACE:
        workload_fidelity_class = WORKLOAD_FIDELITY_CLASS_OUT_OF_SURFACE
        workload_fidelity_status = WORKLOAD_FIDELITY_STATUS_OUT_OF_SURFACE
        blockers.append("operator_model_out_of_surface")
        workload_fidelity_reason = "required_operator_model_out_of_surface"
    return {
        "workload_fidelity_class": workload_fidelity_class,
        "workload_fidelity_status": workload_fidelity_status,
        "workload_fidelity_reason": workload_fidelity_reason,
        "workload_fidelity_blockers": blockers,
        "workload_native_claim_eligible": not blockers,
    }


def _bitstream_requested(config: dict) -> bool:
    from mtl_model.estimator import _resolve_bitstream_estimator_config

    return _resolve_bitstream_estimator_config(config).enabled


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

    # B8 fix: energy is linear (all elements are processed), but latency
    # accounts for hardware parallelism (e.g. SIMD/vector units).
    electronic = config.get("electronic", {})
    parallelism = max(1, int(electronic.get("parallelism_factor", 32)))
    energy_j = elements * energy_pj * 1e-12
    latency_s = (elements / parallelism) * latency_ns * 1e-9
    return {"latency_s": latency_s, "energy_j": energy_j}


def estimate_gemm_energy_latency(m: int, d: int, n: int, config: dict) -> dict[str, object]:
    """Estimate GEMM energy/latency using photonic parameters."""
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

    # Eq. 11-style compute energy model (parameterized), in joules.
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


def summarize_ops(ops: list[dict], config: dict) -> tuple[list[dict], dict[str, float | None]]:
    """Summarize per-op and total energy/latency."""
    if _bitstream_requested(config):
        # Import lazily so the stable proxy path keeps its current package
        # surface and we avoid package-level circular-import churn.
        from mtl_model.estimator import summarize_ops as summarize_bitstream_ops

        return summarize_bitstream_ops(ops, config)

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
    }
    model_abstraction_boundary = _summarize_model_abstraction_boundary(
        ops,
        estimator_mode="proxy",
    )
    generator_stream_state_policy = {"model_abstraction_boundary": model_abstraction_boundary}

    for op in ops:
        op_type = op.get("type", "gemm")
        elements = op.get("elements")
        if op_type in ELEMENTWISE_TYPES:
            # Elementwise ops (softmax/norm/activation) use electronic-domain params.
            est = estimate_elementwise_energy_latency(elements, op_type, config)
            latency_s = est["latency_s"]
            energy_j = est["energy_j"]
            power_w = (energy_j / latency_s) if latency_s > 0 else None
            total_components_j["elementwise_electronic"] += energy_j
            results.append(
                {
                    "name": op.get("name", "op"),
                    "type": op_type,
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
                    "generator_stream_state_policy_json": json.dumps(
                        generator_stream_state_policy,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    **_model_abstraction_boundary_metadata(op, "proxy"),
                }
            )
        else:
            # GEMM-like ops use photonic tile parameters and energy model.
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
            results.append(
                {
                    "name": op.get("name", "op"),
                    "type": op_type,
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
                    "generator_stream_state_policy_json": json.dumps(
                        generator_stream_state_policy,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    **_model_abstraction_boundary_metadata(op, "proxy"),
                }
            )
        total_energy_j += energy_j
        total_latency_s += latency_s

    total_power_w = (total_energy_j / total_latency_s) if total_latency_s > 0 else None
    workload_fidelity = _derive_workload_fidelity(
        model_abstraction_boundary=model_abstraction_boundary,
    )
    summary = {
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
        "generator_stream_state_policy": generator_stream_state_policy,
    }
    return results, summary


__all__ = [
    "ELEMENTWISE_TYPES",
    "build_electronic_params",
    "build_energy_params",
    "estimate_elementwise_energy_latency",
    "estimate_gemm_energy_latency",
    "summarize_ops",
]
