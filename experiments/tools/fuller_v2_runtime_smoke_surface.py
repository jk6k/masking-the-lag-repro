#!/usr/bin/env python3
"""Shared result-surface helpers for FULLER v2 engineering-smoke lanes."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = ROOT_DIR / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from exp_common.det_policy import resolve_det_runtime_metadata
from exp_common.meso_cost_model import compute_meso_cost_model, resolve_meso_load_scale
from exp_common.phy_link_budget import compute_link_budget
from hpat_model.hpat_model import summarize_ops
from tools.fuller_phase1_registry import default_variant_descriptor_for_experiment
from tools.phase1_runner import (
    _aggregate_buffer_trace_for_model,
    _aggregate_flow_diagnostics_for_model,
    _aggregate_timeline_for_model,
    _apply_realism_profile_if_present,
    _apply_workload_scale_to_ops,
    _build_per_layer_timeline_rows,
    _build_realism_assessment,
    _calibrate_p1_alignment,
    _cfg_value,
    _compute_bsl_scale,
    _compute_integrated_system_costs,
    _compute_sparse_scale,
    _load_ops,
    _merge_execution_semantics_into_estimator_config,
    _resolve_execution_semantics_cfg,
    _resolve_existing_path,
    _resolve_flow_summary_fields,
    _resolve_integrated_system_cost_cfg,
    _resolve_switches,
    _resolve_workload_shape,
    _scale_op_results,
    _summarize_scaled_ops,
    _timeline_latency_ms_from_stage_cycles,
    _to_float,
)

DEFAULT_PROGRAM_CONTRACT = (
    ROOT_DIR / "configs" / "fuller_experiment_program_contract_20260422.yaml"
)

LANE_ORDER = ["ASTRA", "MESO", "HOPS", "DET", "SPARSE", "PHY", "FULLER"]
ENGINEERING_SMOKE_VARIANTS = set(LANE_ORDER)
_RUNTIME_ROW_FALLBACKS = {
    "det_prefix_error_mean",
    "det_prefix_error_p95",
    "det_k_global",
    "det_perturbation",
    "sparse_measured_activity_fraction",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _lane_lookup(contract_path: Path) -> dict[str, dict[str, Any]]:
    contract = _load_yaml(contract_path)
    return {
        str(row.get("variant_id") or "").strip().upper(): row
        for row in contract.get("lanes") or []
        if str(row.get("variant_id") or "").strip()
    }


def _variant_id_for_cfg(cfg: dict[str, Any], explicit_variant_id: str | None = None) -> str:
    if explicit_variant_id:
        return str(explicit_variant_id).strip().upper()
    run_cfg = cfg.get("run") or {}
    experiment_id = str(run_cfg.get("experiment_id") or "").strip().upper()
    if experiment_id:
        return str(
            default_variant_descriptor_for_experiment(experiment_id).get("variant_id") or ""
        ).strip().upper()
    return str(run_cfg.get("variant_id") or "").strip().upper()


def _variant_id_from_runtime_row(raw_row: dict[str, Any] | None) -> str:
    if not raw_row:
        return ""
    experiment_id = str(raw_row.get("experiment_id") or "").strip().upper()
    if experiment_id:
        return str(
            default_variant_descriptor_for_experiment(experiment_id).get("variant_id") or ""
        ).strip().upper()
    for key in ("run_id", "source_run_id"):
        value = str(raw_row.get(key) or "").strip().lower()
        if not value:
            continue
        for variant_id in LANE_ORDER:
            if variant_id.lower() in value:
                return variant_id
    return ""


def _model_key_from_cfg(cfg: dict[str, Any], raw_row: dict[str, Any] | None) -> str:
    if raw_row:
        model = str(raw_row.get("model") or "").strip()
        if model:
            return model
    models_cfg = cfg.get("models") or {}
    keys = models_cfg.get("keys")
    if isinstance(keys, list) and keys:
        return str(keys[0]).strip()
    return "mobilevit_s"


def _resolve_eval_row(
    rows: list[dict[str, str]],
    *,
    eval_run_id: str | None = None,
) -> dict[str, str] | None:
    normalized_run_id = str(eval_run_id or "").strip()
    if normalized_run_id:
        for row in rows:
            if str(row.get("run_id") or "").strip() != normalized_run_id:
                continue
            if str(row.get("baseline") or "").strip().lower() == "true":
                continue
            return row
    for row in reversed(rows):
        if str(row.get("baseline") or "").strip().lower() == "true":
            continue
        return row
    return None


def _latency_seconds_from_row(row: dict[str, Any]) -> float:
    latency_ms = _to_float(row.get("latency_ms_per_sample"), None)
    if latency_ms is not None and latency_ms > 0:
        return float(latency_ms) / 1e3
    elapsed_s = _to_float(row.get("measured_pass_elapsed_s"), None)
    processed = _to_float(row.get("measured_processed_samples"), None)
    if elapsed_s is not None and processed is not None and processed > 0:
        return float(elapsed_s) / float(processed)
    return 0.0


def _materialize_ops_surface(
    *,
    cfg: dict[str, Any],
    raw_row: dict[str, Any],
    model_key: str,
    switches: dict[str, bool],
    sparse_scale: float,
    det_runtime_enabled: bool,
    sc_det_cfg: dict[str, Any],
    meso_cfg: dict[str, Any],
    flow_cfg: dict[str, Any],
) -> dict[str, Any]:
    estimator_cfg = cfg.get("mtl") or cfg.get("hpat") or {}
    estimator_config_path = _resolve_existing_path(
        estimator_cfg.get("config_path") or "mtl_model/mtl_config_asic.yaml"
    )
    estimator_config = _apply_realism_profile_if_present(
        _load_yaml(estimator_config_path),
        cfg_path=estimator_config_path,
    )
    if switches.get("phy"):
        energy_cfg = estimator_config.get("energy") or {}
        energy_cfg["laser_power_mw"] = 0.0
        estimator_config["energy"] = energy_cfg

    execution_semantics_cfg = _resolve_execution_semantics_cfg(cfg)
    base_estimator_config = dict(estimator_config)
    if isinstance(estimator_config.get("bitstream"), dict):
        base_estimator_config["bitstream"] = dict(estimator_config.get("bitstream") or {})
    model_estimator_config = _merge_execution_semantics_into_estimator_config(
        base_estimator_config,
        execution_semantics_cfg,
    )

    ops_dir = _resolve_existing_path(estimator_cfg.get("ops_dir") or "mtl_model/ops")
    ops_path = ops_dir / f"ops_{model_key}.json"
    if not ops_path.exists():
        candidates = sorted(path for path in ops_dir.glob("*.json") if path.is_file())
        for candidate in candidates:
            candidate_model, _, _ = _load_ops(candidate)
            if str(candidate_model).strip().lower() == model_key.strip().lower():
                ops_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"No ops JSON found for model={model_key!r} under {ops_dir}"
            )

    model_name, ops, meta = _load_ops(ops_path)
    data_cfg = cfg.get("data") or {}
    workload_shape = _resolve_workload_shape(
        model=model_name,
        meta=meta,
        data_cfg=data_cfg,
    )
    workload_ops = _apply_workload_scale_to_ops(
        ops,
        batch_scale=float(workload_shape["batch_scale"]),
        sequence_scale=float(workload_shape["sequence_scale"]),
        sequence_modeled=bool(workload_shape["sequence_modeled"]),
    )
    op_results, _ = summarize_ops(workload_ops, model_estimator_config)

    bsl_scale, _ = _compute_bsl_scale(sc_det_cfg, det_runtime_enabled)
    meso_load_scale = resolve_meso_load_scale(
        meso_cfg=meso_cfg,
        meso_enabled=bool(switches.get("meso")),
    )
    scaled_ops = _scale_op_results(
        op_results,
        bsl_max=float(sc_det_cfg.get("bsl_max") or 1.0),
        bsl_scale=bsl_scale,
        det_enabled=det_runtime_enabled,
        k_by_layer=(sc_det_cfg.get("early_stop") or {}).get("k_by_layer"),
        sparse_scale=sparse_scale,
        meso_load_scale=meso_load_scale,
    )
    sample_rate_gsps = _to_float(
        (estimator_config.get("photonic") or {}).get("sample_rate_gsps"),
        1.0,
    ) or 1.0
    model_timeline_rows, model_buffer_rows = _build_per_layer_timeline_rows(
        model=model_name,
        scaled_ops=scaled_ops,
        sample_rate_gsps=sample_rate_gsps,
        flow_enabled=bool(switches.get("flow")),
        det_enabled=det_runtime_enabled,
        flow_cfg=flow_cfg,
    )
    stage_cycles, bubble_cycles, utilization_avg = _aggregate_timeline_for_model(
        model=model_name,
        per_layer_timeline_rows=model_timeline_rows,
    )
    flow_buffer_peak_cycles, flow_buffer_peak_frac = _aggregate_buffer_trace_for_model(
        model=model_name,
        per_layer_buffer_rows=model_buffer_rows,
    )
    flow_diagnostics = _aggregate_flow_diagnostics_for_model(
        model=model_name,
        per_layer_buffer_rows=model_buffer_rows,
    )
    flow_summary_fields = _resolve_flow_summary_fields(
        flow_cfg=flow_cfg,
        flow_enabled=bool(switches.get("flow")),
        flow_buffer_peak_cycles=flow_buffer_peak_cycles,
        flow_buffer_peak_frac=flow_buffer_peak_frac,
        diagnostics=flow_diagnostics,
    )
    scaled_summary = _summarize_scaled_ops(
        scaled_ops,
        meso_overhead_mj=(
            (_to_float(meso_cfg.get("broadcast_overhead_mj"), 0.0) or 0.0)
            if switches.get("meso")
            else 0.0
        )
        * float(workload_shape["workload_scale"]),
        meso_overhead_ms=(
            (_to_float(meso_cfg.get("broadcast_overhead_ms"), 0.0) or 0.0)
            if switches.get("meso")
            else 0.0
        )
        * float(workload_shape["workload_scale"]),
    )
    layout_cfg = estimator_config.get("layout") or {}
    energy_model_cfg = cfg.get("energy_model") or {}
    energy_model_mode = str(energy_model_cfg.get("energy_model_mode") or "UpperBound").strip().lower()
    if energy_model_mode not in {"upperbound", "countbased"}:
        energy_model_mode = "upperbound"
    upperbound_scale = _to_float(energy_model_cfg.get("upperbound_scale"), 1.0) or 1.0
    countbased_scale = _to_float(energy_model_cfg.get("countbased_scale"), 0.9) or 0.9
    upperbound_scale = max(upperbound_scale, countbased_scale)
    selected_scale = countbased_scale if energy_model_mode == "countbased" else upperbound_scale
    latency_ms_from_row = _to_float(raw_row.get("latency_ms_per_sample"), None)
    core_latency_ms = (
        float(latency_ms_from_row)
        if latency_ms_from_row is not None and latency_ms_from_row > 0
        else _timeline_latency_ms_from_stage_cycles(stage_cycles, sample_rate_gsps)
    )
    thermal_tuning_mw = _to_float(
        layout_cfg.get("p_thermal_tuning_mw", layout_cfg.get("P_thermal_tuning")),
        0.0,
    ) or 0.0
    thermal_energy_j = thermal_tuning_mw / 1000.0 * (core_latency_ms / 1e3)
    integrated_system_costs = _compute_integrated_system_costs(
        conversion_control_j=scaled_summary["energy_j_conversion_control"] * selected_scale,
        memory_move_j=scaled_summary["energy_j_memory_move"] * selected_scale,
        thermal_energy_j=thermal_energy_j,
        core_latency_ms=core_latency_ms,
        stage_cycles_payload=stage_cycles,
        sample_rate_gsps=sample_rate_gsps,
        flow_enabled=bool(switches.get("flow")),
        phy_enabled=bool(switches.get("phy")),
        noise_enabled=_to_float((cfg.get("noise_injection") or {}).get("sigma_lsb"), 0.0)
        not in {None, 0.0},
        cost_cfg=cfg.get("integrated_system_costs"),
    )
    return {
        "stage_cycles": stage_cycles,
        "bubble_cycles": bubble_cycles,
        "utilization_avg": utilization_avg,
        "flow_summary_fields": flow_summary_fields,
        "scaled_summary": scaled_summary,
        "integrated_system_costs": integrated_system_costs,
        "integrated_system_cost_mode": integrated_system_costs.get(
            "integrated_system_cost_mode"
        ),
    }


def build_lane_annotation_fields_from_row(
    *,
    cfg_path: Path,
    raw_row: dict[str, Any],
    raw_results_csv: Path | None = None,
    variant_id: str | None = None,
) -> dict[str, Any]:
    cfg = _load_yaml(cfg_path)
    resolved_variant_id = _variant_id_for_cfg(cfg, explicit_variant_id=variant_id)
    if resolved_variant_id not in ENGINEERING_SMOKE_VARIANTS:
        resolved_variant_id = _variant_id_from_runtime_row(raw_row)
    if resolved_variant_id not in ENGINEERING_SMOKE_VARIANTS:
        return {}

    run_cfg = cfg.get("run") or {}
    experiment_id = str(run_cfg.get("experiment_id") or "").strip().upper()
    switches = _resolve_switches(cfg, experiment_id)
    sc_det_cfg = cfg.get("sc_det") or {}
    sparse_cfg = cfg.get("sparse") or {}
    meso_cfg = cfg.get("meso") or {}
    flow_cfg = cfg.get("flow") or {}
    phy_cfg = cfg.get("phy") or {}
    accuracy_cfg = cfg.get("accuracy") or {}
    realism_cfg = cfg.get("realism") or {}
    p1_align_cfg = cfg.get("p1_align") or {}
    model_key = _model_key_from_cfg(cfg, raw_row)

    det_runtime_metadata = resolve_det_runtime_metadata(sc_det_cfg, switches)
    det_runtime_enabled = bool(det_runtime_metadata.get("det_runtime_enabled"))
    sparse_scale, sparse_scale_source = _compute_sparse_scale(
        sparse_cfg,
        bool(switches.get("sparse")),
        measured_activity_fraction=_to_float(
            raw_row.get("sparse_measured_activity_fraction"),
            None,
        ),
    )
    latency_s = _latency_seconds_from_row(raw_row)
    meso_metrics = compute_meso_cost_model(
        meso_cfg=meso_cfg,
        meso_enabled=bool(switches.get("meso")),
        latency_s=latency_s,
    )
    p1_align_resolved = _calibrate_p1_alignment(
        p1_align_cfg=p1_align_cfg,
        accuracy_rows=[raw_row],
        model_keys=[model_key],
        quant_bits_default=_to_float((cfg.get("noise_injection") or {}).get("quant_bits", sc_det_cfg.get("quant_bits")), None),
        delta_pp_budget=_to_float((accuracy_cfg.get("target") or {}).get("delta_pp_budget"), None),
    )

    phy_result = None
    if switches.get("phy"):
        phy_result = compute_link_budget(phy_cfg, duty_cycle=sparse_scale)

    ops_surface: dict[str, Any] = {}
    if resolved_variant_id in {"HOPS", "FULLER"} or bool(switches.get("flow")):
        ops_surface = _materialize_ops_surface(
            cfg=cfg,
            raw_row=raw_row,
            model_key=model_key,
            switches=switches,
            sparse_scale=sparse_scale,
            det_runtime_enabled=det_runtime_enabled,
            sc_det_cfg=sc_det_cfg,
            meso_cfg=meso_cfg,
            flow_cfg=flow_cfg,
        )

    integrated_system_cost_cfg = _resolve_integrated_system_cost_cfg(
        cfg.get("integrated_system_costs")
    )
    accuracy_provenance = {
        "accuracy_source_csv": str(raw_results_csv or ""),
        "accuracy_target_analysis_grade_ready": False,
        "accuracy_target_analysis_grade_blockers": json.dumps(
            ["runtime_smoke_only"],
            ensure_ascii=False,
        ),
    }
    accuracy_coupling_metadata = {
        "accuracy_coupling_evidence": (
            "measured" if _to_float(raw_row.get("top1"), None) is not None else "missing"
        ),
        "accuracy_coupling_metric": "top1",
        "accuracy_coupling_source": str(raw_results_csv or ""),
        "accuracy_coupling_reason": "runtime_smoke_quantized_row",
    }
    realism_assessment = _build_realism_assessment(
        integrated_system_cost_cfg=integrated_system_cost_cfg,
        integrated_system_cost_mode=str(
            ops_surface.get("integrated_system_cost_mode")
            or integrated_system_cost_cfg.get("mode")
            or "disabled"
        ),
        flow_cfg=flow_cfg,
        meso_cfg=meso_cfg,
        phy_cfg=phy_cfg,
        realism_cfg=realism_cfg,
        accuracy_source_csv=str(raw_results_csv or ""),
        accuracy_provenance=accuracy_provenance,
        switches=switches,
        accuracy_coupling_metadata=accuracy_coupling_metadata,
        accuracy_measurement_contract_metadata={},
    )

    fields: dict[str, Any] = {}
    if resolved_variant_id == "ASTRA":
        top1 = _to_float(raw_row.get("top1"), None)
        top1_delta = _to_float(raw_row.get("top1_delta"), None)
        fields.update(
            {
                "acc_top1": top1 if top1 is not None else "",
                "acc_drop_pp": (-top1_delta) if top1_delta is not None else "",
                "accuracy_measurement_contract_status": (
                    str(raw_row.get("bitstream_truth_class_authorization_status") or "").strip()
                    or "not_required"
                ),
                "realism_class": realism_assessment.get("realism_class", ""),
            }
        )
    if resolved_variant_id in {"MESO", "FULLER"}:
        fields.update(
            {
                "fanout": meso_metrics["fanout"],
                "topology_dimension": meso_metrics["topology_dimension"],
                "meso_cost_model_mode": meso_metrics["cost_model_mode"],
                "serializers_saved": meso_metrics["serializers_saved"],
                "explicit_total_cost_j": meso_metrics["explicit_total_cost_j"],
                "net_energy_gain_j": meso_metrics["net_energy_gain_j"],
                "meso_cost_evidence": realism_assessment["meso_cost_evidence"],
            }
        )
    if resolved_variant_id in {"HOPS", "FULLER"}:
        flow_summary_fields = ops_surface.get("flow_summary_fields") or {}
        fields.update(
            {
                "hops_scheduler_mode": (
                    flow_summary_fields.get("flow_model_mode")
                    if switches.get("flow")
                    else "disabled"
                ),
                "stage_cycles": ops_surface.get("stage_cycles", ""),
                "bubble_cycles": ops_surface.get("bubble_cycles", ""),
                "utilization_avg": ops_surface.get("utilization_avg", ""),
                "flow_timeline_evidence": realism_assessment["flow_timeline_evidence"],
                "flow_buffer_peak_cycles": flow_summary_fields.get("flow_buffer_peak_cycles", ""),
                "flow_buffer_peak_frac": flow_summary_fields.get("flow_buffer_peak_frac", ""),
                "flow_residency_hit_rate": flow_summary_fields.get("flow_residency_hit_rate", ""),
                "flow_control_backpressure": flow_summary_fields.get("flow_control_backpressure", ""),
                "flow_eviction_count": flow_summary_fields.get("flow_eviction_count", ""),
                "flow_admission_stalls": flow_summary_fields.get("flow_admission_stalls", ""),
            }
        )
    if resolved_variant_id in {"DET", "FULLER"}:
        det_quality_gate_status = det_runtime_metadata.get("det_quality_gate_status") or ""
        det_quality_gate_cfg = sc_det_cfg.get("quality_gate") or {}
        fields.update(
            {
                "det_policy": det_runtime_metadata.get("det_policy") or "",
                "det_k_signature": det_runtime_metadata.get("det_k_signature") or "",
                "det_quality_gate_status": det_quality_gate_status,
                "det_quality_gate_policy": (
                    det_runtime_metadata.get("det_quality_gate_policy")
                    or det_quality_gate_cfg.get("policy_label")
                    or ("quality_gate_not_configured" if det_quality_gate_status else "")
                ),
                "det_quality_gate_reason": det_runtime_metadata.get("det_quality_gate_reason") or "",
                "det_quality_gate_fallback_policy": det_runtime_metadata.get(
                    "det_quality_gate_fallback_policy"
                )
                or "",
            }
        )
    if resolved_variant_id in {"SPARSE", "FULLER"}:
        fields.update(
            {
                "duty_cycle_avg": sparse_scale if switches.get("sparse") else 1.0,
                "sparse_active_fraction": sparse_scale if switches.get("sparse") else 1.0,
                "sparse_scale_source": sparse_scale_source,
                "sparse_measured_activity_fraction": raw_row.get(
                    "sparse_measured_activity_fraction", ""
                ),
            }
        )
    if resolved_variant_id in {"PHY", "FULLER"}:
        fields.update(
            {
                "phy_link_budget_status": (
                    "ready" if switches.get("phy") and phy_result is not None else "disabled"
                ),
                "N_wdm": (phy_result or {}).get("wdm_channels_n", ""),
                "P_laser_mw": (phy_result or {}).get("p_laser_mw", ""),
                "PP_crosstalk_db": (phy_result or {}).get("pp_crosstalk_db", ""),
                "gaussian_noise_std_ref": p1_align_resolved["gaussian_noise_std_ref"],
                "crosstalk_alpha_ref": p1_align_resolved["crosstalk_alpha_ref"],
                "phy_support_evidence": realism_assessment["phy_support_evidence"],
            }
        )
    if resolved_variant_id == "FULLER":
        integrated_system_costs = ops_surface.get("integrated_system_costs") or {}
        fields.update(
            {
                "integrated_system_cost_mode": integrated_system_costs.get(
                    "integrated_system_cost_mode",
                    "",
                ),
                "integrated_system_cost_evidence": realism_assessment[
                    "integrated_system_cost_evidence"
                ],
                "accuracy_coupling_evidence": realism_assessment[
                    "accuracy_coupling_evidence"
                ],
                "proxy_promotion_ready": str(
                    bool(realism_assessment.get("proxy_promotion_ready"))
                ).lower(),
                "benchmark_claim_ready": str(
                    bool(realism_assessment.get("benchmark_claim_ready"))
                ).lower(),
            }
        )
    return fields


def build_lane_annotation_fields(
    *,
    cfg_path: Path,
    raw_results_csv: Path,
    eval_run_id: str,
    variant_id: str | None = None,
) -> dict[str, Any]:
    rows = _load_csv_rows(raw_results_csv)
    raw_row = _resolve_eval_row(rows, eval_run_id=eval_run_id)
    if raw_row is None:
        return {}
    return build_lane_annotation_fields_from_row(
        cfg_path=cfg_path,
        raw_row=raw_row,
        raw_results_csv=raw_results_csv,
        variant_id=variant_id,
    )


def supported_result_surface_fields(
    *,
    cfg_path: Path,
    variant_id: str | None = None,
    contract_path: Path = DEFAULT_PROGRAM_CONTRACT,
) -> set[str]:
    cfg = _load_yaml(cfg_path)
    resolved_variant_id = _variant_id_for_cfg(cfg, explicit_variant_id=variant_id)
    if resolved_variant_id not in ENGINEERING_SMOKE_VARIANTS:
        return set()

    lanes = _lane_lookup(contract_path)
    lane = lanes.get(resolved_variant_id, {})
    supported = {
        str(field).strip()
        for field in lane.get("module_specific_fields") or []
        if str(field).strip()
    }
    if resolved_variant_id == "DET":
        supported.update(
            {
                "det_k_global",
                "det_prefix_error_mean",
                "det_prefix_error_p95",
                "det_perturbation",
            }
        )
    supported.update(_RUNTIME_ROW_FALLBACKS)
    return supported


def required_result_surface_fields(
    *,
    variant_id: str,
    contract_path: Path = DEFAULT_PROGRAM_CONTRACT,
) -> list[str]:
    lanes = _lane_lookup(contract_path)
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
