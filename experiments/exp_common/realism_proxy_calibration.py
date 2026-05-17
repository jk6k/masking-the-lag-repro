"""Helpers to derive and apply calibration profiles for realistic proxy runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _model_row(path: Path, *, model: str) -> dict[str, str]:
    rows = [row for row in _read_csv(path) if str(row.get("model") or "").strip() == model]
    if not rows:
        raise ValueError(f"Missing model={model} row in {path}")
    return rows[0]


def _stage_payload(row: dict[str, str]) -> dict[str, int]:
    payload = str(row.get("stage_cycles") or "").strip()
    if not payload:
        raise ValueError(f"Missing stage_cycles payload in row={row}")
    parsed = json.loads(payload)
    return {str(key): int(value) for key, value in parsed.items()}


def derive_flow_calibration(
    *,
    baseline_master_csv: Path,
    flow_master_csv: Path,
    model: str,
    buffer_depth: int,
) -> dict[str, Any]:
    baseline = _model_row(baseline_master_csv, model=model)
    flow = _model_row(flow_master_csv, model=model)
    baseline_stage = _stage_payload(baseline)
    flow_stage = _stage_payload(flow)
    baseline_total = max(1, sum(baseline_stage.values()))
    flow_total = max(1, sum(flow_stage.values()))
    baseline_bubble = baseline_stage.get("bubble", 0)
    flow_bubble = flow_stage.get("bubble", 0)
    baseline_ratio = baseline_bubble / baseline_total
    flow_ratio = flow_bubble / flow_total
    buffer_factor = buffer_depth / float(buffer_depth + 1) if buffer_depth > 0 else 0.0
    overlap_efficiency = 0.0
    if baseline_ratio > 0.0 and buffer_factor > 0.0:
        overlap_efficiency = _clamp(
            (1.0 - (flow_ratio / baseline_ratio)) / (0.95 * buffer_factor),
            0.0,
            1.0,
        )
    return {
        "enabled": True,
        "evidence_type": "retained_model_calibrated",
        "calibration_source": f"{baseline_master_csv};{flow_master_csv}",
        "buffer_depth": int(buffer_depth),
        "overlap_efficiency": overlap_efficiency,
        "staging_cost_scale": 1.0,
        "sync_penalty_scale": 1.0,
        "calibration_anchor_model": model,
        "baseline_bubble_ratio": baseline_ratio,
        "flow_bubble_ratio": flow_ratio,
    }


def derive_meso_calibration(
    *,
    fanout_sweep_csv: Path,
) -> dict[str, Any]:
    rows = _read_csv(fanout_sweep_csv)
    if not rows:
        raise ValueError(f"Empty MESO fanout sweep: {fanout_sweep_csv}")
    serializer_samples: list[float] = []
    broadcast_samples: list[float] = []
    for row in rows:
        serializers_saved = _to_float(row.get("serializers_saved"), 0.0) or 0.0
        net_gain = _to_float(row.get("net_energy_gain_j"), 0.0) or 0.0
        broadcast_energy = _to_float(row.get("broadcast_driver_energy_j"), 0.0) or 0.0
        latency_ms = _to_float(row.get("latency_ms"), None)
        if serializers_saved > 0:
            serializer_samples.append((net_gain + broadcast_energy) / serializers_saved)
        if latency_ms is not None and latency_ms > 0:
            broadcast_samples.append((broadcast_energy / (latency_ms / 1e3)) * 1000.0)
    serializer_energy_j = sum(serializer_samples) / float(len(serializer_samples))
    broadcast_driver_power_mw = sum(broadcast_samples) / float(len(broadcast_samples))
    return {
        "enabled": True,
        "cost_model_mode": "explicit_topology_v1",
        "evidence_type": "retained_model_calibrated",
        "calibration_source": str(fanout_sweep_csv),
        "serializer_energy_j": serializer_energy_j,
        "broadcast_driver_power_mw": broadcast_driver_power_mw,
        "fabric_control_overhead_j": 0.0,
        "extra_buffering_overhead_j": 0.0,
    }


def _solve_xtalk_db(
    *,
    n_wdm: int,
    pp_crosstalk_db: float,
    er_db: float,
) -> float:
    if n_wdm <= 1:
        return -120.0
    er_linear = 10 ** (er_db / 10.0)
    penalty_linear = 10 ** (pp_crosstalk_db / 10.0) - 1.0
    denom = (n_wdm - 1) * (1.0 + (1.0 / er_linear))
    if penalty_linear <= 0 or denom <= 0:
        return -120.0
    xtalk_linear = penalty_linear / denom
    return 10.0 * math.log10(xtalk_linear)


def derive_phy_calibration(
    *,
    phy_sweep_csv: Path,
    er_db: float,
    p_sensitivity_dbm: float,
    pp_extinction_db: float,
    margin_db: float,
) -> dict[str, Any]:
    rows = _read_csv(phy_sweep_csv)
    if not rows:
        raise ValueError(f"Empty PHY sweep: {phy_sweep_csv}")
    xtalk_values: list[float] = []
    base_budget_values: list[float] = []
    for row in rows:
        n_wdm = int(_to_float(row.get("N_wdm"), 1) or 1)
        pp_crosstalk_db = _to_float(row.get("PP_crosstalk_db"), 0.0) or 0.0
        p_laser_dbm = _to_float(row.get("P_laser_dbm"), 0.0) or 0.0
        loss_path_db = _to_float(row.get("Loss_path_db"), 0.0) or 0.0
        xtalk_values.append(
            _solve_xtalk_db(
                n_wdm=n_wdm,
                pp_crosstalk_db=pp_crosstalk_db,
                er_db=er_db,
            )
        )
        base_budget_values.append(p_laser_dbm - pp_crosstalk_db - loss_path_db)
    xtalk_db = sum(xtalk_values) / float(len(xtalk_values))
    resolved_base_budget = sum(base_budget_values) / float(len(base_budget_values))
    expected_base_budget = p_sensitivity_dbm + pp_extinction_db + margin_db
    if abs(resolved_base_budget - expected_base_budget) > 1e-6:
        raise ValueError(
            "PHY sweep base budget does not match template constants: "
            f"resolved={resolved_base_budget}, expected={expected_base_budget}"
        )
    return {
        "enabled": True,
        "evidence_type": "retained_model_calibrated",
        "calibration_source": str(phy_sweep_csv),
        "crosstalk": {
            "model": "parametric",
            "xtalk_db": xtalk_db,
            "pp_crosstalk_db": None,
            "phy_penalty_table_version": "parametric-v1",
        },
    }


def derive_integrated_system_cost_calibration(
    *,
    astra_summary_csv: Path,
    fuller_summary_csv: Path,
) -> dict[str, Any]:
    astra = _read_csv(astra_summary_csv)[0]
    fuller = _read_csv(fuller_summary_csv)[0]
    memory_move = _to_float(fuller.get("energy_breakdown_memory_move_j"), 0.0) or 0.0
    conversion = _to_float(fuller.get("energy_breakdown_conversion_control_j"), 0.0) or 0.0
    thermal_energy = ((_to_float(fuller.get("P_thermal_tuning"), 0.0) or 0.0) / 1000.0) * (
        (_to_float(fuller.get("core_latency_ms"), 0.0) or 0.0) / 1e3
    )
    astra_memory_move = _to_float(astra.get("energy_breakdown_memory_move_j"), 0.0) or 0.0
    astra_conversion = _to_float(astra.get("energy_breakdown_conversion_control_j"), 0.0) or 0.0
    astra_thermal_energy = ((_to_float(astra.get("P_thermal_tuning"), 0.0) or 0.0) / 1000.0) * (
        (_to_float(astra.get("core_latency_ms"), 0.0) or 0.0) / 1e3
    )
    onchip_scale = (
        (_to_float(fuller.get("integrated_onchip_comm_j"), 0.0) or 0.0) / memory_move
        if memory_move > 0
        else 0.0
    )
    control_scale = (
        (_to_float(fuller.get("integrated_control_sched_j"), 0.0) or 0.0) / conversion
        if conversion > 0
        else 0.0
    )
    host_scale = (
        (_to_float(fuller.get("integrated_host_staging_j"), 0.0) or 0.0) / conversion
        if conversion > 0
        else 0.0
    )
    calibration_scale = (
        (_to_float(fuller.get("integrated_calibration_monitoring_j"), 0.0) or 0.0) / thermal_energy
        if thermal_energy > 0
        else 0.0
    )
    astra_onchip_scale = (
        (_to_float(astra.get("integrated_onchip_comm_j"), 0.0) or 0.0) / astra_memory_move
        if astra_memory_move > 0
        else onchip_scale
    )
    astra_control_scale = (
        (_to_float(astra.get("integrated_control_sched_j"), 0.0) or 0.0) / astra_conversion
        if astra_conversion > 0
        else control_scale
    )
    astra_calibration_scale = (
        (_to_float(astra.get("integrated_calibration_monitoring_j"), 0.0) or 0.0) / astra_thermal_energy
        if astra_thermal_energy > 0
        else 0.0
    )
    if abs(onchip_scale - astra_onchip_scale) > 1e-9:
        raise ValueError("ASTRA/FULLER on-chip scales disagree; hidden-cost anchor is inconsistent.")
    if abs(control_scale - astra_control_scale) > 1e-9:
        raise ValueError("ASTRA/FULLER control scales disagree; hidden-cost anchor is inconsistent.")
    return {
        "enabled": True,
        "mode": "integrated_minimal_v1",
        "calibration_source": f"{astra_summary_csv};{fuller_summary_csv}",
        "uncertainty_method": "bounded_envelope",
        "onchip_scale_vs_move": onchip_scale,
        "control_sched_scale_vs_conv": control_scale,
        "host_staging_scale_vs_conv": host_scale,
        "calibration_monitoring_scale_vs_thermal": calibration_scale,
        "onchip_latency_scale_vs_io": 0.10,
        "control_sched_latency_scale_vs_control": 0.10,
        "host_staging_latency_scale_vs_io": 0.05,
        "calibration_monitoring_latency_scale_vs_core": 0.02,
        "uncertainty_lower_scale": 0.75,
        "uncertainty_upper_scale": 1.25,
        "host_staging_flow_only": True,
        "calibration_monitoring_phy_or_noise_only": True,
        "onchip_comm_evidence_type": "bounded_envelope_calibrated",
        "control_sched_evidence_type": "bounded_envelope_calibrated",
        "host_staging_evidence_type": "bounded_envelope_calibrated",
        "calibration_monitoring_evidence_type": (
            "bounded_envelope_calibrated"
            if calibration_scale > 0 or astra_calibration_scale == 0.0
            else "heuristic_proxy"
        ),
    }


def build_realistic_proxy_profile(
    *,
    model: str,
    buffer_depth: int,
    baseline_flow_run_csv: Path,
    flow_run_csv: Path,
    meso_fanout_sweep_csv: Path,
    phy_n_sweep_csv: Path,
    astra_summary_csv: Path,
    fuller_summary_csv: Path,
    phase6_scope_report_md: Path,
) -> dict[str, Any]:
    flow = derive_flow_calibration(
        baseline_master_csv=baseline_flow_run_csv,
        flow_master_csv=flow_run_csv,
        model=model,
        buffer_depth=buffer_depth,
    )
    meso = derive_meso_calibration(
        fanout_sweep_csv=meso_fanout_sweep_csv,
    )
    phy = derive_phy_calibration(
        phy_sweep_csv=phy_n_sweep_csv,
        er_db=6.0,
        p_sensitivity_dbm=-20.0,
        pp_extinction_db=2.0,
        margin_db=4.0,
    )
    phy["calibration_source"] = f"{phy_n_sweep_csv};{phase6_scope_report_md}"
    integrated = derive_integrated_system_cost_calibration(
        astra_summary_csv=astra_summary_csv,
        fuller_summary_csv=fuller_summary_csv,
    )
    integrated["calibration_source"] = (
        f"{astra_summary_csv};{fuller_summary_csv};{phase6_scope_report_md}"
    )
    return {
        "meta": {
            "profile_id": "20260329_fuller_realistic_proxy_v1",
            "goal": "promote fuller slice from bounded_system_model to realistic_accelerator_proxy using retained-run calibration anchors",
        },
        "realism": {
            "current_class": "realistic_accelerator_proxy",
            "target_class": "realistic_accelerator_proxy",
            "device_comparison_scope": "contextual_only",
            "benchmark_equivalence": False,
            "benchmark_claim_ready": False,
            "comparison_claim_scope": "contextual_only",
            "calibration_source": (
                f"{baseline_flow_run_csv};{flow_run_csv};{meso_fanout_sweep_csv};"
                f"{phy_n_sweep_csv};{astra_summary_csv};{fuller_summary_csv}"
            ),
        },
        "flow": flow,
        "meso": meso,
        "phy": phy,
        "integrated_system_costs": integrated,
    }


def apply_realism_proxy_profile(cfg: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    merged = dict(cfg)
    realism = dict(merged.get("realism") or {})
    realism.update(profile.get("realism") or {})
    merged["realism"] = realism

    for section in ("flow", "meso", "phy"):
        section_cfg = dict(merged.get(section) or {})
        section_cfg.update(profile.get(section) or {})
        merged[section] = section_cfg

    integrated_cfg = dict(merged.get("integrated_system_costs") or {})
    integrated_cfg.update(profile.get("integrated_system_costs") or {})
    merged["integrated_system_costs"] = integrated_cfg
    return merged


def write_realistic_proxy_profile(
    *,
    profile: dict[str, Any],
    out_yaml: Path,
    out_report_md: Path,
) -> None:
    _write_yaml(out_yaml, profile)
    lines = [
        "# Fuller Realistic Proxy Calibration Report",
        "",
        f"Profile: `{profile['meta']['profile_id']}`",
        "",
        "Result",
        "- realism class target: `realistic_accelerator_proxy`",
        "- device comparison scope: `contextual_only`",
        "- benchmark-equivalent claim ready: `false`",
        "",
        "Calibration anchors",
        f"- HOPS: `{profile['flow']['calibration_source']}`",
        f"- MESO: `{profile['meso']['calibration_source']}`",
        f"- PHY: `{profile['phy']['calibration_source']}`",
        f"- hidden system costs: `{profile['integrated_system_costs']['calibration_source']}`",
        "",
        "Derived parameters",
        f"- `flow.overlap_efficiency`: `{profile['flow']['overlap_efficiency']:.6f}`",
        f"- `meso.serializer_energy_j`: `{profile['meso']['serializer_energy_j']:.12g}`",
        f"- `meso.broadcast_driver_power_mw`: `{profile['meso']['broadcast_driver_power_mw']:.12g}`",
        f"- `phy.crosstalk.xtalk_db`: `{profile['phy']['crosstalk']['xtalk_db']:.6f}`",
        f"- `integrated_system_costs.onchip_scale_vs_move`: `{profile['integrated_system_costs']['onchip_scale_vs_move']:.6f}`",
        f"- `integrated_system_costs.control_sched_scale_vs_conv`: `{profile['integrated_system_costs']['control_sched_scale_vs_conv']:.6f}`",
        f"- `integrated_system_costs.host_staging_scale_vs_conv`: `{profile['integrated_system_costs']['host_staging_scale_vs_conv']:.6f}`",
        f"- `integrated_system_costs.calibration_monitoring_scale_vs_thermal`: `{profile['integrated_system_costs']['calibration_monitoring_scale_vs_thermal']:.6f}`",
        "",
        "Boundary",
        "- This profile upgrades the fuller lane to a calibration-backed accelerator proxy.",
        "- It does not upgrade local CPU/MPS comparison into benchmark-equivalent evidence.",
        "- PHY remains a support-only bounded realism envelope, not hardware truth.",
    ]
    _write_text(out_report_md, "\n".join(lines) + "\n")

