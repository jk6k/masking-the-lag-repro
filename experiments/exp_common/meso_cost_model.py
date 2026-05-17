"""Shared MESO explicit-cost model helpers."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
REUSE_METRIC_FILTER = "filter_reuse_fanout_under_operand_reuse"
REUSE_METRIC_PATCH = "patch_reuse_fanout_under_operand_reuse"
DEFAULT_REUSE_METRIC = REUSE_METRIC_FILTER
VALID_REUSE_METRICS = {REUSE_METRIC_FILTER, REUSE_METRIC_PATCH}
OVERHEAD_POLICY_EXPLICIT = "explicit"
OVERHEAD_POLICY_BROADCAST_DRIVER_FRACTION = "broadcast_driver_fraction"


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int) -> int:
    resolved = _to_float(value, None)
    if resolved is None:
        return default
    return int(round(resolved))


def _to_str_list(value: Any) -> list[str]:
    if value in ("", None):
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _resolve_repo_path(value: Any) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def _read_reuse_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _percentile(samples: list[float], q: float) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0]
    idx = max(0.0, min(1.0, q)) * float(len(samples) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return samples[lo]
    weight = idx - float(lo)
    return samples[lo] * (1.0 - weight) + samples[hi] * weight


def _resolve_reuse_metric(meso_cfg: dict[str, Any]) -> str:
    metric = str(meso_cfg.get("reuse_provenance_metric") or "").strip()
    if metric in VALID_REUSE_METRICS:
        return metric
    return DEFAULT_REUSE_METRIC


def _collect_reuse_fanout_samples(meso_cfg: dict[str, Any]) -> list[float]:
    path = _resolve_repo_path(meso_cfg.get("reuse_provenance_csv"))
    if path is None or not path.exists():
        return []
    class_allowlist = set(_to_str_list(meso_cfg.get("reuse_provenance_class_allowlist")))
    risk_allowlist = set(_to_str_list(meso_cfg.get("reuse_provenance_risk_allowlist")))
    metric = _resolve_reuse_metric(meso_cfg)
    samples: list[float] = []
    for row in _read_reuse_rows(path):
        if class_allowlist and str(row.get("memory_layout_provenance_class") or "") not in class_allowlist:
            continue
        if risk_allowlist and str(row.get("risk_class") or "") not in risk_allowlist:
            continue
        sample = _to_float(row.get(metric), None)
        if sample is None or sample <= 0.0:
            continue
        samples.append(sample)
    return sorted(samples)


def summarize_meso_reuse_provenance(meso_cfg: dict[str, Any]) -> dict[str, float | int | str]:
    samples = _collect_reuse_fanout_samples(meso_cfg)
    explicit_fanout = max(0, _to_int(meso_cfg.get("fanout"), 0))
    if not samples:
        return {
            "metric": _resolve_reuse_metric(meso_cfg),
            "aggregate": str(meso_cfg.get("reuse_provenance_aggregate") or "p50").strip().lower(),
            "sample_count": 0,
            "sample_min": 0.0,
            "sample_p50": 0.0,
            "sample_p75": 0.0,
            "sample_p90": 0.0,
            "sample_mean": 0.0,
            "sample_max": 0.0,
            "resolved_fanout": explicit_fanout,
        }
    return {
        "metric": _resolve_reuse_metric(meso_cfg),
        "aggregate": str(meso_cfg.get("reuse_provenance_aggregate") or "p50").strip().lower(),
        "sample_count": len(samples),
        "sample_min": samples[0],
        "sample_p50": _percentile(samples, 0.50),
        "sample_p75": _percentile(samples, 0.75),
        "sample_p90": _percentile(samples, 0.90),
        "sample_mean": sum(samples) / float(len(samples)),
        "sample_max": samples[-1],
        "resolved_fanout": resolve_meso_fanout(meso_cfg),
    }


def resolve_meso_fanout(meso_cfg: dict[str, Any]) -> int:
    explicit_fanout = max(0, _to_int(meso_cfg.get("fanout"), 0))
    policy = str(meso_cfg.get("fanout_policy") or "explicit").strip().lower()
    if policy != "reuse_provenance":
        return explicit_fanout

    samples = _collect_reuse_fanout_samples(meso_cfg)
    if not samples:
        return explicit_fanout

    aggregate = str(meso_cfg.get("reuse_provenance_aggregate") or "p50").strip().lower()
    if aggregate in {"p75", "q75"}:
        resolved = _percentile(samples, 0.75)
    elif aggregate in {"p90", "q90"}:
        resolved = _percentile(samples, 0.90)
    elif aggregate in {"mean", "avg"}:
        resolved = sum(samples) / float(len(samples))
    elif aggregate == "max":
        resolved = samples[-1]
    else:
        resolved = _percentile(samples, 0.50)

    fanout = max(1, int(round(resolved)))
    fanout_min = max(1, _to_int(meso_cfg.get("fanout_clip_min"), 2))
    fanout_max = _to_int(meso_cfg.get("fanout_clip_max"), fanout)
    if fanout_max > 0:
        fanout = min(fanout, fanout_max)
    fanout = max(fanout_min, fanout)
    return fanout


def resolve_meso_topology_dimension(*, meso_cfg: dict[str, Any], fanout: int) -> float:
    explicit = _to_float(meso_cfg.get("topology_dimension"), 1.0) or 1.0
    policy = str(meso_cfg.get("topology_dimension_policy") or "explicit").strip().lower()
    if policy != "fanout_hierarchy":
        return max(1.0, explicit)

    branch_factor = _to_float(meso_cfg.get("topology_branch_factor"), 8.0) or 8.0
    branch_factor = max(2.0, branch_factor)
    if fanout <= 1:
        return 1.0
    levels = math.ceil(math.log(float(max(2, fanout)), branch_factor))
    return max(1.0, float(levels))


def resolve_meso_cost_model_mode(meso_cfg: dict[str, Any]) -> str:
    mode = str(meso_cfg.get("cost_model_mode") or "").strip().lower()
    if mode in {"explicit_topology_v1", "explicit_cost_v1"}:
        return "explicit_topology_v1"
    return "legacy_load_scale_proxy"


def use_explicit_meso_cost_model(
    *,
    meso_cfg: dict[str, Any],
    meso_enabled: bool,
) -> bool:
    return meso_enabled and resolve_meso_cost_model_mode(meso_cfg) == "explicit_topology_v1"


def resolve_meso_load_scale(
    *,
    meso_cfg: dict[str, Any],
    meso_enabled: bool,
) -> float:
    if not meso_enabled:
        return 1.0
    if use_explicit_meso_cost_model(meso_cfg=meso_cfg, meso_enabled=meso_enabled):
        return 1.0
    return _to_float(meso_cfg.get("load_scale"), 1.0) or 1.0


def _resolve_energy_j(
    *,
    cfg: dict[str, Any],
    energy_key_j: str,
    energy_key_mj: str,
    power_key_mw: str,
    latency_s: float,
    default_j: float = 0.0,
) -> float:
    energy_j = _to_float(cfg.get(energy_key_j), None)
    if energy_j is not None:
        return max(0.0, energy_j)
    energy_mj = _to_float(cfg.get(energy_key_mj), None)
    if energy_mj is not None:
        return max(0.0, energy_mj / 1e3)
    power_mw = _to_float(cfg.get(power_key_mw), None)
    if power_mw is not None and latency_s > 0:
        return max(0.0, power_mw / 1000.0 * latency_s)
    return max(0.0, default_j)


def _resolve_overhead_energy_j(
    *,
    cfg: dict[str, Any],
    policy_key: str,
    scale_key: str,
    energy_key_j: str,
    energy_key_mj: str,
    power_key_mw: str,
    latency_s: float,
    broadcast_driver_energy_j: float,
) -> float:
    policy = str(cfg.get(policy_key) or OVERHEAD_POLICY_EXPLICIT).strip().lower()
    if policy == OVERHEAD_POLICY_BROADCAST_DRIVER_FRACTION:
        scale = _to_float(cfg.get(scale_key), 0.0) or 0.0
        return max(0.0, broadcast_driver_energy_j * scale)
    return _resolve_energy_j(
        cfg=cfg,
        energy_key_j=energy_key_j,
        energy_key_mj=energy_key_mj,
        power_key_mw=power_key_mw,
        latency_s=latency_s,
    )


def _resolve_overhead_energy_band_j(
    *,
    cfg: dict[str, Any],
    policy_key: str,
    scale_key: str,
    scale_lower_key: str,
    scale_upper_key: str,
    energy_key_j: str,
    energy_key_mj: str,
    power_key_mw: str,
    latency_s: float,
    broadcast_driver_energy_j: float,
) -> tuple[float, float, float]:
    nominal = _resolve_overhead_energy_j(
        cfg=cfg,
        policy_key=policy_key,
        scale_key=scale_key,
        energy_key_j=energy_key_j,
        energy_key_mj=energy_key_mj,
        power_key_mw=power_key_mw,
        latency_s=latency_s,
        broadcast_driver_energy_j=broadcast_driver_energy_j,
    )
    policy = str(cfg.get(policy_key) or OVERHEAD_POLICY_EXPLICIT).strip().lower()
    if policy != OVERHEAD_POLICY_BROADCAST_DRIVER_FRACTION:
        return nominal, nominal, nominal

    lower = nominal
    upper = nominal
    scale_lower = _to_float(cfg.get(scale_lower_key), None)
    if scale_lower is not None:
        lower = max(0.0, broadcast_driver_energy_j * scale_lower)
    scale_upper = _to_float(cfg.get(scale_upper_key), None)
    if scale_upper is not None:
        upper = max(0.0, broadcast_driver_energy_j * scale_upper)
    lower = min(lower, nominal)
    upper = max(upper, nominal)
    if lower > upper:
        lower, upper = upper, lower
    return nominal, lower, upper


def compute_meso_cost_model(
    *,
    meso_cfg: dict[str, Any],
    meso_enabled: bool,
    latency_s: float,
    fanout_override: int | None = None,
    topology_dimension_override: float | None = None,
    fabric_control_overhead_j_override: float | None = None,
    extra_buffering_overhead_j_override: float | None = None,
) -> dict[str, float | int | str | bool]:
    mode = resolve_meso_cost_model_mode(meso_cfg)
    if not meso_enabled:
        return {
            "cost_model_mode": "disabled",
            "evidence_type": "disabled",
            "calibration_source": "",
            "load_scale_applied": False,
            "fanout": 0,
            "topology_dimension": 0.0,
            "serializers_saved": 0.0,
            "serializer_energy_j": 0.0,
            "broadcast_driver_energy_j": 0.0,
            "fabric_control_overhead_j": 0.0,
            "fabric_control_overhead_lower_j": 0.0,
            "fabric_control_overhead_upper_j": 0.0,
            "extra_buffering_overhead_j": 0.0,
            "extra_buffering_overhead_lower_j": 0.0,
            "extra_buffering_overhead_upper_j": 0.0,
            "explicit_total_cost_j": 0.0,
            "explicit_total_cost_lower_j": 0.0,
            "explicit_total_cost_upper_j": 0.0,
            "explicit_total_savings_j": 0.0,
            "net_energy_gain_j": 0.0,
            "net_energy_gain_lower_j": 0.0,
            "net_energy_gain_upper_j": 0.0,
            "break_even": False,
            "break_even_lower_bound": False,
            "break_even_upper_bound": False,
        }

    fanout = fanout_override if fanout_override is not None else resolve_meso_fanout(meso_cfg)
    serializers_saved = _to_float(meso_cfg.get("serializers_saved"), None)
    if serializers_saved is None:
        serializers_per_tile = _to_float(meso_cfg.get("serializers_per_tile"), 1.0) or 1.0
        serializers_saved = max(0.0, fanout - 1) * serializers_per_tile
    topology_dimension = (
        topology_dimension_override
        if topology_dimension_override is not None
        else resolve_meso_topology_dimension(meso_cfg=meso_cfg, fanout=fanout)
    )
    topology_dimension = max(1.0, topology_dimension)

    serializer_energy_j = _resolve_energy_j(
        cfg=meso_cfg,
        energy_key_j="serializer_energy_j",
        energy_key_mj="serializer_energy_mj",
        power_key_mw="serializer_power_mw",
        latency_s=latency_s,
    )
    broadcast_driver_energy_j = _resolve_energy_j(
        cfg=meso_cfg,
        energy_key_j="broadcast_driver_energy_j",
        energy_key_mj="broadcast_driver_energy_mj",
        power_key_mw="broadcast_driver_power_mw",
        latency_s=latency_s,
        default_j=(_to_float(meso_cfg.get("broadcast_overhead_mj"), 0.0) or 0.0) / 1e3,
    )
    if fabric_control_overhead_j_override is not None:
        fabric_control_overhead_j = max(0.0, fabric_control_overhead_j_override)
        fabric_control_overhead_lower_j = fabric_control_overhead_j
        fabric_control_overhead_upper_j = fabric_control_overhead_j
    else:
        (
            fabric_control_overhead_j,
            fabric_control_overhead_lower_j,
            fabric_control_overhead_upper_j,
        ) = _resolve_overhead_energy_band_j(
            cfg=meso_cfg,
            policy_key="fabric_control_overhead_policy",
            scale_key="fabric_control_scale_vs_broadcast_driver",
            scale_lower_key="fabric_control_scale_vs_broadcast_driver_lower",
            scale_upper_key="fabric_control_scale_vs_broadcast_driver_upper",
            energy_key_j="fabric_control_overhead_j",
            energy_key_mj="fabric_control_overhead_mj",
            power_key_mw="fabric_control_power_mw",
            latency_s=latency_s,
            broadcast_driver_energy_j=broadcast_driver_energy_j,
        )
    if extra_buffering_overhead_j_override is not None:
        extra_buffering_overhead_j = max(0.0, extra_buffering_overhead_j_override)
        extra_buffering_overhead_lower_j = extra_buffering_overhead_j
        extra_buffering_overhead_upper_j = extra_buffering_overhead_j
    else:
        (
            extra_buffering_overhead_j,
            extra_buffering_overhead_lower_j,
            extra_buffering_overhead_upper_j,
        ) = _resolve_overhead_energy_band_j(
            cfg=meso_cfg,
            policy_key="extra_buffering_overhead_policy",
            scale_key="extra_buffering_scale_vs_broadcast_driver",
            scale_lower_key="extra_buffering_scale_vs_broadcast_driver_lower",
            scale_upper_key="extra_buffering_scale_vs_broadcast_driver_upper",
            energy_key_j="extra_buffering_overhead_j",
            energy_key_mj="extra_buffering_overhead_mj",
            power_key_mw="extra_buffering_power_mw",
            latency_s=latency_s,
            broadcast_driver_energy_j=broadcast_driver_energy_j,
        )

    explicit_total_savings_j = max(0.0, serializers_saved) * serializer_energy_j
    explicit_total_cost_j = (
        broadcast_driver_energy_j
        + fabric_control_overhead_j * topology_dimension
        + extra_buffering_overhead_j * max(0.0, fanout - 1)
    )
    explicit_total_cost_lower_j = (
        broadcast_driver_energy_j
        + fabric_control_overhead_lower_j * topology_dimension
        + extra_buffering_overhead_lower_j * max(0.0, fanout - 1)
    )
    explicit_total_cost_upper_j = (
        broadcast_driver_energy_j
        + fabric_control_overhead_upper_j * topology_dimension
        + extra_buffering_overhead_upper_j * max(0.0, fanout - 1)
    )
    net_energy_gain_j = explicit_total_savings_j - explicit_total_cost_j
    net_energy_gain_lower_j = explicit_total_savings_j - explicit_total_cost_upper_j
    net_energy_gain_upper_j = explicit_total_savings_j - explicit_total_cost_lower_j

    return {
        "cost_model_mode": mode,
        "evidence_type": str(meso_cfg.get("evidence_type") or ""),
        "calibration_source": str(meso_cfg.get("calibration_source") or ""),
        "load_scale_applied": mode != "explicit_topology_v1",
        "fanout": fanout,
        "topology_dimension": topology_dimension,
        "serializers_saved": max(0.0, serializers_saved),
        "serializer_energy_j": serializer_energy_j,
        "broadcast_driver_energy_j": broadcast_driver_energy_j,
        "fabric_control_overhead_j": fabric_control_overhead_j,
        "fabric_control_overhead_lower_j": fabric_control_overhead_lower_j,
        "fabric_control_overhead_upper_j": fabric_control_overhead_upper_j,
        "extra_buffering_overhead_j": extra_buffering_overhead_j,
        "extra_buffering_overhead_lower_j": extra_buffering_overhead_lower_j,
        "extra_buffering_overhead_upper_j": extra_buffering_overhead_upper_j,
        "explicit_total_cost_j": explicit_total_cost_j,
        "explicit_total_cost_lower_j": explicit_total_cost_lower_j,
        "explicit_total_cost_upper_j": explicit_total_cost_upper_j,
        "explicit_total_savings_j": explicit_total_savings_j,
        "net_energy_gain_j": net_energy_gain_j,
        "net_energy_gain_lower_j": net_energy_gain_lower_j,
        "net_energy_gain_upper_j": net_energy_gain_upper_j,
        "break_even": net_energy_gain_j > 0.0,
        "break_even_lower_bound": net_energy_gain_lower_j > 0.0,
        "break_even_upper_bound": net_energy_gain_upper_j > 0.0,
    }
