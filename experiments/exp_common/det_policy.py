"""Shared helpers for DET policy metadata and per-layer runtime payloads."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from exp_common.det_adaptive_k_allocator import allocation_to_prefix_errors
from exp_common.det_prefix import compute_prefix_error_stats


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_quality_gate_cfg(sc_det_cfg: dict[str, Any]) -> dict[str, Any]:
    raw_cfg = sc_det_cfg.get("quality_gate") or {}
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}
    policy_label = str(raw_cfg.get("policy_label") or "hybrid_prefix_quality_gate").strip()
    if not policy_label:
        policy_label = "hybrid_prefix_quality_gate"
    fallback_policy = str(raw_cfg.get("fallback_policy") or "disable_det").strip().lower()
    if fallback_policy not in {"disable_det", "keep_det"}:
        fallback_policy = "disable_det"
    return {
        "enabled": _to_bool(raw_cfg.get("enabled")),
        "policy_label": policy_label,
        "max_prefix_error_mean": _to_float(raw_cfg.get("max_prefix_error_mean")),
        "max_prefix_error_p95": _to_float(raw_cfg.get("max_prefix_error_p95")),
        "fallback_policy": fallback_policy,
        "require_measured_accuracy": _to_bool(raw_cfg.get("require_measured_accuracy")),
        "measured_accuracy_ready": _to_bool(raw_cfg.get("measured_accuracy_ready")),
    }


def normalize_det_k_by_layer(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    normalized: dict[str, int] = {}
    for key, raw in value.items():
        k_value = _to_float(raw)
        if k_value is None:
            continue
        normalized[str(key)] = int(round(k_value))
    return normalized or None


def resolve_det_policy_label(
    sc_det_cfg: dict[str, Any],
    switches: dict[str, Any] | None,
) -> str | None:
    if not bool((switches or {}).get("det")):
        return None
    early_stop = sc_det_cfg.get("early_stop") or {}
    if not bool(early_stop.get("enabled")):
        return None
    quality_gate = _resolve_quality_gate_cfg(sc_det_cfg)
    if quality_gate["enabled"]:
        if normalize_det_k_by_layer(early_stop.get("k_by_layer")):
            return f"{quality_gate['policy_label']}_per_layer"
        if _to_float(early_stop.get("k_global")) is not None:
            return quality_gate["policy_label"]
        return f"{quality_gate['policy_label']}_enabled_no_k"
    if normalize_det_k_by_layer(early_stop.get("k_by_layer")):
        return "per_layer"
    if _to_float(early_stop.get("k_global")) is not None:
        return "global_k"
    return "enabled_no_k"


def build_det_policy_signature(
    sc_det_cfg: dict[str, Any],
    switches: dict[str, Any] | None,
) -> str | None:
    policy = resolve_det_policy_label(sc_det_cfg, switches)
    if policy is None:
        return None
    early_stop = sc_det_cfg.get("early_stop") or {}
    payload: dict[str, Any] = {"policy": policy}
    if policy == "per_layer" or policy.endswith("_per_layer"):
        payload["k_by_layer"] = normalize_det_k_by_layer(early_stop.get("k_by_layer")) or {}
    elif policy == "global_k":
        payload["k_global"] = int(round(_to_float(early_stop.get("k_global")) or 0.0))
    elif policy.endswith("_per_layer"):
        payload["k_by_layer"] = normalize_det_k_by_layer(early_stop.get("k_by_layer")) or {}
    elif policy not in {"enabled_no_k"} and _to_float(early_stop.get("k_global")) is not None:
        payload["k_global"] = int(round(_to_float(early_stop.get("k_global")) or 0.0))
    quality_gate = _resolve_quality_gate_cfg(sc_det_cfg)
    if quality_gate["enabled"]:
        payload["quality_gate"] = {
            "policy_label": quality_gate["policy_label"],
            "max_prefix_error_mean": quality_gate["max_prefix_error_mean"],
            "max_prefix_error_p95": quality_gate["max_prefix_error_p95"],
            "fallback_policy": quality_gate["fallback_policy"],
            "require_measured_accuracy": quality_gate["require_measured_accuracy"],
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _resolve_det_mode(sc_det_cfg: dict[str, Any]) -> str:
    mode = str(sc_det_cfg.get("det_mode") or "reorder").strip().lower()
    return mode if mode in {"reorder", "replace"} else "reorder"


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(float(v) for v in values)
    idx = int(math.ceil(q * len(sorted_values))) - 1
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


def _format_float(value: float | None) -> str | None:
    if value is None:
        return None
    return str(float(value))


def resolve_det_runtime_metadata(
    sc_det_cfg: dict[str, Any],
    switches: dict[str, Any] | None,
) -> dict[str, Any]:
    policy = resolve_det_policy_label(sc_det_cfg, switches)
    signature = build_det_policy_signature(sc_det_cfg, switches)
    early_stop = sc_det_cfg.get("early_stop") or {}
    bsl_max = max(1, int(_to_float(sc_det_cfg.get("bsl_max")) or 1))
    quality_gate = _resolve_quality_gate_cfg(sc_det_cfg)

    metadata = {
        "det_policy": policy,
        "det_k_signature": signature,
        "det_mode": _resolve_det_mode(sc_det_cfg),
        "det_bsl_max": str(float(bsl_max)),
        "det_k_global": None,
        "det_k_by_layer": None,
        "det_prefix_error_mean": None,
        "det_prefix_error_p95": None,
        "det_prefix_error_by_layer": None,
        "det_quality_gate_enabled": quality_gate["enabled"],
        "det_quality_gate_policy": quality_gate["policy_label"] if quality_gate["enabled"] else None,
        "det_quality_gate_status": "det_disabled",
        "det_quality_gate_reason": "det_switch_disabled",
        "det_quality_gate_fallback_policy": quality_gate["fallback_policy"],
        "det_quality_gate_require_measured_accuracy": quality_gate["require_measured_accuracy"],
        "det_quality_gate_measured_accuracy_ready": quality_gate["measured_accuracy_ready"],
        "det_quality_gate_max_prefix_error_mean": _format_float(
            quality_gate["max_prefix_error_mean"]
        ),
        "det_quality_gate_max_prefix_error_p95": _format_float(
            quality_gate["max_prefix_error_p95"]
        ),
        "det_runtime_enabled": False,
    }
    if policy is None:
        return metadata

    metadata["det_quality_gate_reason"] = (
        "quality_gate_disabled" if not quality_gate["enabled"] else "pending_evaluation"
    )

    mean_value: float | None = None
    p95_value: float | None = None
    if policy == "per_layer":
        k_by_layer = normalize_det_k_by_layer(early_stop.get("k_by_layer")) or {}
        if not k_by_layer:
            return metadata
        metadata["det_k_by_layer"] = {
            str(key): int(value) for key, value in k_by_layer.items()
        }
        prefix_by_layer = allocation_to_prefix_errors(k_by_layer, bsl_max=bsl_max)
        prefix_values = list(prefix_by_layer.values())
        mean_value = (
            sum(float(v) for v in prefix_values) / len(prefix_values)
            if prefix_values
            else None
        )
        p95_value = _percentile(prefix_values, 0.95)
        metadata["det_prefix_error_mean"] = _format_float(mean_value)
        metadata["det_prefix_error_p95"] = _format_float(p95_value)
        metadata["det_prefix_error_by_layer"] = {
            str(key): float(value) for key, value in prefix_by_layer.items()
        }
    else:
        k_global = _to_float(early_stop.get("k_global"))
        if k_global is not None:
            target_k = max(1, min(int(round(k_global)), bsl_max))
            raw_grid = early_stop.get("k_grid") or [target_k, bsl_max]
            k_grid = sorted(
                {
                    max(1, min(int(round(float(k))), bsl_max))
                    for k in raw_grid
                    if k is not None
                }
            )
            if target_k not in k_grid:
                k_grid = sorted(set(k_grid + [target_k]))

            prefix_cfg = sc_det_cfg.get("prefix_error") or {}
            rows = compute_prefix_error_stats(
                bsl_max=bsl_max,
                k_grid=k_grid,
                num_prob_points=int(float(prefix_cfg.get("num_prob_points") or 129)),
                p_min=float(prefix_cfg.get("p_min") or 1e-3),
                p_max=float(prefix_cfg.get("p_max") or (1.0 - 1e-3)),
                det_mode=_resolve_det_mode(sc_det_cfg),
                phase_shift=int(float(prefix_cfg.get("phase_shift") or 0)),
                scramble_seed=int(float(prefix_cfg.get("scramble_seed") or 0)),
                enforce_monotonic=False,
            )
            metadata["det_k_global"] = _format_float(float(target_k))
            if rows:
                selected = min(rows, key=lambda row: abs(float(row.get("k") or 0.0) - target_k))
                mean_value = _to_float(selected.get("prefix_error_mean"))
                p95_value = _to_float(selected.get("prefix_error_p95"))
                metadata["det_prefix_error_mean"] = _format_float(mean_value)
                metadata["det_prefix_error_p95"] = _format_float(p95_value)

    if not quality_gate["enabled"]:
        metadata["det_quality_gate_status"] = "quality_gate_disabled"
        metadata["det_quality_gate_reason"] = "quality_gate_disabled"
        metadata["det_runtime_enabled"] = True
        return metadata

    blockers: list[str] = []
    mean_limit = quality_gate["max_prefix_error_mean"]
    p95_limit = quality_gate["max_prefix_error_p95"]
    if mean_limit is not None:
        if mean_value is None:
            blockers.append("missing_prefix_error_mean")
        elif mean_value > mean_limit + 1e-12:
            blockers.append("prefix_error_mean_exceeds_gate")
    if p95_limit is not None:
        if p95_value is None:
            blockers.append("missing_prefix_error_p95")
        elif p95_value > p95_limit + 1e-12:
            blockers.append("prefix_error_p95_exceeds_gate")
    if quality_gate["require_measured_accuracy"] and not quality_gate["measured_accuracy_ready"]:
        blockers.append("measured_accuracy_not_marked_ready")

    if blockers:
        metadata["det_quality_gate_reason"] = ",".join(blockers)
        if quality_gate["fallback_policy"] == "disable_det":
            metadata["det_quality_gate_status"] = "disabled_by_quality_gate"
            metadata["det_runtime_enabled"] = False
        else:
            metadata["det_quality_gate_status"] = "warning_threshold_exceeded"
            metadata["det_runtime_enabled"] = True
        return metadata

    metadata["det_quality_gate_status"] = "pass"
    metadata["det_quality_gate_reason"] = "thresholds_satisfied"
    metadata["det_runtime_enabled"] = True
    return metadata


__all__ = [
    "build_det_policy_signature",
    "normalize_det_k_by_layer",
    "resolve_det_policy_label",
    "resolve_det_runtime_metadata",
]
