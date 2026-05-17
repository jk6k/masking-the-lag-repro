"""Adaptive host-tuning policy resolution for current FULLER experiment surfaces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PASS_KIND_RUNTIME_SMOKE = "runtime_smoke"
PASS_KIND_ANALYSIS_GRADE_BASELINE = "analysis_grade_baseline"
PASS_KIND_ANALYSIS_GRADE_QUANTIZED = "analysis_grade_quantized"
PASS_KIND_SUPPORT_FAMILY = "support_family"

PASS_KIND_BASELINE_EVAL = "baseline_eval_pass"
PASS_KIND_QUANTIZED_EVAL = "quantized_eval_pass"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return payload


def load_host_tuning_profile(path: Path) -> dict[str, Any]:
    return _load_json(path)


def _norm_token(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _norm_lower_token(value: Any) -> str | None:
    text = _norm_token(value)
    return text.lower() if text else None


def _norm_gate(raw: Any) -> dict[str, Any]:
    gate = dict(raw or {})
    normalized: dict[str, Any] = {}
    for key in (
        "progress_heartbeat_interval_seconds",
        "stall_timeout_seconds",
        "pathological_min_samples_per_hour",
        "pathological_max_seconds_per_sample",
        "pathological_max_eta_current_rate_seconds",
        "pathological_min_processed_samples",
        "pathological_min_elapsed_seconds",
    ):
        value = gate.get(key)
        if value in ("", None):
            continue
        normalized[key] = value
    return normalized


def _norm_semantic_policy(raw: Any) -> dict[str, Any]:
    payload = dict(raw or {})
    normalized: dict[str, Any] = {}
    for key in (
        "bitstream_generator",
        "bitstream_stream_length",
        "bitstream_stream_reuse_policy",
        "bitstream_encoding_mode",
        "bitstream_multiplier_mode",
        "bitstream_accumulator_mode",
    ):
        value = payload.get(key)
        if value in ("", None):
            continue
        normalized[key] = value
    return normalized


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()[:16]


def fingerprint_runtime_policy(policy: dict[str, Any]) -> str:
    payload = {
        "pass_kind": str(policy.get("pass_kind") or ""),
        "execution_semantics": str(policy.get("execution_semantics") or ""),
        "workers": int(policy.get("workers") or 0),
        "prefetch_batches": int(policy.get("prefetch_batches") or 0),
        "eval_batch_size": int(policy.get("eval_batch_size") or 0),
        "safe_batch_cap": (
            int(policy["safe_batch_cap"])
            if policy.get("safe_batch_cap") not in (None, "")
            else None
        ),
        "runtime_health_gate": _norm_gate(policy.get("runtime_health_gate")),
    }
    return _stable_hash(payload)


def fingerprint_semantic_policy(
    semantic_policy: dict[str, Any] | None,
    *,
    execution_semantics: str | None,
) -> str:
    semantics = _norm_lower_token(execution_semantics) or ""
    if semantics != "bitstream":
        return "non_bitstream"
    return _stable_hash(
        {
            "execution_semantics": semantics,
            "semantic_policy": _norm_semantic_policy(semantic_policy),
        }
    )


def _host_profiles(payload: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = payload.get("host_profiles") or []
    if not isinstance(profiles, list):
        raise SystemExit("Host tuning profile must expose host_profiles as a list")
    return [dict(item or {}) for item in profiles]


def resolve_host_profile(
    payload: dict[str, Any],
    *,
    host_id: str | None = None,
) -> dict[str, Any]:
    selected_host_id = _norm_token(host_id) or _norm_token(payload.get("active_host_id"))
    profiles = _host_profiles(payload)
    if not profiles:
        raise SystemExit("Host tuning profile does not contain any host_profiles")
    if selected_host_id is None:
        return profiles[0]
    for profile in profiles:
        if _norm_token(profile.get("host_id")) == selected_host_id:
            return profile
    raise SystemExit(f"Unknown host_id in host tuning profile: {selected_host_id}")


def _normalized_profile_entry(
    raw: dict[str, Any],
    *,
    fallback_profile_id: str,
    default_calibration_artifact_path: str | None,
) -> dict[str, Any]:
    profile_id = _norm_token(raw.get("profile_id")) or fallback_profile_id
    return {
        "profile_id": profile_id,
        "experiment_family_id": _norm_token(raw.get("experiment_family_id")),
        "lane_id": _norm_token(raw.get("lane_id")),
        "pass_kind": _norm_token(raw.get("pass_kind")),
        "execution_semantics": _norm_lower_token(raw.get("execution_semantics")),
        "workers": int(raw["workers"]) if raw.get("workers") not in (None, "") else None,
        "eval_batch_size": (
            int(raw["eval_batch_size"]) if raw.get("eval_batch_size") not in (None, "") else None
        ),
        "prefetch_batches": (
            int(raw["prefetch_batches"]) if raw.get("prefetch_batches") not in (None, "") else None
        ),
        "runtime_health_gate": _norm_gate(raw.get("runtime_health_gate")),
        "semantic_policy": _norm_semantic_policy(raw.get("semantic_policy")),
        "calibration_artifact_path": (
            _norm_token(raw.get("calibration_artifact_path")) or default_calibration_artifact_path or ""
        ),
    }


def _merge_profile_entry(
    default_entry: dict[str, Any],
    override_entry: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(default_entry)
    for key in ("workers", "eval_batch_size", "prefetch_batches", "calibration_artifact_path"):
        if override_entry.get(key) not in (None, ""):
            merged[key] = override_entry[key]
    merged["profile_id"] = override_entry["profile_id"]
    merged["experiment_family_id"] = override_entry.get("experiment_family_id")
    merged["lane_id"] = override_entry.get("lane_id")
    merged["pass_kind"] = override_entry.get("pass_kind")
    merged["execution_semantics"] = override_entry.get("execution_semantics")
    merged["runtime_health_gate"] = {
        **dict(default_entry.get("runtime_health_gate") or {}),
        **dict(override_entry.get("runtime_health_gate") or {}),
    }
    merged["semantic_policy"] = {
        **dict(default_entry.get("semantic_policy") or {}),
        **dict(override_entry.get("semantic_policy") or {}),
    }
    return merged


def _profile_priority(
    entry: dict[str, Any],
    *,
    experiment_family_id: str | None,
    lane_id: str | None,
    pass_kind: str,
    execution_semantics: str | None,
) -> tuple[int, int]:
    family = _norm_token(experiment_family_id)
    lane = _norm_token(lane_id)
    semantics = _norm_lower_token(execution_semantics)
    entry_family = _norm_token(entry.get("experiment_family_id"))
    entry_lane = _norm_token(entry.get("lane_id"))
    entry_pass_kind = _norm_token(entry.get("pass_kind"))
    entry_semantics = _norm_lower_token(entry.get("execution_semantics"))
    if entry_family is None and entry_lane is None and entry_pass_kind is None:
        base = 0
    elif lane and entry_lane == lane and entry_pass_kind == pass_kind:
        base = 3
    elif family and entry_family == family and entry_pass_kind == pass_kind:
        base = 2
    elif family and entry_family == family:
        base = 1
    else:
        return (-1, -1)
    semantic_bonus = 1 if semantics and entry_semantics == semantics else 0
    return (base, semantic_bonus)


def resolve_host_tuning_entry(
    payload: dict[str, Any],
    *,
    experiment_family_id: str | None,
    lane_id: str | None,
    pass_kind: str,
    execution_semantics: str | None = None,
    host_id: str | None = None,
) -> dict[str, Any]:
    host_profile = resolve_host_profile(payload, host_id=host_id)
    host_profile_id = _norm_token(host_profile.get("host_profile_id")) or (
        f"{_norm_token(host_profile.get('host_id')) or 'host'}__default"
    )
    default_entry = _normalized_profile_entry(
        dict(host_profile.get("default") or {}),
        fallback_profile_id=f"{host_profile_id}__host_default",
        default_calibration_artifact_path=_norm_token(host_profile.get("calibration_artifact_path")),
    )
    selected_entry = default_entry
    best_priority = (0, 0)
    for index, raw_entry in enumerate(host_profile.get("profiles") or []):
        entry = _normalized_profile_entry(
            dict(raw_entry or {}),
            fallback_profile_id=f"{host_profile_id}__entry_{index}",
            default_calibration_artifact_path=default_entry["calibration_artifact_path"],
        )
        priority = _profile_priority(
            entry,
            experiment_family_id=experiment_family_id,
            lane_id=lane_id,
            pass_kind=pass_kind,
            execution_semantics=execution_semantics,
        )
        if priority > best_priority:
            best_priority = priority
            selected_entry = entry
    merged = _merge_profile_entry(default_entry, selected_entry)
    merged["host_id"] = _norm_token(host_profile.get("host_id")) or ""
    merged["host_profile_id"] = host_profile_id
    return merged


def command_pass_kind(
    *,
    experiment_family_id: str | None,
    pass_kind: str,
) -> str:
    family = _norm_token(experiment_family_id) or ""
    if family == "analysis_grade_replay":
        if pass_kind == PASS_KIND_BASELINE_EVAL:
            return PASS_KIND_ANALYSIS_GRADE_BASELINE
        if pass_kind == PASS_KIND_QUANTIZED_EVAL:
            return PASS_KIND_ANALYSIS_GRADE_QUANTIZED
    if family in {
        "realism_calibration_support",
        "noise_robustness",
        "scaling_support",
        "device_compare",
        "holdout_audit",
        "report_pack",
    }:
        return PASS_KIND_SUPPORT_FAMILY
    return PASS_KIND_RUNTIME_SMOKE


def resolve_accuracy_policy_bundle(
    payload: dict[str, Any],
    *,
    experiment_family_id: str | None,
    lane_id: str | None,
    pass_mode: str,
    quantized_execution_semantics: str | None,
    host_id: str | None = None,
) -> dict[str, Any]:
    pass_policies: dict[str, dict[str, Any]] = {}
    requested_passes: list[str] = []
    if pass_mode in {"paired", "baseline_only"}:
        requested_passes.append(PASS_KIND_BASELINE_EVAL)
    if pass_mode in {"paired", "quantized_only"}:
        requested_passes.append(PASS_KIND_QUANTIZED_EVAL)
    for command_pass in requested_passes:
        execution_semantics = (
            _norm_lower_token(quantized_execution_semantics)
            if command_pass == PASS_KIND_QUANTIZED_EVAL
            else "non_bitstream"
        )
        resolved = resolve_host_tuning_entry(
            payload,
            experiment_family_id=experiment_family_id,
            lane_id=lane_id,
            pass_kind=command_pass_kind(
                experiment_family_id=experiment_family_id,
                pass_kind=command_pass,
            ),
            execution_semantics=execution_semantics,
            host_id=host_id,
        )
        policy = {
            "pass_kind": command_pass,
            "profile_id": resolved["profile_id"],
            "workers": int(resolved.get("workers") or 0),
            "prefetch_batches": (
                int(resolved["prefetch_batches"])
                if resolved.get("prefetch_batches") not in (None, "")
                else None
            ),
            "eval_batch_size": (
                int(resolved["eval_batch_size"])
                if resolved.get("eval_batch_size") not in (None, "")
                else None
            ),
            "runtime_health_gate": dict(resolved.get("runtime_health_gate") or {}),
            "semantic_policy": dict(resolved.get("semantic_policy") or {}),
            "execution_semantics": execution_semantics,
            "calibration_artifact_path": str(resolved.get("calibration_artifact_path") or ""),
        }
        policy["runtime_policy_fingerprint"] = fingerprint_runtime_policy(policy)
        policy["semantic_fingerprint"] = fingerprint_semantic_policy(
            policy["semantic_policy"],
            execution_semantics=execution_semantics,
        )
        pass_policies[command_pass] = policy
    host_profile = resolve_host_profile(payload, host_id=host_id)
    host_profile_id = _norm_token(host_profile.get("host_profile_id")) or ""
    calibration_artifact_path = ""
    for policy in pass_policies.values():
        path = str(policy.get("calibration_artifact_path") or "").strip()
        if path:
            calibration_artifact_path = path
            break
    return {
        "host_id": _norm_token(host_profile.get("host_id")) or "",
        "host_profile_id": host_profile_id,
        "experiment_family_id": _norm_token(experiment_family_id) or "",
        "lane_id": _norm_token(lane_id) or "",
        "pass_mode": str(pass_mode),
        "pass_policies": pass_policies,
        "calibration_artifact_path": calibration_artifact_path,
    }


def serialize_accuracy_policy_bundle(bundle: dict[str, Any]) -> str:
    return json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_accuracy_policy_bundle(raw: str | Path | None) -> dict[str, Any] | None:
    if raw in (None, ""):
        return None
    if isinstance(raw, Path):
        return load_host_tuning_profile(raw)
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_file():
        return _load_json(path)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise SystemExit("Expected accuracy policy bundle JSON object")
    return payload


def resolve_pass_policy_from_bundle(
    bundle: dict[str, Any] | None,
    *,
    pass_kind: str,
) -> dict[str, Any] | None:
    if bundle is None:
        return None
    pass_policies = dict(bundle.get("pass_policies") or {})
    raw_policy = pass_policies.get(pass_kind)
    if not isinstance(raw_policy, dict):
        return None
    return dict(raw_policy)


def calibration_selector_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("candidates") or []
    if not isinstance(candidates, list):
        raise SystemExit("Calibration selector candidates must be a list")
    return [dict(item or {}) for item in candidates]


def select_calibration_candidate(
    candidates: list[dict[str, Any]],
    *,
    quantized: bool,
    throughput_tolerance: float = 0.05,
) -> dict[str, Any]:
    eligible: list[dict[str, Any]] = []
    for candidate in candidates:
        failure_flags = {
            str(item).strip().lower()
            for item in candidate.get("failure_kinds") or []
            if str(item).strip()
        }
        if failure_flags & {"crash", "stall", "sigkill", "unsafe_guardrail", "resume_incompatible"}:
            continue
        if quantized and not bool(candidate.get("working_set_guardrail_ok", True)):
            continue
        eligible.append(dict(candidate))
    if not eligible:
        raise SystemExit("No eligible calibration candidates remain after guardrails")

    def _sort_key(candidate: dict[str, Any]) -> tuple[float, int, int]:
        return (
            float(candidate.get("samples_per_hour") or 0.0),
            -int(candidate.get("workers") or 0),
            -int(candidate.get("eval_batch_size") or 0),
        )

    eligible.sort(key=_sort_key, reverse=True)
    best = eligible[0]
    best_throughput = float(best.get("samples_per_hour") or 0.0)
    within_tolerance = [
        dict(candidate)
        for candidate in eligible
        if best_throughput <= 0.0
        or (best_throughput - float(candidate.get("samples_per_hour") or 0.0)) / best_throughput
        <= float(throughput_tolerance)
    ]
    within_tolerance.sort(
        key=lambda candidate: (
            int(candidate.get("workers") or 0),
            int(candidate.get("eval_batch_size") or 0),
            -float(candidate.get("samples_per_hour") or 0.0),
        )
    )
    return within_tolerance[0]


def semantic_promotion_eligible(
    payload: dict[str, Any],
) -> bool:
    throughput_gain_percent = float(payload.get("throughput_gain_percent") or 0.0)
    top1_drift_pp = abs(float(payload.get("top1_drift_pp") or 0.0))
    governance_blocker = bool(payload.get("governance_blocker"))
    return (
        throughput_gain_percent >= 25.0
        and top1_drift_pp <= 0.5
        and not governance_blocker
    )
