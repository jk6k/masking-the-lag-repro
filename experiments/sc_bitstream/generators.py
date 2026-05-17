"""Bitstream generation utilities."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from math import gcd, sqrt
import random
from typing import Iterable, Mapping

import numpy as np

from .encoding import BIPOLAR, normalize_encoding_mode, value_to_probability

_UINT64_MASK = (1 << 64) - 1
_GOLDEN_RATIO_64 = 0x9E3779B97F4A7C15
_MIX64_1 = 0xBF58476D1CE4E5B9
_MIX64_2 = 0x94D049BB133111EB
_POLICY_STATE_SINGLE_DEFAULT = "single_default_justified"
_POLICY_STATE_CONDITIONAL = "conditional_by_stream_length"
_POLICY_STATE_MIXED = "mixed_unresolved"
_POLICY_STATE_OUT_OF_BOUNDS = "out_of_band"
_REGION_POLICY_DEFAULT = "default_with_supporting_assumptions"
_REGION_POLICY_UNRESOLVED = "mixed_unresolved"
_REGION_POLICY_SUPPORTING = "supporting_comparator"
_CORRELATION_RISK_LOW = "low_correlation"
_CORRELATION_RISK_PARTIAL = "partially_correlated"
_CORRELATION_RISK_HIGH = "high_correlation"
_CORRELATION_RISK_INSUFFICIENT = "insufficient_lanes"
_STREAM_CORRELATION_LANE_MODES = {
    "independent_lanes",
    "partially_shared_lanes",
    "shared_lane_state",
}
_STREAM_STATE_POLICY_FIELDS = (
    "generator_family",
    "phase_offset_policy",
    "stream_reuse_policy",
    "correlation_class",
    "lane_sharing_assumption",
)
_STREAM_STATE_POLICY_DEFAULTS: dict[str, dict[str, str]] = {
    "bernoulli": {
        "generator_family": "stochastic_iid",
        "phase_offset_policy": "phase_agnostic",
        "stream_reuse_policy": "fresh_rng_draws_per_lane",
        "correlation_class": "iid_independent",
        "lane_sharing_assumption": "no_shared_lane_state",
    },
    "deterministic_threshold": {
        "generator_family": "phase_locked_threshold",
        "phase_offset_policy": "fixed_phase_rotation",
        "stream_reuse_policy": "reused_phase_grid",
        "correlation_class": "phase_locked_correlated",
        "lane_sharing_assumption": "shared_phase_grid",
    },
    "low_discrepancy": {
        "generator_family": "deterministic_low_discrepancy",
        "phase_offset_policy": "seeded_lane_signature",
        "stream_reuse_policy": "lane_signature_reuse",
        "correlation_class": "deterministic_decorrelated",
        "lane_sharing_assumption": "unique_lane_signature",
    },
}
_EPSILON = 1e-12


def _is_mx_array(value: object) -> bool:
    return hasattr(value, "shape") and type(value).__name__ == "array"


def _mix64(value: int) -> int:
    mixed = (int(value) + _GOLDEN_RATIO_64) & _UINT64_MASK
    mixed = ((mixed ^ (mixed >> 30)) * _MIX64_1) & _UINT64_MASK
    mixed = ((mixed ^ (mixed >> 27)) * _MIX64_2) & _UINT64_MASK
    return mixed ^ (mixed >> 31)


def _mix64_array(values: np.ndarray) -> np.ndarray:
    mixed = (values.astype(np.uint64) + np.uint64(_GOLDEN_RATIO_64)) & np.uint64(
        _UINT64_MASK
    )
    mixed = ((mixed ^ (mixed >> np.uint64(30))) * np.uint64(_MIX64_1)) & np.uint64(
        _UINT64_MASK
    )
    mixed = ((mixed ^ (mixed >> np.uint64(27))) * np.uint64(_MIX64_2)) & np.uint64(
        _UINT64_MASK
    )
    return mixed ^ (mixed >> np.uint64(31))


def _resolve_lane_signature(
    *,
    seed: int | None,
    phase: int,
    stream_length: int,
) -> int:
    seed_value = 0 if seed is None else int(seed)
    phase_value = int(phase)
    length_value = int(stream_length)
    signature = seed_value
    signature ^= _mix64(phase_value + _GOLDEN_RATIO_64)
    signature ^= _mix64(length_value + (_GOLDEN_RATIO_64 >> 1))
    return _mix64(signature)


def _resolve_lane_signature_array(
    *,
    seed_values: np.ndarray,
    phase_values: np.ndarray,
    stream_length: int,
) -> np.ndarray:
    signatures = seed_values.astype(np.uint64)
    signatures ^= _mix64_array(
        phase_values.astype(np.uint64) + np.uint64(_GOLDEN_RATIO_64)
    )
    signatures ^= _mix64_array(
        np.full_like(seed_values, np.uint64(stream_length + (_GOLDEN_RATIO_64 >> 1)))
    )
    return _mix64_array(signatures)


def _choose_coprime_stride(stream_length: int, signature: int) -> int:
    if stream_length == 1:
        return 0
    candidate = int(signature % stream_length)
    if candidate == 0:
        candidate = 1
    if candidate % 2 == 0 and stream_length % 2 == 0:
        candidate = (candidate + 1) % stream_length or 1
    while gcd(candidate, stream_length) != 1:
        candidate = (candidate + 1) % stream_length or 1
    return candidate


@lru_cache(maxsize=None)
def _lane_stride_lookup(stream_length: int) -> tuple[np.ndarray, np.ndarray]:
    stride_lookup = np.zeros((max(1, int(stream_length)),), dtype=np.int32)
    inverse_lookup = np.zeros((max(1, int(stream_length)),), dtype=np.int32)
    if stream_length <= 1:
        return stride_lookup, inverse_lookup
    for candidate in range(stream_length):
        stride = _choose_coprime_stride(stream_length, candidate)
        stride_lookup[candidate] = int(stride)
        inverse_lookup[candidate] = int(pow(int(stride), -1, int(stream_length)))
    return stride_lookup, inverse_lookup


def _clamp_probability(probability: float) -> float:
    return max(0.0, min(1.0, float(probability)))


def _probability_from_value(value: float, *, encoding_mode: str) -> float:
    if encoding_mode == BIPOLAR:
        return _clamp_probability((float(value) + 1.0) / 2.0)
    return _clamp_probability(float(value))


def _generate_low_discrepancy_stream_from_probability(
    probability: float | object,
    *,
    stream_length: int,
    seed: int | None = None,
    phase: int = 0,
    zero_stream: tuple[int, ...] | None = None,
    one_stream: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    signature = _resolve_lane_signature(seed=seed, phase=phase, stream_length=stream_length)
    stride = _choose_coprime_stride(stream_length, signature)
    offset = int(signature % stream_length)

    is_mx_array = _is_mx_array(probability)

    if not is_mx_array:
        prob = _clamp_probability(float(probability))
        if prob <= 0.0:
            return zero_stream if zero_stream is not None else tuple(0 for _ in range(stream_length))
        if prob >= 1.0:
            return one_stream if one_stream is not None else tuple(1 for _ in range(stream_length))

        target_ones = max(0, min(stream_length, int(round(prob * stream_length))))
        if target_ones == 0:
            return zero_stream if zero_stream is not None else tuple(0 for _ in range(stream_length))
        if target_ones == stream_length:
            return one_stream if one_stream is not None else tuple(1 for _ in range(stream_length))

        stream = [0] * stream_length
        position = offset
        for _ in range(target_ones):
            stream[position] = 1
            position = (position + stride) % stream_length
        return tuple(stream)

    import mlx.core as mx

    prob = mx.clip(probability, 0.0, 1.0)
    target_ones = mx.round(prob * stream_length).astype(mx.int32)
    target_ones = mx.minimum(
        mx.maximum(target_ones, mx.array(0, dtype=mx.int32)),
        mx.array(stream_length, dtype=mx.int32),
    )
    inverse_stride = 0 if stream_length <= 1 else int(pow(stride, -1, stream_length))
    cycles = mx.arange(stream_length, dtype=mx.int32)
    relative = mx.remainder((cycles - offset) * inverse_stride, stream_length)
    return (relative < target_ones).astype(mx.uint8)


def _generate_low_discrepancy_bitstreams_mx(
    values,
    *,
    stream_length: int,
    encoding_mode: str,
    seed: int | None = None,
    phase: int = 0,
):
    import mlx.core as mx

    flat_values = mx.reshape(values, (-1,))
    value_count = int(flat_values.shape[0])
    resolved_mode = normalize_encoding_mode(encoding_mode)
    if resolved_mode == BIPOLAR:
        probabilities = mx.clip((flat_values + 1.0) / 2.0, 0.0, 1.0)
    else:
        probabilities = mx.clip(flat_values, 0.0, 1.0)

    lane_indices = np.arange(value_count, dtype=np.int64)
    phase_values = lane_indices.astype(np.uint64) + np.uint64(int(phase))
    if seed is None:
        seed_values = np.zeros((value_count,), dtype=np.uint64)
    else:
        seed_values = (
            np.full((value_count,), np.uint64(int(seed)), dtype=np.uint64)
            + (lane_indices.astype(np.uint64) * np.uint64(7919))
        )
    signatures = _resolve_lane_signature_array(
        seed_values=seed_values,
        phase_values=phase_values,
        stream_length=int(stream_length),
    )
    offsets = (signatures % np.uint64(stream_length)).astype(np.int32)
    if int(stream_length) <= 1:
        inverse_strides = np.zeros((value_count,), dtype=np.int32)
    else:
        _, inverse_lookup = _lane_stride_lookup(int(stream_length))
        inverse_strides = inverse_lookup[offsets]

    cycles = mx.arange(stream_length, dtype=mx.int32)[None, :]
    target_ones = mx.round(probabilities * stream_length).astype(mx.int32)[:, None]
    target_ones = mx.minimum(
        mx.maximum(target_ones, mx.array(0, dtype=mx.int32)),
        mx.array(stream_length, dtype=mx.int32),
    )
    offsets_arr = mx.array(offsets, dtype=mx.int32)[:, None]
    inverse_arr = mx.array(inverse_strides, dtype=mx.int32)[:, None]
    relative = mx.remainder((cycles - offsets_arr) * inverse_arr, stream_length)
    return (relative < target_ones).astype(mx.uint8)


def generate_bernoulli_bitstream(
    probability: float,
    *,
    stream_length: int,
    rng: random.Random,
) -> tuple[int, ...]:
    if stream_length <= 0:
        raise ValueError("stream_length must be positive.")
    prob = value_to_probability(probability, encoding_mode="unipolar")
    return tuple(1 if rng.random() < prob else 0 for _ in range(stream_length))


def generate_deterministic_threshold_bitstream(
    probability: float,
    *,
    stream_length: int,
    phase: int = 0,
) -> tuple[int, ...]:
    if stream_length <= 0:
        raise ValueError("stream_length must be positive.")
    prob = value_to_probability(probability, encoding_mode="unipolar")
    resolved_phase = int(phase) % stream_length
    return tuple(
        1
        if (((resolved_phase + index) % stream_length) + 0.5) / float(stream_length) < prob
        else 0
        for index in range(stream_length)
    )


def generate_low_discrepancy_bitstream(
    probability: float,
    *,
    stream_length: int,
    seed: int | None = None,
    phase: int = 0,
) -> tuple[int, ...]:
    if stream_length <= 0:
        raise ValueError("stream_length must be positive.")
    prob = value_to_probability(probability, encoding_mode="unipolar")
    return _generate_low_discrepancy_stream_from_probability(
        prob,
        stream_length=stream_length,
        seed=seed,
        phase=phase,
    )


def _generate_low_discrepancy_bitstreams(
    values: Iterable[float] | object,
    *,
    stream_length: int,
    encoding_mode: str,
    seed: int | None = None,
    phase: int = 0,
) -> tuple[tuple[int, ...], ...]:
    resolved_mode = normalize_encoding_mode(encoding_mode)
    resolved_seed = None if seed is None else int(seed)
    resolved_phase = int(phase)
    zero_stream = (0,) * stream_length
    one_stream = (1,) * stream_length

    is_mx_array = _is_mx_array(values)
    if is_mx_array:
        return _generate_low_discrepancy_bitstreams_mx(
            values,
            stream_length=stream_length,
            encoding_mode=encoding_mode,
            seed=seed,
            phase=phase,
        )

    # Legacy CPU sequence generator
    streams: list[tuple[int, ...]] = []
    streams_by_value: dict[float, set[tuple[int, ...]]] = defaultdict(set)

    for index, value in enumerate(values):
        value_key = float(value)
        probability = _probability_from_value(value_key, encoding_mode=resolved_mode)
        lane_seed = None if resolved_seed is None else resolved_seed + (index * 7919)
        lane_phase = resolved_phase + index
        stream = _generate_low_discrepancy_stream_from_probability(
            probability,
            stream_length=stream_length,
            seed=lane_seed,
            phase=lane_phase,
            zero_stream=zero_stream,
            one_stream=one_stream,
        )
        attempts = 0
        while stream in streams_by_value[value_key] and attempts < 8:
            attempts += 1
            lane_seed = None
            if resolved_seed is not None:
                lane_seed = _mix64(resolved_seed ^ _mix64((index + 1) * (attempts + 1)))
            lane_phase = resolved_phase + ((index + attempts) * 104729)
            stream = _generate_low_discrepancy_stream_from_probability(
                probability,
                stream_length=stream_length,
                seed=lane_seed,
                phase=lane_phase,
                zero_stream=zero_stream,
                one_stream=one_stream,
            )
        streams_by_value[value_key].add(stream)
        streams.append(stream)
    return tuple(streams)



def generate_bitstream(
    value: float,
    *,
    stream_length: int,
    generator: str = "bernoulli",
    encoding_mode: str = BIPOLAR,
    seed: int | None = None,
    phase: int = 0,
) -> tuple[int, ...]:
    normalize_encoding_mode(encoding_mode)
    probability = value_to_probability(value, encoding_mode=encoding_mode)
    generator_name = str(generator or "bernoulli").strip().lower() or "bernoulli"
    if generator_name == "bernoulli":
        rng = random.Random(seed)
        return generate_bernoulli_bitstream(probability, stream_length=stream_length, rng=rng)
    if generator_name == "deterministic_threshold":
        return generate_deterministic_threshold_bitstream(
            probability,
            stream_length=stream_length,
            phase=phase,
        )
    if generator_name == "low_discrepancy":
        return generate_low_discrepancy_bitstream(
            probability,
            stream_length=stream_length,
            seed=seed,
            phase=phase,
        )
    raise ValueError(f"Unsupported bitstream generator: {generator!r}")


def generate_bitstreams(
    values: Iterable[float] | object,
    *,
    stream_length: int,
    generator: str = "bernoulli",
    encoding_mode: str = BIPOLAR,
    seed: int | None = None,
    phase: int = 0,
) -> tuple[tuple[int, ...], ...]:
    generator_name = str(generator or "bernoulli").strip().lower() or "bernoulli"
    if _is_mx_array(values) and generator_name == "deterministic_threshold":
        import mlx.core as mx

        flat_values = mx.reshape(values, (-1,))
        probabilities = (
            mx.clip((flat_values + 1.0) / 2.0, 0.0, 1.0)
            if normalize_encoding_mode(encoding_mode) == BIPOLAR
            else mx.clip(flat_values, 0.0, 1.0)
        )
        lane_count = int(flat_values.shape[0])
        phase_arr = mx.array(
            np.arange(lane_count, dtype=np.int32) + np.int32(int(phase)),
            dtype=mx.int32,
        )[:, None]
        cycles = mx.arange(stream_length, dtype=mx.int32)[None, :]
        thresholds = (
            mx.remainder(phase_arr + cycles, stream_length).astype(mx.float32) + 0.5
        ) / float(stream_length)
        return (thresholds < probabilities[:, None]).astype(mx.uint8)
    if generator_name == "low_discrepancy":
        return _generate_low_discrepancy_bitstreams(
            values,
            stream_length=stream_length,
            encoding_mode=encoding_mode,
            seed=seed,
            phase=phase,
        )
    if _is_mx_array(values) and generator_name == "bernoulli":
        import mlx.core as mx

        flat_values = mx.reshape(values, (-1,))
        streams = [
            generate_bitstream(
                float(value),
                stream_length=stream_length,
                generator=generator_name,
                encoding_mode=encoding_mode,
                seed=None if seed is None else int(seed) + (index * 7919),
                phase=int(phase) + index,
            )
            for index, value in enumerate(np.asarray(flat_values.tolist(), dtype=float))
        ]
        return mx.array(streams, dtype=mx.uint8)

    streams: list[tuple[int, ...]] = []
    for index, value in enumerate(values):
        value_key = float(value)
        lane_seed = None if seed is None else int(seed) + (index * 7919)
        lane_phase = int(phase) + index
        stream = generate_bitstream(
            value_key,
            stream_length=stream_length,
            generator=generator_name,
            encoding_mode=encoding_mode,
            seed=lane_seed,
            phase=lane_phase,
        )
        streams.append(stream)
    return tuple(streams)


def _binary_pearson_correlation(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> float:
    if len(lhs) != len(rhs):
        raise ValueError("streams must have equal length.")
    if not lhs:
        return 0.0
    lhs_mean = sum(lhs) / float(len(lhs))
    rhs_mean = sum(rhs) / float(len(rhs))
    numerator = sum((a - lhs_mean) * (b - rhs_mean) for a, b in zip(lhs, rhs))
    lhs_var = sum((a - lhs_mean) ** 2 for a in lhs)
    rhs_var = sum((b - rhs_mean) ** 2 for b in rhs)
    denominator = sqrt(lhs_var * rhs_var)
    if denominator <= _EPSILON:
        return 1.0 if lhs == rhs else 0.0
    return numerator / denominator


def _correlation_risk_class(max_abs_pairwise_correlation: float | None) -> str:
    if max_abs_pairwise_correlation is None:
        return _CORRELATION_RISK_INSUFFICIENT
    if max_abs_pairwise_correlation >= 0.95:
        return _CORRELATION_RISK_HIGH
    if max_abs_pairwise_correlation >= 0.35:
        return _CORRELATION_RISK_PARTIAL
    return _CORRELATION_RISK_LOW


def _generate_probe_streams(
    values: Iterable[float],
    *,
    stream_length: int,
    generator: str,
    encoding_mode: str,
    seed: int | None,
    phase: int,
    lane_mode: str,
) -> tuple[tuple[int, ...], ...]:
    values_tuple = tuple(float(value) for value in values)
    if lane_mode == "independent_lanes":
        return generate_bitstreams(
            values_tuple,
            stream_length=stream_length,
            generator=generator,
            encoding_mode=encoding_mode,
            seed=seed,
            phase=phase,
        )
    if lane_mode == "partially_shared_lanes":
        return tuple(
            generate_bitstream(
                value,
                stream_length=stream_length,
                generator=generator,
                encoding_mode=encoding_mode,
                seed=seed,
                phase=int(phase) + lane_index,
            )
            for lane_index, value in enumerate(values_tuple)
        )
    if lane_mode == "shared_lane_state":
        return tuple(
            generate_bitstream(
                value,
                stream_length=stream_length,
                generator=generator,
                encoding_mode=encoding_mode,
                seed=seed,
                phase=phase,
            )
            for value in values_tuple
        )
    raise ValueError(f"Unsupported stream correlation lane_mode: {lane_mode!r}")


def summarize_stream_correlations(streams: Iterable[tuple[int, ...]]) -> dict[str, object]:
    """Summarize pairwise binary-stream correlations for lane-state probes."""

    streams_tuple = tuple(tuple(int(bit) for bit in stream) for stream in streams)
    pairwise: list[dict[str, object]] = []
    for lhs_index in range(len(streams_tuple)):
        for rhs_index in range(lhs_index + 1, len(streams_tuple)):
            correlation = _binary_pearson_correlation(
                streams_tuple[lhs_index],
                streams_tuple[rhs_index],
            )
            pairwise.append(
                {
                    "lhs_lane": lhs_index,
                    "rhs_lane": rhs_index,
                    "correlation": correlation,
                    "abs_correlation": abs(correlation),
                }
            )

    abs_values = [float(row["abs_correlation"]) for row in pairwise]
    max_abs = max(abs_values) if abs_values else None
    mean_abs = sum(abs_values) / float(len(abs_values)) if abs_values else None
    return {
        "lane_count": len(streams_tuple),
        "pair_count": len(pairwise),
        "pairwise_correlations": pairwise,
        "max_abs_pairwise_correlation": max_abs,
        "mean_abs_pairwise_correlation": mean_abs,
        "correlation_risk_class": _correlation_risk_class(max_abs),
    }


def run_stream_correlation_probe(
    values: Iterable[float],
    *,
    stream_length: int,
    generator: str = "bernoulli",
    encoding_mode: str = BIPOLAR,
    seed: int | None = None,
    phase: int = 0,
    lane_mode: str = "independent_lanes",
    policy_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run a bounded machine-readable stream correlation probe.

    ``lane_mode`` intentionally covers the T3 stress cases:
    independent lanes, partially shared state, and fully shared lane state.
    """

    if stream_length <= 0:
        raise ValueError("stream_length must be positive.")
    normalized_lane_mode = str(lane_mode or "").strip().lower()
    if normalized_lane_mode not in _STREAM_CORRELATION_LANE_MODES:
        raise ValueError(f"Unsupported stream correlation lane_mode: {lane_mode!r}")

    generator_name = str(generator or "bernoulli").strip().lower() or "bernoulli"
    stream_state_policy = resolve_generator_stream_state_policy(
        generator_name,
        policy_config=policy_config,
    )
    streams = _generate_probe_streams(
        values,
        stream_length=stream_length,
        generator=generator_name,
        encoding_mode=encoding_mode,
        seed=seed,
        phase=phase,
        lane_mode=normalized_lane_mode,
    )
    summary = summarize_stream_correlations(streams)
    return {
        "probe_kind": "stream_correlation_probe",
        "generator": generator_name,
        "stream_length": int(stream_length),
        "encoding_mode": normalize_encoding_mode(encoding_mode),
        "seed": seed,
        "phase": int(phase),
        "lane_mode": normalized_lane_mode,
        "stream_state_policy": stream_state_policy,
        **summary,
    }


def run_stream_correlation_scenarios(
    values: Iterable[float],
    *,
    stream_length: int,
    generator: str = "bernoulli",
    encoding_mode: str = BIPOLAR,
    seed: int | None = None,
    phase: int = 0,
    lane_modes: Iterable[str] = (
        "independent_lanes",
        "partially_shared_lanes",
        "shared_lane_state",
    ),
    policy_config: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    """Run the standard T3 correlation stress scenarios."""

    return [
        run_stream_correlation_probe(
            values,
            stream_length=stream_length,
            generator=generator,
            encoding_mode=encoding_mode,
            seed=seed,
            phase=phase,
            lane_mode=lane_mode,
            policy_config=policy_config,
        )
        for lane_mode in lane_modes
    ]


def resolve_generator_stream_state_policy(
    generator: str,
    *,
    policy_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Resolve the machine-readable stream-state policy for a generator.

    The policy is deterministic by default and can be lightly overridden by a
    config mapping under ``stream_state_policy``.
    """

    generator_name = str(generator or "").strip().lower()
    if not generator_name:
        raise ValueError("generator must be a non-empty string.")
    if generator_name not in _STREAM_STATE_POLICY_DEFAULTS:
        raise ValueError(f"Unsupported bitstream generator: {generator!r}")

    resolved_policy = dict(_STREAM_STATE_POLICY_DEFAULTS[generator_name])
    overrides: dict[str, str] = {}
    if policy_config is not None:
        override_sources: list[Mapping[str, object]] = []
        nested_policy = policy_config.get("stream_state_policy") if isinstance(policy_config, Mapping) else None
        if isinstance(nested_policy, Mapping):
            override_sources.append(nested_policy)
        override_sources.append(policy_config)
        for source in override_sources:
            for field in _STREAM_STATE_POLICY_FIELDS:
                if field in source and source[field] is not None:
                    overrides[field] = _coerce_text(source[field])

    resolved_policy.update(overrides)
    return {
        "policy_kind": "generator_stream_state_policy",
        "policy_state": "config_override" if overrides else "default",
        "generator": generator_name,
        **resolved_policy,
        "override_fields": sorted(overrides),
    }


def _coerce_text(value: object) -> str:
    return str(value or "").strip()


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _scope_key(workload_class: str, stream_length: int) -> str:
    return f"{workload_class}|{int(stream_length)}"


def _matches_scope(
    row: Mapping[str, object],
    *,
    workload_class: str | None,
    stream_length: int | None,
) -> bool:
    if workload_class is not None and _coerce_text(row.get("workload_class")) != workload_class:
        return False
    if stream_length is not None and _coerce_int(row.get("stream_length"), -1) != int(stream_length):
        return False
    return True


def _group_policy_rows(
    evidence_rows: Iterable[Mapping[str, object]],
) -> dict[tuple[str, int], list[dict[str, object]]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in evidence_rows:
        workload_class = _coerce_text(row.get("workload_class") or row.get("model_family"))
        stream_length = _coerce_int(row.get("stream_length"), 0)
        generator = _coerce_text(row.get("generator"))
        if not workload_class or stream_length <= 0 or not generator:
            continue
        grouped[(workload_class, stream_length)].append(dict(row))
    return grouped


def _summarize_policy_region(
    workload_class: str,
    stream_length: int,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    ranked_rows = sorted(
        rows,
        key=lambda row: (
            _coerce_float(row.get("median_abs_error_vs_raw_exact"), float("inf")),
            _coerce_text(row.get("generator")),
        ),
    )
    metrics = [_coerce_float(row.get("median_abs_error_vs_raw_exact"), float("inf")) for row in ranked_rows]
    best_metric = min(metrics)
    best_rows = [row for row in ranked_rows if abs(_coerce_float(row.get("median_abs_error_vs_raw_exact"), float("inf")) - best_metric) <= _EPSILON]
    runner_up_metric = None
    for metric in metrics:
        if metric > best_metric + _EPSILON:
            runner_up_metric = metric
            break

    generator_ranking = []
    seen_metrics: list[float] = []
    for row in ranked_rows:
        metric = _coerce_float(row.get("median_abs_error_vs_raw_exact"), float("inf"))
        rank = 1 + sum(1 for seen in seen_metrics if seen + _EPSILON < metric)
        seen_metrics.append(metric)
        generator_ranking.append(
            {
                "generator": _coerce_text(row.get("generator")),
                "generator_rank": rank,
                "median_abs_error_vs_raw_exact": metric,
                "metric_margin_vs_best": metric - best_metric,
                "summary_count": _coerce_int(row.get("summary_count"), 0),
            }
        )

    unique_best = len(best_rows) == 1
    best_generator = _coerce_text(best_rows[0].get("generator")) if unique_best else ""
    region_policy_state = _REGION_POLICY_DEFAULT if unique_best else _REGION_POLICY_UNRESOLVED

    return {
        "scope_key": _scope_key(workload_class, stream_length),
        "workload_class": workload_class,
        "model_family": workload_class,
        "stream_length": stream_length,
        "generator_count": len(ranked_rows),
        "summary_count": sum(_coerce_int(row.get("summary_count"), 0) for row in ranked_rows),
        "sample_count": sum(_coerce_int(row.get("sample_count"), 0) for row in ranked_rows),
        "best_generator": best_generator,
        "best_generators": [_coerce_text(row.get("generator")) for row in best_rows],
        "best_generator_count": len(best_rows),
        "best_generator_metric": best_metric,
        "runner_up_metric": runner_up_metric,
        "policy_state": region_policy_state,
        "default_generator": best_generator if unique_best else "",
        "generator_ranking": generator_ranking,
    }


def resolve_generator_default_policy(
    evidence_rows: Iterable[Mapping[str, object]],
    *,
    workload_class: str | None = None,
    stream_length: int | None = None,
) -> dict[str, object]:
    """Resolve a generator default policy from a machine-readable evidence matrix.

    The resolver is intentionally conservative: it only promotes a single
    repository-wide default when every observed region points to the same
    generator. Otherwise, it returns a conditional policy that keeps the mixed
    regions explicit.
    """

    grouped_rows = _group_policy_rows(evidence_rows)
    filtered_groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for key, rows in grouped_rows.items():
        if workload_class is not None and key[0] != workload_class:
            continue
        if stream_length is not None and key[1] != int(stream_length):
            continue
        filtered_groups[key] = rows

    if not filtered_groups:
        return {
            "policy_kind": "generator_default_policy",
            "policy_state": _POLICY_STATE_OUT_OF_BOUNDS,
            "repository_default_generator": None,
            "workload_class": workload_class,
            "stream_length": stream_length,
            "regions": [],
            "region_lookup": {},
            "regional_default_generators": [],
            "regional_default_generator_counts": {},
            "resolved_region_count": 0,
            "out_of_band": True,
        }

    regions = [
        _summarize_policy_region(workload_class_name, resolved_stream_length, rows)
        for (workload_class_name, resolved_stream_length), rows in sorted(filtered_groups.items())
    ]
    region_lookup = {str(region["scope_key"]): region for region in regions}
    regional_default_generators = sorted(
        {
            region["default_generator"]
            for region in regions
            if _coerce_text(region.get("default_generator"))
        }
    )
    regional_default_generator_counts: dict[str, int] = defaultdict(int)
    for region in regions:
        default_generator = _coerce_text(region.get("default_generator"))
        if default_generator:
            regional_default_generator_counts[default_generator] += 1

    resolved_region_count = len(regions)
    unresolved_region_count = sum(1 for region in regions if region["policy_state"] != _REGION_POLICY_DEFAULT)
    if len(regional_default_generators) == 1 and unresolved_region_count == 0:
        policy_state = _POLICY_STATE_SINGLE_DEFAULT
        repository_default_generator = regional_default_generators[0]
    elif unresolved_region_count > 0:
        policy_state = _POLICY_STATE_MIXED
        repository_default_generator = None
    else:
        policy_state = _POLICY_STATE_CONDITIONAL
        repository_default_generator = None

    if workload_class is not None and stream_length is not None and len(regions) == 1:
        return {
            "policy_kind": "generator_default_policy",
            "policy_state": regions[0]["policy_state"],
            "repository_default_generator": regions[0]["default_generator"] or None,
            "workload_class": workload_class,
            "stream_length": stream_length,
            "region": regions[0],
            "regions": regions,
            "region_lookup": region_lookup,
            "regional_default_generators": regional_default_generators,
            "regional_default_generator_counts": dict(regional_default_generator_counts),
            "resolved_region_count": resolved_region_count,
            "out_of_band": False,
        }

    return {
        "policy_kind": "generator_default_policy",
        "policy_state": policy_state,
        "repository_default_generator": repository_default_generator,
        "workload_class": workload_class,
        "stream_length": stream_length,
        "regions": regions,
        "region_lookup": region_lookup,
        "regional_default_generators": regional_default_generators,
        "regional_default_generator_counts": dict(regional_default_generator_counts),
        "resolved_region_count": resolved_region_count,
        "out_of_band": False,
    }
