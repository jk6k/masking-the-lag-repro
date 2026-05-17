"""MLX MobileViT inference stack for Apple Silicon evaluation."""

from __future__ import annotations

import hashlib
import math
import os
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

mx = sys.modules.get("mlx.core")
nn = sys.modules.get("mlx.nn")
if mx is None or nn is None:
    try:
        import mlx.core as mx_module
        import mlx.nn as nn_module
    except Exception:
        mx = sys.modules.get("mlx.core")
        nn = sys.modules.get("mlx.nn")
    else:
        mx = mx_module
        nn = nn_module
if nn is None:
    nn = types.SimpleNamespace(Module=object)

from accuracy.bitstream_capture import build_capture_record
from exp_common.cvnets_utils import parse_weights_override, resolve_weights_path
from exp_common.model_specs import MODEL_SPECS
from sc_bitstream.generators import generate_bitstreams
from sc_bitstream.kernels import (
    estimate_dot_product_value,
    estimate_dot_product_value_from_stream_arrays,
)
from sc_bitstream.generators import (
    _lane_stride_lookup,
    _resolve_lane_signature_array,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
BITSTREAM_RUNTIME_FAMILY_POLICY_SOURCE = (
    "accuracy.mlx_mobilevit:bitstream_runtime_family_policy_v1"
)
DEFAULT_BITSTREAM_RUNTIME_STREAM_REUSE_POLICY = "operand_factored_module_call_reuse"
_BITSTREAM_STATE_MODULUS = 1_000_000_007
_BITSTREAM_MATMUL_TILE_BUDGET = 32768
_BITSTREAM_CONV_SPATIAL_TILE_BUDGET = 16
_BITSTREAM_STREAM_ENCODE_TILE_BUDGET = 65536
_BITSTREAM_MX_ACCUMULATOR_EVAL_TILE_STRIDE = 16
_BITSTREAM_MATMUL_PROGRESS_TILE_STRIDE = 8
_BITSTREAM_MATMUL_ROW_BAND_PROGRESS_STRIDE = 4
_BITSTREAM_MATMUL_BATCH_TILE_COUNT = 256
_UINT64_WRAP = 1 << 64
_BITSTREAM_RUNTIME_FAMILY_POLICY: dict[str, dict[str, str]] = {
    "activation": {
        "coverage_class": "targetable_module_family",
        "coverage_mechanism": "targetable_module_family",
        "claim_surface_role": "runtime_targetable_module_family",
    },
    "conv2d": {
        "coverage_class": "targetable_module_family",
        "coverage_mechanism": "targetable_module_family",
        "claim_surface_role": "runtime_targetable_module_family",
    },
    "linear": {
        "coverage_class": "targetable_module_family",
        "coverage_mechanism": "targetable_module_family",
        "claim_surface_role": "runtime_targetable_module_family",
    },
    "norm": {
        "coverage_class": "targetable_module_family",
        "coverage_mechanism": "targetable_module_family",
        "claim_surface_role": "runtime_targetable_module_family",
    },
    "softmax": {
        "coverage_class": "workload_native_support_qualified_family",
        "coverage_mechanism": "declared_runtime_support_policy",
        "claim_surface_role": "runtime_workload_support_qualified_family",
    },
}
_mlx_photonic_perturb_module = None
if "mlx.core" in sys.modules:
    try:
        from accuracy import mlx_photonic_perturb as _mlx_photonic_perturb_module
    except Exception:
        _mlx_photonic_perturb_module = None


def _ensure_mlx_modules() -> tuple[Any, Any]:
    global mx, nn
    if mx is not None and nn is not None and hasattr(nn, "Conv2d"):
        return mx, nn
    import mlx.core as mx_module
    import mlx.nn as nn_module

    mx = mx_module
    nn = nn_module
    return mx, nn


def _load_mlx_photonic_perturb_module():
    global _mlx_photonic_perturb_module
    if _mlx_photonic_perturb_module is None:
        from accuracy import mlx_photonic_perturb as mlx_photonic_perturb_module

        _mlx_photonic_perturb_module = mlx_photonic_perturb_module
    return _mlx_photonic_perturb_module


@dataclass
class BitstreamExecutionConfig:
    """Runtime options for bounded true-bitstream execution."""

    enabled: bool = False
    target_module_keys: set[str] | None = None
    encoding_mode: str = "bipolar"
    multiplier_mode: str = "xnor"
    accumulator_mode: str = "bitcount"
    stream_length: int = 64
    generator: str = "low_discrepancy"
    stream_reuse_policy: str = DEFAULT_BITSTREAM_RUNTIME_STREAM_REUSE_POLICY
    seed: int | None = 0
    surface_scope: str = ""
    measurement_truth_class: str = ""
    contract_note: str = ""
    progress_callback: Any | None = None
    module_call_counts: dict[str, int] = field(default_factory=dict)

    def next_call_index(self, module_key: str) -> int:
        call_index = int(self.module_call_counts.get(module_key, 0))
        self.module_call_counts[module_key] = call_index + 1
        return call_index

    def emit_progress(
        self,
        *,
        module_key: str,
        stage: str,
        call_index: int,
        detail: str = "",
    ) -> None:
        callback = self.progress_callback
        if callable(callback):
            callback(
                module_key=str(module_key),
                stage=str(stage),
                call_index=int(call_index),
                detail=str(detail),
                timestamp=time.perf_counter(),
            )


@dataclass
class BitstreamSliceRecorder:
    """Capture bounded operand/output samples for offline replay."""

    enabled: bool = False
    target_module_keys: set[str] | None = None
    encoding_mode: str = "bipolar"
    multiplier_mode: str = "xnor"
    accumulator_mode: str = "bitcount"
    stream_length: int = 64
    generator: str = "low_discrepancy"
    max_values_per_capture: int = 256
    _rows: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self._rows.clear()

    def record(
        self,
        module_key: str,
        tensor: Any,
        *,
        operand_role: str = "output",
        call_index: int = 0,
        row_context: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        requested = self.target_module_keys
        if requested is not None and module_key not in requested:
            return
        context = dict(row_context or {})
        values = _tensor_to_capture_values(tensor, limit=self.max_values_per_capture)
        row = build_capture_record(
            capture_id=f"{module_key}:{operand_role}:{call_index}",
            model=str(context.get("model") or ""),
            module_key=module_key,
            operand_role=operand_role,
            tensor_shape=tuple(int(dim) for dim in getattr(tensor, "shape", ())),
            dtype=str(getattr(tensor, "dtype", "")),
            encoding_mode=self.encoding_mode,
            multiplier_mode=self.multiplier_mode,
            accumulator_mode=self.accumulator_mode,
            stream_length=self.stream_length,
            generator=self.generator,
            operand_values=values,
            notes=notes,
        )
        for key, value in context.items():
            row.setdefault(key, value)
        self._rows.append(row)

    def rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._rows]


@dataclass
class SparseActivityRecorder:
    """Track sparse gating activity statistics during MLX inference."""

    enabled: bool = False
    total_elements: int = 0
    active_elements: int = 0
    call_count: int = 0
    gate_mode: str | None = None
    per_module: dict[str, dict[str, Any]] = field(default_factory=dict)

    def reset(self) -> None:
        self.total_elements = 0
        self.active_elements = 0
        self.call_count = 0
        self.gate_mode = None
        self.per_module.clear()

    def record(self, module_key: str, stats: dict[str, Any] | None) -> None:
        if not self.enabled or not stats:
            return
        total = int(stats.get("total_elements") or 0)
        active = int(stats.get("active_elements") or 0)
        gate_mode = str(stats.get("gate_mode") or "").strip() or None
        self.total_elements += total
        self.active_elements += active
        self.call_count += 1
        if gate_mode:
            if self.gate_mode is None:
                self.gate_mode = gate_mode
            elif self.gate_mode != gate_mode:
                self.gate_mode = "mixed"
        entry = self.per_module.setdefault(
            module_key,
            {
                "gate_mode": gate_mode,
                "total_elements": 0,
                "active_elements": 0,
                "call_count": 0,
            },
        )
        entry["total_elements"] = int(entry["total_elements"]) + total
        entry["active_elements"] = int(entry["active_elements"]) + active
        entry["call_count"] = int(entry["call_count"]) + 1
        if gate_mode:
            existing_gate_mode = str(entry.get("gate_mode") or "").strip() or None
            if existing_gate_mode is None:
                entry["gate_mode"] = gate_mode
            elif existing_gate_mode != gate_mode:
                entry["gate_mode"] = "mixed"

    def summary(self) -> dict[str, Any]:
        if not self.enabled or self.total_elements <= 0:
            return {
                "sparse_gate_mode": None,
                "sparse_measured_activity_fraction": None,
                "sparse_measured_zero_fraction": None,
                "sparse_stats_total_elements": None,
                "sparse_stats_active_elements": None,
                "sparse_stats_call_count": None,
                "sparse_stats_module_count": None,
            }
        activity_fraction = self.active_elements / self.total_elements
        return {
            "sparse_gate_mode": self.gate_mode,
            "sparse_measured_activity_fraction": activity_fraction,
            "sparse_measured_zero_fraction": 1.0 - activity_fraction,
            "sparse_stats_total_elements": self.total_elements,
            "sparse_stats_active_elements": self.active_elements,
            "sparse_stats_call_count": self.call_count,
            "sparse_stats_module_count": len(self.per_module),
        }

    def layer_rows(self, *, row_context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        context = dict(row_context or {})
        for module_key, entry in sorted(self.per_module.items()):
            total = int(entry.get("total_elements") or 0)
            active = int(entry.get("active_elements") or 0)
            activity_fraction = active / total if total > 0 else None
            rows.append(
                {
                    **context,
                    "module_key": module_key,
                    "sparse_gate_mode": entry.get("gate_mode"),
                    "sparse_stats_total_elements": total,
                    "sparse_stats_active_elements": active,
                    "sparse_stats_call_count": int(entry.get("call_count") or 0),
                    "sparse_measured_activity_fraction": activity_fraction,
                    "sparse_measured_zero_fraction": (
                        None if activity_fraction is None else 1.0 - activity_fraction
                    ),
                }
            )
        return rows


@dataclass
class MLXPerturbationConfig:
    """Runtime perturbation parameters for the MLX inference path."""

    bits: int = 8
    gaussian_noise_std: float = 0.0
    crosstalk_alpha: float = 0.0
    drift_lsb: float = 0.0
    noise_correlation: float = 0.0
    burst_error_prob: float = 0.0
    burst_error_scale_lsb: float = 0.0
    burst_span: int = 1
    enabled: bool = False
    calibration_mode: bool = False
    use_calibrated_scale: bool = False
    strict_calibrated_scale: bool = False
    det_enabled: bool = False
    det_mode: str | None = None
    det_bsl_max: float | None = None
    det_k_global: float | None = None
    det_k_by_layer: dict[str, float] | None = None
    det_prefix_error_mean: float = 0.0
    det_prefix_error_by_layer: dict[str, float] | None = None
    sparse_enabled: bool = False
    sparse_active_fraction: float | None = None
    sparse_tau_global: float | None = None
    sparse_target_module_keys: set[str] | None = None
    sparse_protected_layers: set[str] | None = None
    conv_channel_dim: int = -1
    linear_channel_dim: int = -1
    attention_channel_dim: int = -1
    attention_output_channel_dim: int = -1
    scale_cache: dict[str, Any] = field(default_factory=dict)
    sparse_activity_recorder: SparseActivityRecorder | None = None
    bitstream_slice_recorder: BitstreamSliceRecorder | None = None
    bitstream_execution_config: BitstreamExecutionConfig | None = None
    bitstream_runtime_targetable_module_families: dict[str, str] = field(
        default_factory=dict
    )


def resolve_bitstream_runtime_family_policy() -> dict[str, dict[str, str]]:
    return {
        family: dict(policy)
        for family, policy in _BITSTREAM_RUNTIME_FAMILY_POLICY.items()
    }


def recommended_bitstream_slice_targets(model_key: str) -> dict[str, str]:
    del model_key
    return {
        "primary": "layer_4.1.global_rep.0.pre_norm_mha.1.attn_scores",
        "control": "layer_4.1.global_rep.0.pre_norm_mha.1.attn_output",
    }


def _stable_module_seed(namespace: str) -> int:
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % _BITSTREAM_STATE_MODULUS


def _mix_integer_state(base: int, value: int) -> int:
    return (
        (int(base) * 1_000_003)
        + int(value)
        + 0x9E3779B9
    ) % _BITSTREAM_STATE_MODULUS


def _compose_stream_seed_state(
    runtime_cfg: BitstreamExecutionConfig,
    *,
    module_key: str,
    operand_role: str,
    call_index: int,
    coordinate: tuple[int, ...],
) -> tuple[int | None, int]:
    phase = _stable_module_seed(f"{module_key}:{operand_role}")
    phase = _mix_integer_state(phase, int(call_index))
    for value in coordinate:
        phase = _mix_integer_state(phase, int(value))
    seed = None if runtime_cfg.seed is None else int(runtime_cfg.seed) + phase
    return seed, phase


def _resolve_row_seed_states(
    runtime_cfg: BitstreamExecutionConfig,
    *,
    module_key: str,
    operand_role: str,
    call_index: int,
    coordinates: list[Any],
) -> tuple[tuple[int | None, int], ...]:
    return tuple(
        _resolve_stream_namespace(
            runtime_cfg,
            f"{module_key}:{operand_role}:{call_index}:{coordinate}",
        )
        for coordinate in coordinates
    )


def _register_bitstream_targetable_module_family(
    perturb_config: MLXPerturbationConfig | None,
    module_key: str,
    family: str,
) -> None:
    if perturb_config is None:
        return
    perturb_config.bitstream_runtime_targetable_module_families.setdefault(
        module_key,
        family,
    )


def _bitstream_execution_wants(
    perturb_config: MLXPerturbationConfig | None,
    module_key: str,
    *,
    family: str | None = None,
) -> BitstreamExecutionConfig | None:
    if family is not None:
        _register_bitstream_targetable_module_family(perturb_config, module_key, family)
    if perturb_config is None or not perturb_config.enabled or perturb_config.calibration_mode:
        return None
    runtime_cfg = perturb_config.bitstream_execution_config
    if runtime_cfg is None or not runtime_cfg.enabled:
        return None
    requested = runtime_cfg.target_module_keys
    if requested is not None and module_key not in requested:
        return None
    return runtime_cfg


def _tensor_to_capture_values(tensor: Any, *, limit: int) -> list[float]:
    if limit <= 0:
        return []
    if isinstance(tensor, np.ndarray):
        return [float(value) for value in tensor.reshape(-1)[:limit].tolist()]
    tolist = getattr(tensor, "tolist", None)
    if callable(tolist):
        payload = tolist()
        array = np.asarray(payload, dtype=float)
        return [float(value) for value in array.reshape(-1)[:limit].tolist()]
    return []


def _record_bitstream_tensor(
    perturb_config: MLXPerturbationConfig | None,
    module_key: str,
    tensor: Any,
    *,
    operand_role: str = "output",
    call_index: int = 0,
    row_context: dict[str, Any] | None = None,
    notes: str | None = None,
) -> None:
    if perturb_config is None or perturb_config.bitstream_slice_recorder is None:
        return
    perturb_config.bitstream_slice_recorder.record(
        module_key,
        tensor,
        operand_role=operand_role,
        call_index=call_index,
        row_context=row_context,
        notes=notes,
    )


def _capture_tensor_shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(tensor, "shape", ()))


def _project_broadcast_index(index: tuple[int, ...], shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    offset = len(index) - len(shape)
    projected: list[int] = []
    for dim, size in enumerate(shape):
        index_value = index[offset + dim] if offset + dim >= 0 else 0
        projected.append(0 if int(size) == 1 else int(index_value))
    return tuple(projected)


def _as_numpy_array(value: Any, *, dtype: Any = float) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return np.asarray(tolist(), dtype=dtype)
    return np.asarray(value, dtype=dtype)


def _is_mx_array(value: Any) -> bool:
    return mx is not None and type(value).__name__ == "array"


def _restore_backend_array(value: np.ndarray, template: Any):
    if isinstance(template, np.ndarray):
        return value.astype(template.dtype, copy=False)
    if mx is None:
        return value
    mx_module, _ = _ensure_mlx_modules()
    return mx_module.array(value)


def _broadcast_shape(lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...]) -> tuple[int, ...]:
    left = [1] * max(0, len(rhs_shape) - len(lhs_shape)) + [int(dim) for dim in lhs_shape]
    right = [1] * max(0, len(lhs_shape) - len(rhs_shape)) + [int(dim) for dim in rhs_shape]
    result: list[int] = []
    for lhs_dim, rhs_dim in zip(left, right):
        if lhs_dim == rhs_dim:
            result.append(lhs_dim)
        elif lhs_dim == 1:
            result.append(rhs_dim)
        elif rhs_dim == 1:
            result.append(lhs_dim)
        else:
            raise ValueError(f"Incompatible broadcast shapes: {lhs_shape} vs {rhs_shape}")
    return tuple(result)


def _resolve_stream_namespace(
    runtime_cfg: BitstreamExecutionConfig,
    namespace: str,
) -> tuple[int | None, int]:
    phase = _stable_module_seed(namespace)
    seed = None if runtime_cfg.seed is None else int(runtime_cfg.seed) + phase
    return seed, phase


def _materialize_stream_array(
    streams: Any,
    *,
    stream_length: int,
) -> np.ndarray:
    matrix = np.asarray(streams, dtype=np.uint8)
    if matrix.ndim == 1 and matrix.size == 0:
        return np.empty((0, int(stream_length)), dtype=np.uint8)
    if matrix.ndim != 2:
        matrix = np.reshape(matrix, (-1, int(stream_length)))
    return matrix


def _resolve_lane_constants(
    *,
    vector_count: int,
    stream_length: int,
    seed: int | None,
    phase: int,
    generator: str,
) -> tuple[list[int], list[int], list[int]]:
    resolved_generator = str(generator).strip().lower()
    lane_count = int(vector_count)
    if lane_count <= 0:
        return [], [], []
    lane_indices_i64 = np.arange(lane_count, dtype=np.int64)
    phase_offsets = (int(phase) + lane_indices_i64).astype(object).tolist()
    if resolved_generator == "deterministic_threshold":
        return [0] * lane_count, [0] * lane_count, phase_offsets
    if resolved_generator != "low_discrepancy":
        raise ValueError(
            f"Unsupported MLX-native bitstream generator for compiled path: {generator!r}"
        )
    resolved_stream_length = int(stream_length)
    if resolved_stream_length <= 1:
        return [0] * lane_count, [0] * lane_count, phase_offsets

    lane_indices_u64 = np.arange(lane_count, dtype=np.uint64)
    phase_values = (
        np.uint64(int(phase) % _UINT64_WRAP) + lane_indices_u64
    )
    if seed is None:
        seed_values = np.zeros((lane_count,), dtype=np.uint64)
    else:
        seed_values = (
            np.uint64(int(seed) % _UINT64_WRAP)
            + (lane_indices_u64 * np.uint64(7919))
        )
    signatures = _resolve_lane_signature_array(
        seed_values=seed_values,
        phase_values=phase_values,
        stream_length=resolved_stream_length,
    )
    offsets = (signatures % np.uint64(resolved_stream_length)).astype(np.int32)
    _, inverse_lookup = _lane_stride_lookup(resolved_stream_length)
    inverse_strides = inverse_lookup[offsets]
    return (
        offsets.astype(int).tolist(),
        inverse_strides.astype(int).tolist(),
        phase_offsets,
    )


def _resolve_lane_constants_for_rows(
    *,
    row_seed_states: tuple[tuple[int | None, int], ...],
    vector_count: int,
    stream_length: int,
    generator: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolved_generator = str(generator).strip().lower()
    row_count = len(row_seed_states)
    lane_count = int(vector_count)
    total_count = row_count * lane_count
    if row_count <= 0 or lane_count <= 0:
        empty = np.empty((0,), dtype=np.int32)
        return empty, empty, empty

    lane_indices_i64 = np.arange(lane_count, dtype=np.int64)
    phase_bases_i64 = np.asarray(
        [int(phase) for _, phase in row_seed_states],
        dtype=np.int64,
    )
    if resolved_generator == "deterministic_threshold":
        phase_offsets = (phase_bases_i64[:, None] + lane_indices_i64[None, :]).reshape(-1)
        zeros = np.zeros((total_count,), dtype=np.int32)
        return zeros, zeros, phase_offsets
    if resolved_generator != "low_discrepancy":
        raise ValueError(
            f"Unsupported MLX-native bitstream generator for compiled path: {generator!r}"
        )

    resolved_stream_length = int(stream_length)
    if resolved_stream_length <= 1:
        zeros = np.zeros((total_count,), dtype=np.int32)
        return zeros, zeros, np.empty((0,), dtype=np.int64)

    lane_indices_u64 = np.arange(lane_count, dtype=np.uint64)
    phase_bases_u64 = np.asarray(
        [int(phase) % _UINT64_WRAP for _, phase in row_seed_states],
        dtype=np.uint64,
    )
    phase_values = (phase_bases_u64[:, None] + lane_indices_u64[None, :]).reshape(-1)

    seed_bases_u64 = np.asarray(
        [0 if seed is None else int(seed) % _UINT64_WRAP for seed, _ in row_seed_states],
        dtype=np.uint64,
    )
    seed_values = np.broadcast_to(seed_bases_u64[:, None], (row_count, lane_count)).copy()
    seeded_row_mask = np.asarray(
        [seed is not None for seed, _ in row_seed_states],
        dtype=bool,
    )
    if bool(np.any(seeded_row_mask)):
        seed_values[seeded_row_mask, :] += lane_indices_u64[None, :] * np.uint64(7919)

    signatures = _resolve_lane_signature_array(
        seed_values=seed_values.reshape(-1),
        phase_values=phase_values,
        stream_length=resolved_stream_length,
    )
    offsets = (signatures % np.uint64(resolved_stream_length)).astype(np.int32)
    _, inverse_lookup = _lane_stride_lookup(resolved_stream_length)
    inverse_strides = inverse_lookup[offsets]
    return offsets, inverse_strides, np.empty((0,), dtype=np.int64)


def _resolve_matmul_tile_sizes(
    *,
    dot_length: int,
    lhs_rows: int,
    rhs_cols: int,
) -> tuple[int, int]:
    max_row_col_pairs = max(
        1,
        int(_BITSTREAM_MATMUL_TILE_BUDGET) // max(1, int(dot_length)),
    )
    row_tile = max(1, min(int(lhs_rows), int(math.sqrt(max_row_col_pairs)) or 1))
    col_tile = max(1, min(int(rhs_cols), max_row_col_pairs // row_tile))
    return row_tile, max(1, col_tile)


def _resolve_matmul_batch_tile_count(*, batch_count: int) -> int:
    raw_override = os.environ.get("FULLER_BITSTREAM_MATMUL_BATCH_TILE_COUNT")
    if raw_override:
        try:
            requested = int(raw_override)
        except ValueError:
            requested = int(_BITSTREAM_MATMUL_BATCH_TILE_COUNT)
        else:
            requested = max(1, requested)
    else:
        requested = int(_BITSTREAM_MATMUL_BATCH_TILE_COUNT)
    return max(1, min(int(batch_count), requested))


def _resolve_conv_out_channel_tile_size(
    *,
    dot_length: int,
    stream_length: int,
    out_channels_per_group: int,
) -> int:
    max_stream_safe_channels = max(
        1,
        int(_BITSTREAM_STREAM_ENCODE_TILE_BUDGET)
        // max(1, int(dot_length) * max(1, int(stream_length))),
    )
    return max(
        1,
        min(
            int(out_channels_per_group),
            int(_BITSTREAM_MATMUL_TILE_BUDGET) // max(1, int(dot_length)),
            max_stream_safe_channels,
        ),
    )


def _resolve_conv_pointwise_pixel_tile_size(
    *,
    dot_length: int,
    out_channel_tile_size: int,
) -> int:
    return max(
        1,
        int(_BITSTREAM_MATMUL_TILE_BUDGET)
        // max(1, int(dot_length) * max(1, int(out_channel_tile_size))),
    )


def _resolve_stream_feature_tile_size(
    *,
    stream_length: int,
    lhs_rows: int,
    rhs_cols: int,
    dot_length: int,
) -> int:
    per_feature_encoded_values = max(
        1,
        (int(lhs_rows) + int(rhs_cols)) * max(1, int(stream_length)),
    )
    return max(
        1,
        min(
            int(dot_length),
            int(_BITSTREAM_STREAM_ENCODE_TILE_BUDGET) // per_feature_encoded_values,
        ),
    )


def _should_flush_mx_accumulator(
    *,
    tile_index: int,
    total_tiles: int,
) -> bool:
    stride = max(1, int(_BITSTREAM_MX_ACCUMULATOR_EVAL_TILE_STRIDE))
    if tile_index + 1 >= max(1, int(total_tiles)):
        return True
    return ((tile_index + 1) % stride) == 0


def _should_emit_matmul_tile_progress(
    *,
    tile_index: int,
    total_tiles: int,
) -> bool:
    stride = max(1, int(_BITSTREAM_MATMUL_PROGRESS_TILE_STRIDE))
    if tile_index <= 0:
        return True
    if tile_index + 1 >= max(1, int(total_tiles)):
        return True
    return ((tile_index + 1) % stride) == 0


def _should_emit_matmul_row_band_progress(
    *,
    row_band_index: int,
    total_row_bands: int,
) -> bool:
    stride = max(1, int(_BITSTREAM_MATMUL_ROW_BAND_PROGRESS_STRIDE))
    if row_band_index <= 0:
        return True
    if row_band_index + 1 >= max(1, int(total_row_bands)):
        return True
    return ((row_band_index + 1) % stride) == 0


def _should_cache_conv_filter_streams(
    *,
    dot_length: int,
    stream_length: int,
    out_channels_per_group: int,
) -> bool:
    total_stream_values = (
        int(out_channels_per_group) * int(dot_length) * max(1, int(stream_length))
    )
    return total_stream_values <= int(_BITSTREAM_STREAM_ENCODE_TILE_BUDGET)


def _mx_stream_matrix_for_values(
    values,
    *,
    runtime_cfg: BitstreamExecutionConfig,
    namespace: str,
):
    mx_module, _ = _ensure_mlx_modules()
    stream_tensor = _mx_stream_tensor_for_matrix(
        mx_module.reshape(values, (1, -1)),
        runtime_cfg=runtime_cfg,
        row_namespaces=(namespace,),
    )
    return stream_tensor[0]


def _mx_stream_tensor_for_matrix(
    vectors,
    *,
    runtime_cfg: BitstreamExecutionConfig,
    row_namespaces: tuple[str, ...] | list[str] | None = None,
    row_seed_states: tuple[tuple[int | None, int], ...] | list[tuple[int | None, int]] | None = None,
):
    mx_module, _ = _ensure_mlx_modules()
    if int(vectors.ndim) != 2:
        raise ValueError("vectors must be rank-2 for MLX-native stream generation.")
    num_rows = int(vectors.shape[0])
    vector_length = int(vectors.shape[1])
    if row_seed_states is None:
        if row_namespaces is None:
            raise ValueError("row_namespaces or row_seed_states is required.")
        if len(row_namespaces) != num_rows:
            raise ValueError("row_namespaces must align with the number of vectors.")
        resolved_row_seed_states = tuple(
            _resolve_stream_namespace(runtime_cfg, namespace)
            for namespace in row_namespaces
        )
    else:
        if len(row_seed_states) != num_rows:
            raise ValueError("row_seed_states must align with the number of vectors.")
        resolved_row_seed_states = tuple(row_seed_states)
    stream_length = int(runtime_cfg.stream_length)
    master_offsets, master_inverse_strides, master_phase_offsets = (
        _resolve_lane_constants_for_rows(
            row_seed_states=resolved_row_seed_states,
            vector_count=vector_length,
            stream_length=stream_length,
            generator=runtime_cfg.generator,
        )
    )
    flat_values = mx_module.reshape(vectors, (-1,))
    if str(runtime_cfg.encoding_mode).strip().lower() == "bipolar":
        probabilities = mx_module.clip((flat_values + 1.0) / 2.0, 0.0, 1.0)
    else:
        probabilities = mx_module.clip(flat_values, 0.0, 1.0)
    cycles = mx_module.arange(stream_length, dtype=mx_module.int32)[None, :]
    if str(runtime_cfg.generator).strip().lower() == "low_discrepancy":
        target_ones = mx_module.round(probabilities * stream_length).astype(mx_module.int32)[:, None]
        target_ones = mx_module.minimum(
            mx_module.maximum(target_ones, mx_module.array(0, dtype=mx_module.int32)),
            mx_module.array(stream_length, dtype=mx_module.int32),
        )
        offsets_arr = mx_module.array(master_offsets, dtype=mx_module.int32)[:, None]
        inverse_arr = mx_module.array(master_inverse_strides, dtype=mx_module.int32)[:, None]
        relative = mx_module.remainder((cycles - offsets_arr) * inverse_arr, stream_length)
        streams = (relative < target_ones).astype(mx_module.uint8)
        return mx_module.reshape(streams, (num_rows, vector_length, stream_length))
    phase_arr = mx_module.array(
        np.remainder(master_phase_offsets, stream_length).astype(np.int32),
        dtype=mx_module.int32,
    )[:, None]
    thresholds = (mx_module.remainder(phase_arr + cycles, stream_length).astype(mx_module.float32) + 0.5) / float(
        stream_length
    )
    streams = (thresholds < probabilities[:, None]).astype(mx_module.uint8)
    return mx_module.reshape(streams, (num_rows, vector_length, stream_length))


def _decode_total_counts_mx(
    total_counts,
    *,
    stream_length: int,
    vector_length: int,
    encoding_mode: str,
    multiplier_mode: str,
):
    mx_module, _ = _ensure_mlx_modules()
    counts = total_counts.astype(mx_module.float32)
    resolved_encoding = str(encoding_mode).strip().lower()
    resolved_multiplier = str(multiplier_mode).strip().lower()
    if resolved_encoding == "bipolar" and resolved_multiplier == "xnor":
        return ((2.0 * counts) / float(stream_length)) - float(vector_length)
    if resolved_encoding == "unipolar" and resolved_multiplier == "and":
        return counts / float(stream_length)
    raise ValueError(
        "Unsupported encoding/multiplier combination: "
        f"{encoding_mode}/{multiplier_mode}. Expected bipolar/xnor or unipolar/and."
    )


def _mx_estimate_from_stream_matrices(
    lhs_streams,
    rhs_streams,
    *,
    runtime_cfg: BitstreamExecutionConfig,
):
    mx_module, _ = _ensure_mlx_modules()
    lhs_rows = int(lhs_streams.shape[-3])
    rhs_cols = int(rhs_streams.shape[-3])
    dot_length = int(lhs_streams.shape[-2])
    stream_length = int(lhs_streams.shape[-1])
    resolved_multiplier = str(runtime_cfg.multiplier_mode).strip().lower()
    dot_tile_size = _resolve_stream_feature_tile_size(
        stream_length=stream_length,
        lhs_rows=lhs_rows,
        rhs_cols=rhs_cols,
        dot_length=dot_length,
    )
    total_dot_tiles = max(1, math.ceil(dot_length / max(1, dot_tile_size)))
    total_counts = None
    for dot_tile_index, dot_start in enumerate(range(0, dot_length, dot_tile_size)):
        dot_stop = min(dot_length, dot_start + dot_tile_size)
        lhs_chunk = lhs_streams[..., dot_start:dot_stop, :]
        rhs_chunk = rhs_streams[..., dot_start:dot_stop, :]
        lhs_pairs = lhs_chunk[..., :, None, :, :]
        rhs_pairs = rhs_chunk[..., None, :, :, :]
        if resolved_multiplier == "xnor":
            chunk_counts = mx_module.sum(lhs_pairs == rhs_pairs, axis=(-1, -2))
        elif resolved_multiplier == "and":
            chunk_counts = mx_module.sum(lhs_pairs * rhs_pairs, axis=(-1, -2))
        else:
            raise ValueError(f"Unsupported multiplier mode: {runtime_cfg.multiplier_mode!r}")
        total_counts = chunk_counts if total_counts is None else total_counts + chunk_counts
        if _should_flush_mx_accumulator(
            tile_index=dot_tile_index,
            total_tiles=total_dot_tiles,
        ):
            mx_module.eval(total_counts)
    estimates = _decode_total_counts_mx(
        total_counts,
        stream_length=int(runtime_cfg.stream_length),
        vector_length=dot_length,
        encoding_mode=runtime_cfg.encoding_mode,
        multiplier_mode=runtime_cfg.multiplier_mode,
    )
    mx_module.eval(estimates)
    return estimates


def _mx_estimate_from_stream_vector_pairs(
    lhs_streams,
    rhs_streams,
    *,
    runtime_cfg: BitstreamExecutionConfig,
):
    mx_module, _ = _ensure_mlx_modules()
    if tuple(int(dim) for dim in lhs_streams.shape) != tuple(int(dim) for dim in rhs_streams.shape):
        raise ValueError("lhs_streams and rhs_streams must have identical shapes for pairwise estimation.")
    dot_length = int(lhs_streams.shape[-2])
    stream_length = int(lhs_streams.shape[-1])
    row_count = int(lhs_streams.shape[-3])
    resolved_multiplier = str(runtime_cfg.multiplier_mode).strip().lower()
    dot_tile_size = _resolve_stream_feature_tile_size(
        stream_length=stream_length,
        lhs_rows=row_count,
        rhs_cols=1,
        dot_length=dot_length,
    )
    total_dot_tiles = max(1, math.ceil(dot_length / max(1, dot_tile_size)))
    total_counts = None
    for dot_tile_index, dot_start in enumerate(range(0, dot_length, dot_tile_size)):
        dot_stop = min(dot_length, dot_start + dot_tile_size)
        lhs_chunk = lhs_streams[..., dot_start:dot_stop, :]
        rhs_chunk = rhs_streams[..., dot_start:dot_stop, :]
        if resolved_multiplier == "xnor":
            chunk_counts = mx_module.sum(lhs_chunk == rhs_chunk, axis=(-1, -2))
        elif resolved_multiplier == "and":
            chunk_counts = mx_module.sum(lhs_chunk * rhs_chunk, axis=(-1, -2))
        else:
            raise ValueError(f"Unsupported multiplier mode: {runtime_cfg.multiplier_mode!r}")
        total_counts = chunk_counts if total_counts is None else total_counts + chunk_counts
        if _should_flush_mx_accumulator(
            tile_index=dot_tile_index,
            total_tiles=total_dot_tiles,
        ):
            mx_module.eval(total_counts)
    estimates = _decode_total_counts_mx(
        total_counts,
        stream_length=int(runtime_cfg.stream_length),
        vector_length=dot_length,
        encoding_mode=runtime_cfg.encoding_mode,
        multiplier_mode=runtime_cfg.multiplier_mode,
    )
    mx_module.eval(estimates)
    return estimates


def _stream_array_for_values(
    values: np.ndarray,
    *,
    runtime_cfg: BitstreamExecutionConfig,
    namespace: str,
) -> np.ndarray:
    if _is_mx_array(values):
        return _mx_stream_matrix_for_values(
            values,
            runtime_cfg=runtime_cfg,
            namespace=namespace,
        )
    seed, phase = _resolve_stream_namespace(runtime_cfg, namespace)
    streams = generate_bitstreams(
        values.reshape(-1).tolist(),
        stream_length=int(runtime_cfg.stream_length),
        generator=runtime_cfg.generator,
        encoding_mode=runtime_cfg.encoding_mode,
        seed=seed,
        phase=phase,
    )
    return _materialize_stream_array(
        streams,
        stream_length=int(runtime_cfg.stream_length),
    )


def _decode_stream_counts(
    counts: np.ndarray,
    *,
    stream_length: int,
    encoding_mode: str,
) -> np.ndarray:
    ratio = counts.astype(float) / float(stream_length)
    if str(encoding_mode).strip().lower() == "bipolar":
        return (ratio * 2.0) - 1.0
    return ratio


def _decode_stream_counts_mx(
    counts,
    *,
    stream_length: int,
    encoding_mode: str,
):
    mx_module, _ = _ensure_mlx_modules()
    ratio = counts.astype(mx_module.float32) / float(stream_length)
    if str(encoding_mode).strip().lower() == "bipolar":
        return (ratio * 2.0) - 1.0
    return ratio


def _decode_bitstream_tensor(
    tensor: Any,
    *,
    runtime_cfg: BitstreamExecutionConfig,
    module_key: str,
    operand_role: str,
    call_index: int,
):
    if _is_mx_array(tensor):
        mx_module, _ = _ensure_mlx_modules()
        stream_matrix = _stream_array_for_values(
            mx_module.reshape(tensor, (-1,)),
            runtime_cfg=runtime_cfg,
            namespace=f"{module_key}:{operand_role}:{call_index}",
        )
        decoded = _decode_stream_counts_mx(
            mx_module.sum(stream_matrix, axis=-1),
            stream_length=int(runtime_cfg.stream_length),
            encoding_mode=runtime_cfg.encoding_mode,
        )
        return mx_module.reshape(decoded, tensor.shape)
    tensor_np = _as_numpy_array(tensor, dtype=float)
    stream_matrix = _stream_array_for_values(
        tensor_np.reshape(-1),
        runtime_cfg=runtime_cfg,
        namespace=f"{module_key}:{operand_role}:{call_index}",
    )
    decoded = _decode_stream_counts(
        np.sum(stream_matrix, axis=1),
        stream_length=int(runtime_cfg.stream_length),
        encoding_mode=runtime_cfg.encoding_mode,
    ).reshape(tensor_np.shape)
    return _restore_backend_array(decoded, tensor)


def _execute_bitstream_activation(
    tensor: Any,
    *,
    activation_kind: str,
    runtime_cfg: BitstreamExecutionConfig,
    module_key: str,
):
    call_index = runtime_cfg.next_call_index(module_key)
    if _is_mx_array(tensor):
        mx_module, _ = _ensure_mlx_modules()
        decoded = _decode_bitstream_tensor(
            tensor,
            runtime_cfg=runtime_cfg,
            module_key=module_key,
            operand_role="activation",
            call_index=call_index,
        )
        kind = str(activation_kind).strip().lower()
        if kind == "relu":
            return mx_module.maximum(decoded, 0.0)
        if kind in {"silu", "swish"}:
            return decoded / (1.0 + mx_module.exp(-decoded))
        raise ValueError(f"Unsupported activation: {activation_kind}")
    decoded = _as_numpy_array(
        _decode_bitstream_tensor(
            tensor,
            runtime_cfg=runtime_cfg,
            module_key=module_key,
            operand_role="activation",
            call_index=call_index,
        ),
        dtype=float,
    )
    kind = str(activation_kind).strip().lower()
    if kind == "relu":
        result = np.maximum(decoded, 0.0)
    elif kind in {"silu", "swish"}:
        result = decoded / (1.0 + np.exp(-decoded))
    else:
        raise ValueError(f"Unsupported activation: {activation_kind}")
    return _restore_backend_array(result, tensor)


def _execute_bitstream_layer_norm(
    tensor: Any,
    *,
    weight: Any,
    bias: Any,
    eps: float,
    runtime_cfg: BitstreamExecutionConfig,
    module_key: str,
):
    call_index = runtime_cfg.next_call_index(module_key)
    if _is_mx_array(tensor):
        mx_module, _ = _ensure_mlx_modules()
        decoded = _decode_bitstream_tensor(
            tensor,
            runtime_cfg=runtime_cfg,
            module_key=module_key,
            operand_role="layer_norm",
            call_index=call_index,
        )
        mean = mx_module.mean(decoded, axis=-1, keepdims=True)
        var = mx_module.var(decoded, axis=-1, keepdims=True)
        normalized = (decoded - mean) * mx_module.rsqrt(var + float(eps))
        return normalized * weight.astype(decoded.dtype) + bias.astype(decoded.dtype)
    decoded = _as_numpy_array(
        _decode_bitstream_tensor(
            tensor,
            runtime_cfg=runtime_cfg,
            module_key=module_key,
            operand_role="layer_norm",
            call_index=call_index,
        ),
        dtype=float,
    )
    mean = np.mean(decoded, axis=-1, keepdims=True)
    var = np.var(decoded, axis=-1, keepdims=True)
    normalized = (decoded - mean) / np.sqrt(var + float(eps))
    result = normalized * _as_numpy_array(weight, dtype=float) + _as_numpy_array(
        bias,
        dtype=float,
    )
    return _restore_backend_array(result, tensor)


def _execute_bitstream_batch_norm2d(
    tensor: Any,
    *,
    weight: Any,
    bias: Any,
    running_mean: Any,
    running_var: Any,
    eps: float,
    runtime_cfg: BitstreamExecutionConfig,
    module_key: str,
):
    call_index = runtime_cfg.next_call_index(module_key)
    if _is_mx_array(tensor):
        mx_module, _ = _ensure_mlx_modules()
        decoded = _decode_bitstream_tensor(
            tensor,
            runtime_cfg=runtime_cfg,
            module_key=module_key,
            operand_role="batch_norm",
            call_index=call_index,
        )
        shape = [1] * decoded.ndim
        shape[-1] = int(weight.shape[0])
        mean = running_mean.reshape(shape).astype(decoded.dtype)
        var = running_var.reshape(shape).astype(decoded.dtype)
        gamma = weight.reshape(shape).astype(decoded.dtype)
        beta = bias.reshape(shape).astype(decoded.dtype)
        return ((decoded - mean) * mx_module.rsqrt(var + float(eps))) * gamma + beta
    decoded = _as_numpy_array(
        _decode_bitstream_tensor(
            tensor,
            runtime_cfg=runtime_cfg,
            module_key=module_key,
            operand_role="batch_norm",
            call_index=call_index,
        ),
        dtype=float,
    )
    shape = [1] * decoded.ndim
    shape[-1] = int(_as_numpy_array(weight, dtype=float).shape[0])
    mean = _as_numpy_array(running_mean, dtype=float).reshape(shape)
    var = _as_numpy_array(running_var, dtype=float).reshape(shape)
    gamma = _as_numpy_array(weight, dtype=float).reshape(shape)
    beta = _as_numpy_array(bias, dtype=float).reshape(shape)
    result = ((decoded - mean) / np.sqrt(var + float(eps))) * gamma + beta
    return _restore_backend_array(result, tensor)


def _execute_bitstream_matmul_mx_batch_chunked(
    lhs: Any,
    rhs: Any,
    *,
    runtime_cfg: BitstreamExecutionConfig,
    module_key: str,
):
    mx_module, _ = _ensure_mlx_modules()
    batch_shape = _broadcast_shape(tuple(lhs.shape[:-2]), tuple(rhs.shape[:-2]))
    batch_count = int(np.prod(batch_shape)) if batch_shape else 1
    if batch_count <= 1:
        return None

    call_index = runtime_cfg.next_call_index(module_key)
    lhs_b = mx_module.broadcast_to(lhs, batch_shape + tuple(lhs.shape[-2:]))
    rhs_b = mx_module.broadcast_to(rhs, batch_shape + tuple(rhs.shape[-2:]))
    lhs_rows = int(lhs_b.shape[-2])
    dot_length = int(lhs_b.shape[-1])
    rhs_cols = int(rhs_b.shape[-1])
    row_tile_size, col_tile_size = _resolve_matmul_tile_sizes(
        dot_length=dot_length,
        lhs_rows=lhs_rows,
        rhs_cols=rhs_cols,
    )
    batch_tile_count = _resolve_matmul_batch_tile_count(batch_count=batch_count)
    lhs_batches = mx_module.reshape(lhs_b, (batch_count, lhs_rows, dot_length))
    rhs_rank = len(batch_shape) + 2
    rhs_batches = mx_module.reshape(
        mx_module.transpose(
            rhs_b,
            tuple(range(rhs_rank - 2)) + (rhs_rank - 1, rhs_rank - 2),
        ),
        (batch_count, rhs_cols, dot_length),
    )
    batch_indices = list(np.ndindex(batch_shape))
    total_matmul_tiles = max(
        1,
        batch_count
        * math.ceil(lhs_rows / max(1, row_tile_size))
        * math.ceil(rhs_cols / max(1, col_tile_size)),
    )
    total_row_bands = max(
        1,
        batch_count * math.ceil(lhs_rows / max(1, row_tile_size)),
    )
    matmul_tile_index = 0
    row_band_index = 0
    chunk_outputs: list[Any] = []

    for batch_start in range(0, batch_count, batch_tile_count):
        batch_stop = min(batch_count, batch_start + batch_tile_count)
        chunk_size = int(batch_stop - batch_start)
        chunk_indices = batch_indices[batch_start:batch_stop]
        lhs_source_indices = [
            _project_broadcast_index(batch_index, tuple(lhs.shape[:-2]))
            for batch_index in chunk_indices
        ]
        rhs_source_indices = [
            _project_broadcast_index(batch_index, tuple(rhs.shape[:-2]))
            for batch_index in chunk_indices
        ]
        rhs_stream_cache: dict[tuple[int, int], Any] = {}
        row_outputs: list[Any] = []
        for row_start in range(0, lhs_rows, row_tile_size):
            row_stop = min(lhs_rows, row_start + row_tile_size)
            lhs_vectors = lhs_batches[batch_start:batch_stop, row_start:row_stop, :]
            lhs_seed_states = _resolve_row_seed_states(
                runtime_cfg,
                module_key=module_key,
                operand_role="lhs",
                call_index=call_index,
                coordinates=[
                    source_index + (row_idx,)
                    for source_index in lhs_source_indices
                    for row_idx in range(row_start, row_stop)
                ],
            )
            lhs_streams = _mx_stream_tensor_for_matrix(
                mx_module.reshape(lhs_vectors, (-1, dot_length)),
                runtime_cfg=runtime_cfg,
                row_seed_states=lhs_seed_states,
            )
            lhs_streams = mx_module.reshape(
                lhs_streams,
                (
                    chunk_size,
                    row_stop - row_start,
                    dot_length,
                    int(runtime_cfg.stream_length),
                ),
            )
            col_outputs: list[Any] = []
            for col_start in range(0, rhs_cols, col_tile_size):
                col_stop = min(rhs_cols, col_start + col_tile_size)
                rhs_cache_key = (int(col_start), int(col_stop))
                rhs_streams = rhs_stream_cache.get(rhs_cache_key)
                if rhs_streams is None:
                    rhs_vectors = rhs_batches[batch_start:batch_stop, col_start:col_stop, :]
                    rhs_seed_states = _resolve_row_seed_states(
                        runtime_cfg,
                        module_key=module_key,
                        operand_role="rhs",
                        call_index=call_index,
                        coordinates=[
                            source_index + (col_idx,)
                            for source_index in rhs_source_indices
                            for col_idx in range(col_start, col_stop)
                        ],
                    )
                    rhs_streams = _mx_stream_tensor_for_matrix(
                        mx_module.reshape(rhs_vectors, (-1, dot_length)),
                        runtime_cfg=runtime_cfg,
                        row_seed_states=rhs_seed_states,
                    )
                    rhs_streams = mx_module.reshape(
                        rhs_streams,
                        (
                            chunk_size,
                            col_stop - col_start,
                            dot_length,
                            int(runtime_cfg.stream_length),
                        ),
                    )
                    if runtime_cfg.stream_reuse_policy == "operand_factored_module_call_reuse":
                        rhs_stream_cache[rhs_cache_key] = rhs_streams
                estimates = _mx_estimate_from_stream_matrices(
                    lhs_streams,
                    rhs_streams,
                    runtime_cfg=runtime_cfg,
                )
                col_outputs.append(estimates)
                tile_start = matmul_tile_index
                tile_stop = min(total_matmul_tiles, matmul_tile_index + chunk_size)
                if (
                    _should_emit_matmul_tile_progress(
                        tile_index=tile_start,
                        total_tiles=total_matmul_tiles,
                    )
                    or tile_stop >= total_matmul_tiles
                ):
                    runtime_cfg.emit_progress(
                        module_key=module_key,
                        stage="matmul_tile_complete",
                        call_index=call_index,
                        detail=(
                            f"tile_index={tile_start} tile_stop={tile_stop} "
                            f"total_tiles={total_matmul_tiles} "
                            f"batch_flat_start={batch_start} batch_flat_stop={batch_stop} "
                            f"row_start={row_start} row_stop={row_stop} "
                            f"col_start={col_start} col_stop={col_stop}"
                        ),
                    )
                matmul_tile_index = tile_stop
            row_outputs.append(
                col_outputs[0]
                if len(col_outputs) == 1
                else mx_module.concatenate(tuple(col_outputs), axis=-1)
            )
            row_band_start = row_band_index
            row_band_stop = min(total_row_bands, row_band_index + chunk_size)
            if (
                _should_emit_matmul_row_band_progress(
                    row_band_index=row_band_start,
                    total_row_bands=total_row_bands,
                )
                or row_band_stop >= total_row_bands
            ):
                runtime_cfg.emit_progress(
                    module_key=module_key,
                    stage="matmul_row_band_complete",
                    call_index=call_index,
                    detail=(
                        f"row_band_index={row_band_start} row_band_stop={row_band_stop} "
                        f"total_row_bands={total_row_bands} "
                        f"batch_flat_start={batch_start} batch_flat_stop={batch_stop} "
                        f"row_start={row_start} row_stop={row_stop}"
                    ),
                )
            row_band_index = row_band_stop
        chunk_output = (
            row_outputs[0]
            if len(row_outputs) == 1
            else mx_module.concatenate(tuple(row_outputs), axis=-2)
        )
        chunk_outputs.append(chunk_output)
        runtime_cfg.emit_progress(
            module_key=module_key,
            stage="matmul_batch_complete",
            call_index=call_index,
            detail=f"batch_flat_start={batch_start} batch_flat_stop={batch_stop}",
        )
    flat_output = (
        chunk_outputs[0]
        if len(chunk_outputs) == 1
        else mx_module.concatenate(tuple(chunk_outputs), axis=0)
    )
    return mx_module.reshape(flat_output, batch_shape + (lhs_rows, rhs_cols))


def _execute_bitstream_matmul(
    lhs: Any,
    rhs: Any,
    *,
    runtime_cfg: BitstreamExecutionConfig,
    module_key: str,
):
    if _is_mx_array(lhs) and _is_mx_array(rhs):
        chunked_output = _execute_bitstream_matmul_mx_batch_chunked(
            lhs,
            rhs,
            runtime_cfg=runtime_cfg,
            module_key=module_key,
        )
        if chunked_output is not None:
            return chunked_output
        mx_module, _ = _ensure_mlx_modules()
        call_index = runtime_cfg.next_call_index(module_key)
        batch_shape = _broadcast_shape(tuple(lhs.shape[:-2]), tuple(rhs.shape[:-2]))
        lhs_b = mx_module.broadcast_to(lhs, batch_shape + tuple(lhs.shape[-2:]))
        rhs_b = mx_module.broadcast_to(rhs, batch_shape + tuple(rhs.shape[-2:]))
        lhs_rows = int(lhs_b.shape[-2])
        dot_length = int(lhs_b.shape[-1])
        rhs_cols = int(rhs_b.shape[-1])
        row_tile_size, col_tile_size = _resolve_matmul_tile_sizes(
            dot_length=dot_length,
            lhs_rows=lhs_rows,
            rhs_cols=rhs_cols,
        )
        batch_count = int(np.prod(batch_shape)) if batch_shape else 1
        lhs_batches = mx_module.reshape(lhs_b, (batch_count, lhs_rows, dot_length))
        rhs_rank = len(batch_shape) + 2
        rhs_batches = mx_module.reshape(
            mx_module.transpose(
                rhs_b,
                tuple(range(rhs_rank - 2)) + (rhs_rank - 1, rhs_rank - 2),
            ),
            (batch_count, rhs_cols, dot_length),
        )
        batch_outputs: list[Any] = []
        batch_indices = list(np.ndindex(batch_shape)) if batch_shape else [()]
        lhs_cache: dict[tuple[Any, ...], Any] = {}
        rhs_cache: dict[tuple[Any, ...], Any] = {}
        total_matmul_tiles = max(
            1,
            batch_count
            * math.ceil(lhs_rows / max(1, row_tile_size))
            * math.ceil(rhs_cols / max(1, col_tile_size)),
        )
        total_row_bands = max(
            1,
            batch_count * math.ceil(lhs_rows / max(1, row_tile_size)),
        )
        matmul_tile_index = 0
        row_band_index = 0
        for batch_flat_index, batch_index in enumerate(batch_indices):
            lhs_source_index = _project_broadcast_index(batch_index, tuple(lhs.shape[:-2]))
            rhs_source_index = _project_broadcast_index(batch_index, tuple(rhs.shape[:-2]))
            row_outputs: list[Any] = []
            for row_start in range(0, lhs_rows, row_tile_size):
                row_stop = min(lhs_rows, row_start + row_tile_size)
                lhs_vectors = lhs_batches[batch_flat_index, row_start:row_stop, :]
                lhs_cache_key = (tuple(lhs_source_index), int(row_start), int(row_stop))
                lhs_streams = lhs_cache.get(lhs_cache_key)
                if lhs_streams is None:
                    lhs_seed_states = _resolve_row_seed_states(
                        runtime_cfg,
                        module_key=module_key,
                        operand_role="lhs",
                        call_index=call_index,
                        coordinates=[
                            lhs_source_index + (row_idx,)
                            for row_idx in range(row_start, row_stop)
                        ],
                    )
                    lhs_streams = _mx_stream_tensor_for_matrix(
                        lhs_vectors,
                        runtime_cfg=runtime_cfg,
                        row_seed_states=lhs_seed_states,
                    )
                    if runtime_cfg.stream_reuse_policy == "operand_factored_module_call_reuse":
                        lhs_cache[lhs_cache_key] = lhs_streams
                col_outputs: list[Any] = []
                for col_start in range(0, rhs_cols, col_tile_size):
                    col_stop = min(rhs_cols, col_start + col_tile_size)
                    rhs_vectors = rhs_batches[batch_flat_index, col_start:col_stop, :]
                    rhs_cache_key = (tuple(rhs_source_index), int(col_start), int(col_stop))
                    rhs_streams = rhs_cache.get(rhs_cache_key)
                    if rhs_streams is None:
                        rhs_seed_states = _resolve_row_seed_states(
                            runtime_cfg,
                            module_key=module_key,
                            operand_role="rhs",
                            call_index=call_index,
                            coordinates=[
                                rhs_source_index + (col_idx,)
                                for col_idx in range(col_start, col_stop)
                            ],
                        )
                        rhs_streams = _mx_stream_tensor_for_matrix(
                            rhs_vectors,
                            runtime_cfg=runtime_cfg,
                            row_seed_states=rhs_seed_states,
                        )
                        if runtime_cfg.stream_reuse_policy == "operand_factored_module_call_reuse":
                            rhs_cache[rhs_cache_key] = rhs_streams
                    estimates = _mx_estimate_from_stream_matrices(
                        mx_module.expand_dims(lhs_streams, axis=0),
                        mx_module.expand_dims(rhs_streams, axis=0),
                        runtime_cfg=runtime_cfg,
                    )[0]
                    col_outputs.append(estimates)
                    if _should_emit_matmul_tile_progress(
                        tile_index=matmul_tile_index,
                        total_tiles=total_matmul_tiles,
                    ):
                        runtime_cfg.emit_progress(
                            module_key=module_key,
                            stage="matmul_tile_complete",
                            call_index=call_index,
                            detail=(
                                f"tile_index={matmul_tile_index} "
                                f"total_tiles={total_matmul_tiles} "
                                f"batch_flat_index={batch_flat_index} "
                                f"row_start={row_start} row_stop={row_stop} "
                                f"col_start={col_start} col_stop={col_stop}"
                            ),
                        )
                    matmul_tile_index += 1
                row_outputs.append(
                    col_outputs[0]
                    if len(col_outputs) == 1
                    else mx_module.concatenate(tuple(col_outputs), axis=-1)
                )
                if _should_emit_matmul_row_band_progress(
                    row_band_index=row_band_index,
                    total_row_bands=total_row_bands,
                ):
                    runtime_cfg.emit_progress(
                        module_key=module_key,
                        stage="matmul_row_band_complete",
                        call_index=call_index,
                        detail=(
                            f"row_band_index={row_band_index} "
                            f"total_row_bands={total_row_bands} "
                            f"batch_flat_index={batch_flat_index} "
                            f"row_start={row_start} row_stop={row_stop}"
                        ),
                    )
                row_band_index += 1
            batch_output = (
                row_outputs[0]
                if len(row_outputs) == 1
                else mx_module.concatenate(tuple(row_outputs), axis=0)
            )
            batch_outputs.append(batch_output)
            runtime_cfg.emit_progress(
                module_key=module_key,
                stage="matmul_batch_complete",
                call_index=call_index,
                detail=f"batch_flat_index={batch_flat_index}",
            )
        if not batch_shape:
            return batch_outputs[0]
        stacked = mx_module.concatenate(
            tuple(mx_module.expand_dims(item, axis=0) for item in batch_outputs),
            axis=0,
        )
        return mx_module.reshape(stacked, batch_shape + (lhs_rows, rhs_cols))
    lhs_np = _as_numpy_array(lhs, dtype=float)
    rhs_np = _as_numpy_array(rhs, dtype=float)
    output = np.matmul(lhs_np, rhs_np)
    output[...] = 0.0
    call_index = runtime_cfg.next_call_index(module_key)
    batch_shape = np.broadcast_shapes(lhs_np.shape[:-2], rhs_np.shape[:-2])
    lhs_b = np.broadcast_to(lhs_np, batch_shape + lhs_np.shape[-2:])
    rhs_b = np.broadcast_to(rhs_np, batch_shape + rhs_np.shape[-2:])
    lhs_cache: dict[tuple[Any, ...], np.ndarray] = {}
    rhs_cache: dict[tuple[Any, ...], np.ndarray] = {}
    iter_indices = list(np.ndindex(batch_shape)) if batch_shape else [()]
    for batch_index in iter_indices:
        lhs_source_index = _project_broadcast_index(batch_index, lhs_np.shape[:-2])
        rhs_source_index = _project_broadcast_index(batch_index, rhs_np.shape[:-2])
        lhs_matrix = lhs_b[batch_index] if batch_shape else lhs_b
        rhs_matrix = rhs_b[batch_index] if batch_shape else rhs_b
        for row_idx in range(int(lhs_matrix.shape[0])):
            lhs_vector = np.asarray(lhs_matrix[row_idx], dtype=float)
            lhs_key = lhs_source_index + (row_idx,)
            if runtime_cfg.stream_reuse_policy == "operand_factored_module_call_reuse":
                lhs_streams = lhs_cache.get(lhs_key)
                if lhs_streams is None:
                    lhs_streams = _stream_array_for_values(
                        lhs_vector,
                        runtime_cfg=runtime_cfg,
                        namespace=f"{module_key}:lhs:{call_index}:{lhs_key}",
                    )
                    lhs_cache[lhs_key] = lhs_streams
            for col_idx in range(int(rhs_matrix.shape[1])):
                rhs_vector = np.asarray(rhs_matrix[:, col_idx], dtype=float)
                if runtime_cfg.stream_reuse_policy == "operand_factored_module_call_reuse":
                    rhs_key = rhs_source_index + (col_idx,)
                    rhs_streams = rhs_cache.get(rhs_key)
                    if rhs_streams is None:
                        rhs_streams = _stream_array_for_values(
                            rhs_vector,
                            runtime_cfg=runtime_cfg,
                            namespace=f"{module_key}:rhs:{call_index}:{rhs_key}",
                        )
                        rhs_cache[rhs_key] = rhs_streams
                    estimate = estimate_dot_product_value_from_stream_arrays(
                        lhs_streams,
                        rhs_streams,
                        stream_length=int(runtime_cfg.stream_length),
                        encoding_mode=runtime_cfg.encoding_mode,
                        multiplier_mode=runtime_cfg.multiplier_mode,
                    )
                else:
                    seed, phase = _resolve_stream_namespace(
                        runtime_cfg,
                        f"{module_key}:matmul:{call_index}:{batch_index}:{row_idx}:{col_idx}",
                    )
                    estimate = estimate_dot_product_value(
                        lhs_vector.tolist(),
                        rhs_vector.tolist(),
                        stream_length=int(runtime_cfg.stream_length),
                        generator=runtime_cfg.generator,
                        encoding_mode=runtime_cfg.encoding_mode,
                        multiplier_mode=runtime_cfg.multiplier_mode,
                        seed=seed,
                        phase=phase,
                    )
                target = batch_index + (row_idx, col_idx) if batch_shape else (row_idx, col_idx)
                output[target] = float(estimate)
    return _restore_backend_array(output, lhs)


def _resolve_pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _execute_bitstream_conv2d_nhwc(
    x: Any,
    *,
    weight: Any,
    bias: Any,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    dilation: int | tuple[int, int],
    groups: int,
    runtime_cfg: BitstreamExecutionConfig,
    module_key: str,
):
    if _is_mx_array(x) and _is_mx_array(weight):
        mx_module, _ = _ensure_mlx_modules()
        stride_h, stride_w = _resolve_pair(stride)
        pad_h, pad_w = _resolve_pair(padding)
        dil_h, dil_w = _resolve_pair(dilation)
        batch_size, in_h, in_w, in_channels = (int(dim) for dim in x.shape)
        out_channels, kernel_h, kernel_w, group_in_channels = (int(dim) for dim in weight.shape)
        resolved_groups = max(1, int(groups))
        out_channels_per_group = out_channels // resolved_groups
        padded = mx_module.pad(
            x,
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        )
        eff_kernel_h = (kernel_h - 1) * dil_h + 1
        eff_kernel_w = (kernel_w - 1) * dil_w + 1
        out_h = ((int(padded.shape[1]) - eff_kernel_h) // stride_h) + 1
        out_w = ((int(padded.shape[2]) - eff_kernel_w) // stride_w) + 1
        call_index = runtime_cfg.next_call_index(module_key)
        dot_length = kernel_h * kernel_w * group_in_channels
        out_channel_tile_size = _resolve_conv_out_channel_tile_size(
            dot_length=dot_length,
            stream_length=int(runtime_cfg.stream_length),
            out_channels_per_group=out_channels_per_group,
        )
        cache_filter_streams = _should_cache_conv_filter_streams(
            dot_length=dot_length,
            stream_length=int(runtime_cfg.stream_length),
            out_channels_per_group=out_channels_per_group,
        )
        out_y_tile_size = max(
            1,
            min(out_h, int(_BITSTREAM_CONV_SPATIAL_TILE_BUDGET) // max(1, out_w)),
        )
        weight_groups = mx_module.reshape(
            weight,
            (resolved_groups, out_channels_per_group, dot_length),
        )
        pointwise_fast_path = (
            kernel_h == 1
            and kernel_w == 1
            and dil_h == 1
            and dil_w == 1
            and resolved_groups == 1
        )
        pointwise_pixel_tile_size = (
            _resolve_conv_pointwise_pixel_tile_size(
                dot_length=dot_length,
                out_channel_tile_size=out_channel_tile_size,
            )
            if pointwise_fast_path
            else 1
        )
        batch_outputs: list[Any] = []
        filter_stream_cache: dict[tuple[Any, ...], Any] = {}
        for batch_index in range(batch_size):
            row_outputs: list[Any] = []
            for out_y_start in range(0, out_h, out_y_tile_size):
                out_y_stop = min(out_h, out_y_start + out_y_tile_size)
                tile_rows: list[Any] = []
                for out_y in range(out_y_start, out_y_stop):
                    start_y = out_y * stride_h
                    stop_y = start_y + eff_kernel_h
                    if pointwise_fast_path:
                        row_vectors = padded[
                            batch_index,
                            start_y,
                            0 : (out_w * stride_w) : stride_w,
                            :,
                        ]
                        row_tiles: list[Any] = []
                        for out_x_start in range(0, out_w, pointwise_pixel_tile_size):
                            out_x_stop = min(out_w, out_x_start + pointwise_pixel_tile_size)
                            patch_vectors = row_vectors[out_x_start:out_x_stop, :]
                            patch_streams = _mx_stream_tensor_for_matrix(
                                patch_vectors,
                                runtime_cfg=runtime_cfg,
                                row_seed_states=_resolve_row_seed_states(
                                    runtime_cfg,
                                    module_key=module_key,
                                    operand_role="patch",
                                    call_index=call_index,
                                    coordinates=[
                                        (batch_index, out_y, out_x, 0)
                                        for out_x in range(out_x_start, out_x_stop)
                                    ],
                                ),
                            )
                            channel_outputs: list[Any] = []
                            for out_channel_start in range(
                                0,
                                out_channels_per_group,
                                out_channel_tile_size,
                            ):
                                out_channel_stop = min(
                                    out_channels_per_group,
                                    out_channel_start + out_channel_tile_size,
                                )
                                cache_key = (
                                    "pointwise",
                                    out_channel_start,
                                    out_channel_stop,
                                )
                                filter_streams = (
                                    filter_stream_cache.get(cache_key)
                                    if cache_filter_streams
                                    else None
                                )
                                if filter_streams is None:
                                    filter_streams = _mx_stream_tensor_for_matrix(
                                        weight_groups[
                                            0,
                                            out_channel_start:out_channel_stop,
                                            :,
                                        ],
                                        runtime_cfg=runtime_cfg,
                                        row_seed_states=_resolve_row_seed_states(
                                            runtime_cfg,
                                            module_key=module_key,
                                            operand_role="filter",
                                            call_index=call_index,
                                            coordinates=list(
                                                range(out_channel_start, out_channel_stop)
                                            ),
                                        ),
                                    )
                                    if cache_filter_streams:
                                        filter_stream_cache[cache_key] = filter_streams
                                try:
                                    estimates = _mx_estimate_from_stream_matrices(
                                        mx_module.expand_dims(patch_streams, axis=0),
                                        mx_module.expand_dims(filter_streams, axis=0),
                                        runtime_cfg=runtime_cfg,
                                    )[0]
                                except RuntimeError as exc:
                                    if "Resource limit" not in str(exc):
                                        raise
                                    raise RuntimeError(
                                        f"{exc} module_key={module_key} "
                                        f"x_shape={tuple(int(dim) for dim in x.shape)} "
                                        f"weight_shape={tuple(int(dim) for dim in weight.shape)} "
                                        f"batch_index={batch_index} out_y={out_y} "
                                        f"out_x_start={out_x_start} out_x_stop={out_x_stop} "
                                        f"out_channel_start={out_channel_start} "
                                        f"out_channel_stop={out_channel_stop} "
                                        f"dot_length={dot_length} "
                                        f"pointwise_fast_path=true "
                                        f"pointwise_pixel_tile_size={pointwise_pixel_tile_size} "
                                        f"cache_filter_streams={cache_filter_streams}"
                                    ) from exc
                                channel_outputs.append(estimates)
                                runtime_cfg.emit_progress(
                                    module_key=module_key,
                                    stage="conv2d_pointwise_tile_complete",
                                    call_index=call_index,
                                    detail=(
                                        f"batch_index={batch_index} out_y={out_y} "
                                        f"out_x_start={out_x_start} out_x_stop={out_x_stop} "
                                        f"out_channel_start={out_channel_start} "
                                        f"out_channel_stop={out_channel_stop}"
                                    ),
                                )
                            row_tiles.append(
                                channel_outputs[0]
                                if len(channel_outputs) == 1
                                else mx_module.concatenate(tuple(channel_outputs), axis=-1)
                            )
                        row_output = (
                            row_tiles[0]
                            if len(row_tiles) == 1
                            else mx_module.concatenate(tuple(row_tiles), axis=0)
                        )
                        mx_module.eval(row_output)
                        runtime_cfg.emit_progress(
                            module_key=module_key,
                            stage="conv2d_tile_row_complete",
                            call_index=call_index,
                            detail=(
                                f"batch_index={batch_index} out_y={out_y} "
                                f"pointwise_fast_path=true"
                            ),
                        )
                        tile_rows.append(mx_module.expand_dims(row_output, axis=0))
                        continue
                    pixel_outputs: list[Any] = []
                    for out_x in range(out_w):
                        start_x = out_x * stride_w
                        stop_x = start_x + eff_kernel_w
                        if out_channels_per_group == 1 and group_in_channels == 1:
                            patch = padded[
                                batch_index : batch_index + 1,
                                start_y:stop_y:dil_h,
                                start_x:stop_x:dil_w,
                                :,
                            ]
                            patch_vectors = mx_module.transpose(
                                mx_module.reshape(
                                    patch,
                                    (1, kernel_h, kernel_w, resolved_groups),
                                )[0],
                                (2, 0, 1),
                            )
                            patch_vectors = mx_module.reshape(
                                patch_vectors,
                                (resolved_groups, dot_length),
                            )
                            patch_streams = _mx_stream_tensor_for_matrix(
                                patch_vectors,
                                runtime_cfg=runtime_cfg,
                                row_seed_states=_resolve_row_seed_states(
                                    runtime_cfg,
                                    module_key=module_key,
                                    operand_role="patch",
                                    call_index=call_index,
                                    coordinates=[
                                        (batch_index, out_y, out_x, group_index)
                                        for group_index in range(resolved_groups)
                                    ],
                                ),
                            )
                            filter_streams = filter_stream_cache.get(("depthwise",))
                            if filter_streams is None:
                                filter_streams = _mx_stream_tensor_for_matrix(
                                    weight_groups[:, 0, :],
                                    runtime_cfg=runtime_cfg,
                                    row_seed_states=_resolve_row_seed_states(
                                        runtime_cfg,
                                        module_key=module_key,
                                        operand_role="filter",
                                        call_index=call_index,
                                        coordinates=list(range(resolved_groups)),
                                    ),
                                )
                                if cache_filter_streams:
                                    filter_stream_cache[("depthwise",)] = filter_streams
                            try:
                                pixel_output = _mx_estimate_from_stream_vector_pairs(
                                    patch_streams,
                                    filter_streams,
                                    runtime_cfg=runtime_cfg,
                                )
                            except RuntimeError as exc:
                                if "Resource limit" not in str(exc):
                                    raise
                                raise RuntimeError(
                                    f"{exc} module_key={module_key} "
                                    f"x_shape={tuple(int(dim) for dim in x.shape)} "
                                    f"weight_shape={tuple(int(dim) for dim in weight.shape)} "
                                    f"batch_index={batch_index} out_y={out_y} out_x={out_x} "
                                    f"resolved_groups={resolved_groups} "
                                    f"out_channels_per_group={out_channels_per_group} "
                                    f"dot_length={dot_length} depthwise_fast_path=true"
                                ) from exc
                            mx_module.eval(pixel_output)
                            runtime_cfg.emit_progress(
                                module_key=module_key,
                                stage="conv2d_pixel_complete",
                                call_index=call_index,
                                detail=(
                                    f"batch_index={batch_index} out_y={out_y} out_x={out_x}"
                                ),
                            )
                            pixel_outputs.append(mx_module.expand_dims(pixel_output, axis=0))
                            continue
                        group_outputs: list[Any] = []
                        for group_index in range(resolved_groups):
                            channel_start = group_index * group_in_channels
                            channel_stop = channel_start + group_in_channels
                            patch = padded[
                                batch_index : batch_index + 1,
                                start_y:stop_y:dil_h,
                                start_x:stop_x:dil_w,
                                channel_start:channel_stop,
                            ]
                            patch_streams = _mx_stream_tensor_for_matrix(
                                mx_module.reshape(patch, (1, dot_length)),
                                runtime_cfg=runtime_cfg,
                                row_seed_states=_resolve_row_seed_states(
                                    runtime_cfg,
                                    module_key=module_key,
                                    operand_role="patch",
                                    call_index=call_index,
                                    coordinates=[(batch_index, out_y, out_x, group_index)],
                                ),
                            )
                            channel_outputs: list[Any] = []
                            for out_channel_start in range(
                                0,
                                out_channels_per_group,
                                out_channel_tile_size,
                            ):
                                out_channel_stop = min(
                                    out_channels_per_group,
                                    out_channel_start + out_channel_tile_size,
                                )
                                cache_key = (
                                    group_index,
                                    out_channel_start,
                                    out_channel_stop,
                                )
                                filter_streams = (
                                    filter_stream_cache.get(cache_key)
                                    if cache_filter_streams
                                    else None
                                )
                                if filter_streams is None:
                                    filter_streams = _mx_stream_tensor_for_matrix(
                                        weight_groups[
                                            group_index,
                                            out_channel_start:out_channel_stop,
                                            :,
                                        ],
                                        runtime_cfg=runtime_cfg,
                                        row_seed_states=_resolve_row_seed_states(
                                            runtime_cfg,
                                            module_key=module_key,
                                            operand_role="filter",
                                            call_index=call_index,
                                            coordinates=[
                                                (group_index * out_channels_per_group)
                                                + out_index
                                                for out_index in range(
                                                    out_channel_start,
                                                    out_channel_stop,
                                                )
                                            ],
                                        ),
                                    )
                                    if cache_filter_streams:
                                        filter_stream_cache[cache_key] = filter_streams
                                try:
                                    estimates = _mx_estimate_from_stream_matrices(
                                        mx_module.expand_dims(patch_streams, axis=0),
                                        mx_module.expand_dims(filter_streams, axis=0),
                                        runtime_cfg=runtime_cfg,
                                    )[0][0]
                                except RuntimeError as exc:
                                    if "Resource limit" not in str(exc):
                                        raise
                                    raise RuntimeError(
                                        f"{exc} module_key={module_key} "
                                        f"x_shape={tuple(int(dim) for dim in x.shape)} "
                                        f"weight_shape={tuple(int(dim) for dim in weight.shape)} "
                                        f"batch_index={batch_index} out_y={out_y} out_x={out_x} "
                                        f"group_index={group_index} out_channel_start={out_channel_start} "
                                        f"out_channel_stop={out_channel_stop} "
                                        f"resolved_groups={resolved_groups} "
                                        f"out_channels_per_group={out_channels_per_group} "
                                        f"dot_length={dot_length} "
                                        f"out_channel_tile_size={out_channel_tile_size} "
                                        f"cache_filter_streams={cache_filter_streams}"
                                    ) from exc
                                channel_outputs.append(estimates)
                            group_outputs.append(
                                channel_outputs[0]
                                if len(channel_outputs) == 1
                                else mx_module.concatenate(tuple(channel_outputs), axis=-1)
                            )
                        pixel_output = mx_module.concatenate(
                            tuple(group_outputs),
                            axis=-1,
                        )
                        mx_module.eval(pixel_output)
                        runtime_cfg.emit_progress(
                            module_key=module_key,
                            stage="conv2d_pixel_complete",
                            call_index=call_index,
                            detail=(
                                f"batch_index={batch_index} out_y={out_y} out_x={out_x}"
                            ),
                        )
                        pixel_outputs.append(mx_module.expand_dims(pixel_output, axis=0))
                    tile_row = mx_module.concatenate(tuple(pixel_outputs), axis=0)
                    mx_module.eval(tile_row)
                    runtime_cfg.emit_progress(
                        module_key=module_key,
                        stage="conv2d_tile_row_complete",
                        call_index=call_index,
                        detail=(
                            f"batch_index={batch_index} out_y_start={out_y_start} "
                            f"out_y_stop={out_y_stop}"
                        ),
                    )
                    tile_rows.append(mx_module.expand_dims(tile_row, axis=0))
                row_output = mx_module.concatenate(tuple(tile_rows), axis=0)
                mx_module.eval(row_output)
                runtime_cfg.emit_progress(
                    module_key=module_key,
                    stage="conv2d_row_band_complete",
                    call_index=call_index,
                    detail=(
                        f"batch_index={batch_index} out_y_start={out_y_start} "
                        f"out_y_stop={out_y_stop}"
                    ),
                )
                row_outputs.append(row_output)
            batch_output = (
                row_outputs[0]
                if len(row_outputs) == 1
                else mx_module.concatenate(tuple(row_outputs), axis=0)
            )
            mx_module.eval(batch_output)
            batch_outputs.append(mx_module.expand_dims(batch_output, axis=0))
        output = mx_module.concatenate(tuple(batch_outputs), axis=0)
        mx_module.eval(output)
        if bias is not None:
            output = output + mx_module.reshape(bias, (1, 1, 1, out_channels))
            mx_module.eval(output)
        return output
    x_np = _as_numpy_array(x, dtype=float)
    weight_np = _as_numpy_array(weight, dtype=float)
    bias_np = None if bias is None else _as_numpy_array(bias, dtype=float)
    stride_h, stride_w = _resolve_pair(stride)
    pad_h, pad_w = _resolve_pair(padding)
    dil_h, dil_w = _resolve_pair(dilation)
    batch_size, in_h, in_w, in_channels = x_np.shape
    out_channels, kernel_h, kernel_w, group_in_channels = weight_np.shape
    resolved_groups = max(1, int(groups))
    out_channels_per_group = out_channels // resolved_groups
    padded = np.pad(
        x_np,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode="constant",
    )
    eff_kernel_h = (kernel_h - 1) * dil_h + 1
    eff_kernel_w = (kernel_w - 1) * dil_w + 1
    out_h = ((padded.shape[1] - eff_kernel_h) // stride_h) + 1
    out_w = ((padded.shape[2] - eff_kernel_w) // stride_w) + 1
    output = np.zeros((batch_size, out_h, out_w, out_channels), dtype=float)
    call_index = runtime_cfg.next_call_index(module_key)
    filter_cache: dict[int, np.ndarray] = {}
    patch_cache: dict[tuple[int, int, int, int], np.ndarray] = {}
    filter_vectors = [weight_np[out_channel].reshape(-1) for out_channel in range(out_channels)]
    for out_channel in range(out_channels):
        if runtime_cfg.stream_reuse_policy == "operand_factored_module_call_reuse":
            filter_cache[out_channel] = _stream_array_for_values(
                filter_vectors[out_channel],
                runtime_cfg=runtime_cfg,
                namespace=f"{module_key}:filter:{call_index}:{out_channel}",
            )
    for batch_index in range(batch_size):
        for out_y in range(out_h):
            start_y = out_y * stride_h
            stop_y = start_y + eff_kernel_h
            for out_x in range(out_w):
                start_x = out_x * stride_w
                stop_x = start_x + eff_kernel_w
                for out_channel in range(out_channels):
                    group_index = out_channel // out_channels_per_group
                    channel_start = group_index * group_in_channels
                    channel_stop = channel_start + group_in_channels
                    patch = padded[
                        batch_index,
                        start_y:stop_y:dil_h,
                        start_x:stop_x:dil_w,
                        channel_start:channel_stop,
                    ].reshape(-1)
                    if runtime_cfg.stream_reuse_policy == "operand_factored_module_call_reuse":
                        patch_key = (batch_index, out_y, out_x, group_index)
                        patch_streams = patch_cache.get(patch_key)
                        if patch_streams is None:
                            patch_streams = _stream_array_for_values(
                                patch,
                                runtime_cfg=runtime_cfg,
                                namespace=f"{module_key}:patch:{call_index}:{patch_key}",
                            )
                            patch_cache[patch_key] = patch_streams
                        estimate = estimate_dot_product_value_from_stream_arrays(
                            patch_streams,
                            filter_cache[out_channel],
                            stream_length=int(runtime_cfg.stream_length),
                            encoding_mode=runtime_cfg.encoding_mode,
                            multiplier_mode=runtime_cfg.multiplier_mode,
                        )
                    else:
                        seed, phase = _resolve_stream_namespace(
                            runtime_cfg,
                            f"{module_key}:conv:{call_index}:{batch_index}:{out_y}:{out_x}:{out_channel}",
                        )
                        estimate = estimate_dot_product_value(
                            patch.tolist(),
                            filter_vectors[out_channel].tolist(),
                            stream_length=int(runtime_cfg.stream_length),
                            generator=runtime_cfg.generator,
                            encoding_mode=runtime_cfg.encoding_mode,
                            multiplier_mode=runtime_cfg.multiplier_mode,
                            seed=seed,
                            phase=phase,
                        )
                    output[batch_index, out_y, out_x, out_channel] = float(estimate)
    if bias_np is not None:
        output += bias_np.reshape((1, 1, 1, -1))
    return _restore_backend_array(output, x)


def capture_mlx_bitstream_slices(
    model,
    inputs: Any,
    *,
    recorder: BitstreamSliceRecorder,
    row_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if recorder is None:
        return []
    recorder.reset()
    perturb_config = getattr(model, "perturb_config", None)
    previous = None if perturb_config is None else perturb_config.bitstream_slice_recorder
    if perturb_config is not None:
        perturb_config.bitstream_slice_recorder = recorder
    try:
        outputs = model(inputs)
        if mx is not None:
            try:
                mx_module, _ = _ensure_mlx_modules()
                mx_module.eval(outputs)
            except Exception:
                pass
    finally:
        if perturb_config is not None:
            perturb_config.bitstream_slice_recorder = previous
    rows = recorder.rows()
    if not row_context:
        return rows
    merged_rows: list[dict[str, Any]] = []
    for row in rows:
        merged = dict(row)
        for key, value in dict(row_context).items():
            if merged.get(key) in {None, ""}:
                merged[key] = value
        merged_rows.append(merged)
    return merged_rows


def _resize_image_nhwc(x: Any, target_h: int, target_w: int) -> Any:
    _, nn_module = _ensure_mlx_modules()
    if int(x.shape[1]) == int(target_h) and int(x.shape[2]) == int(target_w):
        return x
    scale_h = float(target_h) / float(x.shape[1])
    scale_w = float(target_w) / float(x.shape[2])
    upsample = nn_module.Upsample(
        scale_factor=(scale_h, scale_w),
        mode="linear",
        align_corners=False,
    )
    return upsample(x)


def _update_scale_cache(
    tensor: Any,
    perturb_config: MLXPerturbationConfig,
    module_key: str,
    channel_dim: int | None,
) -> None:
    mx_module, _ = _ensure_mlx_modules()
    compute_symmetric_scale = _load_mlx_photonic_perturb_module().compute_symmetric_scale

    scale = compute_symmetric_scale(
        tensor,
        perturb_config.bits,
        channel_dim=channel_dim,
    )
    if scale is None:
        return
    previous = perturb_config.scale_cache.get(module_key)
    if previous is None:
        mx_module.eval(scale)
        perturb_config.scale_cache[module_key] = scale
        return
    if previous.shape != scale.shape:
        updated_scale = mx_module.maximum(
            mx_module.max(previous.reshape((-1,))),
            mx_module.max(scale.reshape((-1,))),
        )
        mx_module.eval(updated_scale)
        perturb_config.scale_cache[module_key] = updated_scale
        return
    updated_scale = mx_module.maximum(previous, scale)
    # Materialize the running scale so calibration does not retain a lazy graph
    # chain across thousands of batches.
    mx_module.eval(updated_scale)
    perturb_config.scale_cache[module_key] = updated_scale


def _resolve_scale_override(
    perturb_config: MLXPerturbationConfig,
    module_key: str,
):
    if not perturb_config.use_calibrated_scale:
        return None
    scale_value = perturb_config.scale_cache.get(module_key)
    if scale_value is None:
        if perturb_config.strict_calibrated_scale:
            raise RuntimeError(f"Missing calibrated scale for {module_key}")
        return None
    return scale_value


def _sparse_enabled_for_module(
    perturb_config: MLXPerturbationConfig | None,
    module_key: str,
) -> bool:
    if perturb_config is None:
        return False
    if perturb_config.calibration_mode or not perturb_config.enabled:
        return False
    if not perturb_config.sparse_enabled:
        return False
    if (
        perturb_config.sparse_target_module_keys is not None
        and module_key not in perturb_config.sparse_target_module_keys
    ):
        return False
    if (
        perturb_config.sparse_protected_layers is not None
        and module_key in perturb_config.sparse_protected_layers
    ):
        return False
    return True


def _apply_sparse_gate_to_tensor(
    tensor: Any,
    perturb_config: MLXPerturbationConfig | None,
    channel_dim: int | None,
    module_key: str,
):
    """Apply sparse gating without re-running the full quant/noise pipeline."""
    if not _sparse_enabled_for_module(perturb_config, module_key):
        return tensor
    sparse_module = _load_mlx_photonic_perturb_module()
    if perturb_config.sparse_activity_recorder is not None:
        gated, sparse_stats = sparse_module.apply_sparse_gating(
            tensor,
            perturb_config.sparse_active_fraction,
            channel_dim,
            sparse_tau_global=perturb_config.sparse_tau_global,
            return_stats=True,
        )
        if sparse_stats is not None:
            perturb_config.sparse_activity_recorder.record(module_key, sparse_stats)
        return gated
    return sparse_module.apply_sparse_gating(
        tensor,
        perturb_config.sparse_active_fraction,
        channel_dim,
        sparse_tau_global=perturb_config.sparse_tau_global,
    )


def _apply_perturb_to_tensor(
    tensor: Any,
    perturb_config: MLXPerturbationConfig | None,
    channel_dim: int | None,
    module_key: str,
):
    if perturb_config is None:
        return tensor
    if perturb_config.bitstream_slice_recorder is not None:
        if mx is not None:
            try:
                mx_module, _ = _ensure_mlx_modules()
                mx_module.eval(tensor)
            except Exception:
                pass
        _record_bitstream_tensor(
            perturb_config,
            module_key,
            tensor,
            operand_role="output",
        )
    if perturb_config.calibration_mode:
        _update_scale_cache(tensor, perturb_config, module_key, channel_dim)
        return tensor
    if not perturb_config.enabled:
        return tensor

    scale_override = _resolve_scale_override(perturb_config, module_key)
    layer_sparse_enabled = _sparse_enabled_for_module(perturb_config, module_key)

    sparse_activity_callback = None
    if layer_sparse_enabled and perturb_config.sparse_activity_recorder is not None:
        sparse_activity_callback = lambda stats: perturb_config.sparse_activity_recorder.record(  # noqa: E731
            module_key,
            stats,
        )

    det_k = perturb_config.det_k_global
    if perturb_config.det_enabled and perturb_config.det_k_by_layer is not None:
        det_k = perturb_config.det_k_by_layer.get(module_key, det_k)
    det_prefix_error = perturb_config.det_prefix_error_mean
    if (
        perturb_config.det_enabled
        and perturb_config.det_prefix_error_by_layer is not None
    ):
        det_prefix_error = perturb_config.det_prefix_error_by_layer.get(
            module_key,
            det_prefix_error,
        )

    return _load_mlx_photonic_perturb_module().apply_perturb(
        tensor,
        perturb_config.bits,
        perturb_config.gaussian_noise_std,
        perturb_config.crosstalk_alpha,
        channel_dim,
        scale_override=scale_override,
        drift_lsb=perturb_config.drift_lsb,
        noise_correlation=perturb_config.noise_correlation,
        burst_error_prob=perturb_config.burst_error_prob,
        burst_error_scale_lsb=perturb_config.burst_error_scale_lsb,
        burst_span=perturb_config.burst_span,
        det_enabled=perturb_config.det_enabled,
        det_k=det_k,
        det_bsl_max=perturb_config.det_bsl_max,
        det_mode=perturb_config.det_mode,
        det_prefix_error_mean=det_prefix_error,
        sparse_enabled=layer_sparse_enabled,
        sparse_active_fraction=perturb_config.sparse_active_fraction,
        sparse_tau_global=perturb_config.sparse_tau_global,
        sparse_activity_callback=sparse_activity_callback,
    )


class MLXActivation(nn.Module):
    """Activation wrapper with optional bitstream execution."""

    def __init__(
        self,
        *,
        activation_kind: str,
        module_key: str,
        perturb_config: MLXPerturbationConfig | None = None,
    ) -> None:
        super().__init__()
        self.activation_kind = str(activation_kind).strip().lower()
        if self.activation_kind not in {"relu", "silu", "swish"}:
            raise ValueError(f"Unsupported activation: {activation_kind}")
        self.module_key = module_key
        self.perturb_config = perturb_config
        _register_bitstream_targetable_module_family(
            perturb_config,
            module_key,
            "activation",
        )

    def __call__(self, x):
        _, nn_module = _ensure_mlx_modules()
        runtime_cfg = _bitstream_execution_wants(
            self.perturb_config,
            self.module_key,
            family="activation",
        )
        if runtime_cfg is not None:
            y = _execute_bitstream_activation(
                x,
                activation_kind=self.activation_kind,
                runtime_cfg=runtime_cfg,
                module_key=self.module_key,
            )
            return _apply_sparse_gate_to_tensor(
                y,
                self.perturb_config,
                channel_dim=-1,
                module_key=self.module_key,
            )
        if self.activation_kind == "relu":
            y = nn_module.ReLU()(x)
        else:
            y = nn_module.SiLU()(x)
        return _apply_perturb_to_tensor(
            y,
            self.perturb_config,
            channel_dim=-1,
            module_key=self.module_key,
        )


class MLXLayerNorm(nn.Module):
    """LayerNorm wrapper with optional bitstream execution."""

    def __init__(
        self,
        normalized_shape: int,
        *,
        eps: float = 1e-5,
        module_key: str,
        perturb_config: MLXPerturbationConfig | None = None,
    ) -> None:
        super().__init__()
        mx_module, _ = _ensure_mlx_modules()
        self.weight = mx_module.ones((normalized_shape,))
        self.bias = mx_module.zeros((normalized_shape,))
        self.eps = float(eps)
        self.module_key = module_key
        self.perturb_config = perturb_config
        _register_bitstream_targetable_module_family(
            perturb_config,
            module_key,
            "norm",
        )

    def __call__(self, x):
        mx_module, _ = _ensure_mlx_modules()
        runtime_cfg = _bitstream_execution_wants(
            self.perturb_config,
            self.module_key,
            family="norm",
        )
        if runtime_cfg is not None:
            y = _execute_bitstream_layer_norm(
                x,
                weight=self.weight,
                bias=self.bias,
                eps=self.eps,
                runtime_cfg=runtime_cfg,
                module_key=self.module_key,
            )
            return _apply_sparse_gate_to_tensor(
                y,
                self.perturb_config,
                channel_dim=-1,
                module_key=self.module_key,
            )
        mean = mx_module.mean(x, axis=-1, keepdims=True)
        var = mx_module.var(x, axis=-1, keepdims=True)
        y = ((x - mean) * mx_module.rsqrt(var + self.eps)) * self.weight + self.bias
        return _apply_perturb_to_tensor(
            y,
            self.perturb_config,
            channel_dim=-1,
            module_key=self.module_key,
        )


class MLXConvNormAct(nn.Module):
    """Conv + optional norm + optional activation with CVNets-compatible naming."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        use_norm: bool = True,
        use_act: bool = True,
        act_name: str = "silu",
        module_key: str,
        perturb_config: MLXPerturbationConfig | None = None,
    ) -> None:
        super().__init__()
        _, nn_module = _ensure_mlx_modules()
        padding = ((kernel_size - 1) // 2) * dilation
        conv_module_key = f"{module_key}.conv"
        self.block: dict[str, Any] = {
            "conv": nn_module.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
        }
        _register_bitstream_targetable_module_family(
            perturb_config,
            conv_module_key,
            "conv2d",
        )
        if use_norm:
            self.block["norm"] = MLXBatchNorm2d(
                num_features=out_channels,
                eps=1e-5,
                module_key=f"{module_key}.norm",
                perturb_config=perturb_config,
            )
        self.use_norm = use_norm
        self.use_act = use_act
        self.act = None
        if use_act:
            self.act = MLXActivation(
                activation_kind=act_name,
                module_key=f"{module_key}.act",
                perturb_config=perturb_config,
            )
        self.module_key = module_key
        self.conv_module_key = conv_module_key
        self.perturb_config = perturb_config

    def __call__(self, x):
        runtime_cfg = _bitstream_execution_wants(
            self.perturb_config,
            self.conv_module_key,
            family="conv2d",
        )
        if runtime_cfg is not None:
            y = _execute_bitstream_conv2d_nhwc(
                x,
                weight=self.block["conv"].weight,
                bias=getattr(self.block["conv"], "bias", None),
                stride=getattr(self.block["conv"], "stride", (1, 1)),
                padding=getattr(self.block["conv"], "padding", (0, 0)),
                dilation=getattr(self.block["conv"], "dilation", (1, 1)),
                groups=int(getattr(self.block["conv"], "groups", 1)),
                runtime_cfg=runtime_cfg,
                module_key=self.conv_module_key,
            )
            y = _apply_sparse_gate_to_tensor(
                y,
                self.perturb_config,
                channel_dim=-1,
                module_key=self.conv_module_key,
            )
        else:
            y = self.block["conv"](x)
        if self.use_norm:
            y = self.block["norm"](y)
        if self.use_act and self.act is not None:
            y = self.act(y)
        if runtime_cfg is not None:
            return y
        return _apply_perturb_to_tensor(
            y,
            self.perturb_config,
            channel_dim=-1,
            module_key=self.module_key,
        )


class MLXBatchNorm2d(nn.Module):
    """Inference-only BatchNorm with explicit PyTorch eval-mode math."""

    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        module_key: str | None = None,
        perturb_config: MLXPerturbationConfig | None = None,
    ) -> None:
        super().__init__()
        mx_module, _ = _ensure_mlx_modules()
        self.weight = mx_module.ones((num_features,))
        self.bias = mx_module.zeros((num_features,))
        self.running_mean = mx_module.zeros((num_features,))
        self.running_var = mx_module.ones((num_features,))
        self.eps = float(eps)
        self.module_key = module_key
        self.perturb_config = perturb_config
        if module_key is not None:
            _register_bitstream_targetable_module_family(
                perturb_config,
                module_key,
                "norm",
            )

    def __call__(self, x):
        mx_module, _ = _ensure_mlx_modules()
        if self.module_key is not None:
            runtime_cfg = _bitstream_execution_wants(
                self.perturb_config,
                self.module_key,
                family="norm",
            )
            if runtime_cfg is not None:
                y = _execute_bitstream_batch_norm2d(
                    x,
                    weight=self.weight,
                    bias=self.bias,
                    running_mean=self.running_mean,
                    running_var=self.running_var,
                    eps=self.eps,
                    runtime_cfg=runtime_cfg,
                    module_key=self.module_key,
                )
                return _apply_sparse_gate_to_tensor(
                    y,
                    self.perturb_config,
                    channel_dim=-1,
                    module_key=self.module_key,
                )
        shape = [1] * x.ndim
        shape[-1] = int(self.weight.shape[0])
        mean = self.running_mean.reshape(shape).astype(x.dtype)
        var = self.running_var.reshape(shape).astype(x.dtype)
        weight = self.weight.reshape(shape).astype(x.dtype)
        bias = self.bias.reshape(shape).astype(x.dtype)
        return ((x - mean) * mx_module.rsqrt(var + self.eps)) * weight + bias


class MLXLinearLayer(nn.Module):
    """Linear layer with optional perturbation."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        bias: bool,
        module_key: str,
        perturb_config: MLXPerturbationConfig | None = None,
    ) -> None:
        super().__init__()
        mx_module, _ = _ensure_mlx_modules()
        self.weight = mx_module.zeros((out_features, in_features))
        if bias:
            self.bias = mx_module.zeros((out_features,))
        self.module_key = module_key
        self.perturb_config = perturb_config
        _register_bitstream_targetable_module_family(
            perturb_config,
            module_key,
            "linear",
        )

    def __call__(self, x):
        runtime_cfg = _bitstream_execution_wants(
            self.perturb_config,
            self.module_key,
            family="linear",
        )
        if runtime_cfg is not None:
            call_index = int(runtime_cfg.module_call_counts.get(self.module_key, 0))
            if self.perturb_config is not None and self.perturb_config.bitstream_slice_recorder is not None:
                _record_bitstream_tensor(
                    self.perturb_config,
                    self.module_key,
                    x.reshape((-1, x.shape[-1]))[0],
                    operand_role="lhs_activation_slice",
                    call_index=call_index,
                )
                _record_bitstream_tensor(
                    self.perturb_config,
                    self.module_key,
                    self["weight"][0],
                    operand_role="rhs_weight_row_slice",
                    call_index=call_index,
                )
            y = _execute_bitstream_matmul(
                x,
                self["weight"].T,
                runtime_cfg=runtime_cfg,
                module_key=self.module_key,
            )
            if self.perturb_config is not None and self.perturb_config.bitstream_slice_recorder is not None:
                _record_bitstream_tensor(
                    self.perturb_config,
                    self.module_key,
                    y.reshape((-1,))[:1],
                    operand_role="output_dot_slice",
                    call_index=call_index,
                )
        else:
            y = x @ self["weight"].T
        if "bias" in self:
            if (
                runtime_cfg is not None
                and self.perturb_config is not None
                and self.perturb_config.bitstream_slice_recorder is not None
            ):
                _record_bitstream_tensor(
                    self.perturb_config,
                    self.module_key,
                    self["bias"][:1],
                    operand_role="bias_scalar",
                    call_index=call_index,
                )
            y = y + self["bias"]
        if runtime_cfg is not None:
            if self.perturb_config is not None and self.perturb_config.bitstream_slice_recorder is not None:
                _record_bitstream_tensor(
                    self.perturb_config,
                    self.module_key,
                    y.reshape((-1,))[:1],
                    operand_role="output",
                    call_index=int(runtime_cfg.module_call_counts.get(self.module_key, 0)) - 1,
                )
            return _apply_sparse_gate_to_tensor(
                y,
                self.perturb_config,
                channel_dim=-1,
                module_key=self.module_key,
            )
        return _apply_perturb_to_tensor(
            y,
            self.perturb_config,
            channel_dim=-1,
            module_key=self.module_key,
        )


class MLXMultiHeadAttention(nn.Module):
    """CVNets-compatible multi-head attention in MLX."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        module_key: str,
        perturb_config: MLXPerturbationConfig | None = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}"
            )
        self.qkv_proj = MLXLinearLayer(
            in_features=embed_dim,
            out_features=3 * embed_dim,
            bias=True,
            module_key=f"{module_key}.qkv_proj",
            perturb_config=perturb_config,
        )
        self.out_proj = MLXLinearLayer(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=True,
            module_key=f"{module_key}.out_proj",
            perturb_config=perturb_config,
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.module_key = module_key
        self.perturb_config = perturb_config
        _register_bitstream_targetable_module_family(
            perturb_config,
            f"{module_key}.attn_scores",
            "linear",
        )
        _register_bitstream_targetable_module_family(
            perturb_config,
            f"{module_key}.attn_output",
            "linear",
        )

    def __call__(
        self,
        *,
        x_q,
        x_kv=None,
        key_padding_mask=None,
        attn_mask=None,
    ):
        mx_module, _ = _ensure_mlx_modules()
        batch_size, src_len, _ = x_q.shape
        if x_kv is None:
            qkv = self.qkv_proj(x_q).reshape(
                batch_size,
                src_len,
                3,
                self.num_heads,
                self.head_dim,
            )
            qkv = mx_module.transpose(qkv, (0, 3, 2, 1, 4))
            query = qkv[:, :, 0]
            key = qkv[:, :, 1]
            value = qkv[:, :, 2]
        else:
            tgt_len = x_kv.shape[1]
            qkv_q = self.qkv_proj(x_q)
            query = qkv_q[:, :, : self.embed_dim]
            query = mx_module.transpose(
                query.reshape(batch_size, src_len, self.num_heads, self.head_dim),
                (0, 2, 1, 3),
            )
            kv = self.qkv_proj(x_kv)[:, :, self.embed_dim :]
            kv = kv.reshape(batch_size, tgt_len, 2, self.num_heads, self.head_dim)
            kv = mx_module.transpose(kv, (0, 3, 2, 1, 4))
            key = kv[:, :, 0]
            value = kv[:, :, 1]

        query = query * self.scaling
        key = mx_module.transpose(key, (0, 1, 3, 2))
        attn_scores_key = f"{self.module_key}.attn_scores"
        attn_scores_runtime = _bitstream_execution_wants(
            self.perturb_config,
            attn_scores_key,
            family="linear",
        )
        if attn_scores_runtime is not None:
            attention_scores = _execute_bitstream_matmul(
                query,
                key,
                runtime_cfg=attn_scores_runtime,
                module_key=attn_scores_key,
            )
            attention_scores = _apply_sparse_gate_to_tensor(
                attention_scores,
                self.perturb_config,
                channel_dim=-1,
                module_key=attn_scores_key,
            )
        else:
            attention_scores = mx_module.matmul(query, key)
            attention_scores = _apply_perturb_to_tensor(
                attention_scores,
                self.perturb_config,
                channel_dim=-1,
                module_key=attn_scores_key,
            )
        if attn_mask is not None:
            attention_scores = attention_scores + attn_mask[:, None, :, :]
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]
            neg_inf = mx_module.full(attention_scores.shape, float("-inf"))
            attention_scores = mx_module.where(mask, neg_inf, attention_scores)

        attention_weights = mx_module.softmax(attention_scores, axis=-1)
        attn_output_key = f"{self.module_key}.attn_output"
        attn_output_runtime = _bitstream_execution_wants(
            self.perturb_config,
            attn_output_key,
            family="linear",
        )
        if attn_output_runtime is not None:
            attention_output = _execute_bitstream_matmul(
                attention_weights,
                value,
                runtime_cfg=attn_output_runtime,
                module_key=attn_output_key,
            )
            attention_output = _apply_sparse_gate_to_tensor(
                attention_output,
                self.perturb_config,
                channel_dim=-1,
                module_key=attn_output_key,
            )
        else:
            attention_output = mx_module.matmul(attention_weights, value)
            attention_output = _apply_perturb_to_tensor(
                attention_output,
                self.perturb_config,
                channel_dim=-1,
                module_key=attn_output_key,
            )
        attention_output = mx_module.transpose(attention_output, (0, 2, 1, 3)).reshape(
            batch_size,
            src_len,
            self.embed_dim,
        )
        return self.out_proj(attention_output)


class MLXTransformerEncoder(nn.Module):
    """CVNets-compatible transformer encoder block."""

    def __init__(
        self,
        *,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: int,
        module_key: str,
        perturb_config: MLXPerturbationConfig | None = None,
    ) -> None:
        super().__init__()
        _, nn_module = _ensure_mlx_modules()
        self.pre_norm_mha = [
            MLXLayerNorm(
                embed_dim,
                eps=1e-5,
                module_key=f"{module_key}.pre_norm_mha.0",
                perturb_config=perturb_config,
            ),
            MLXMultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                module_key=f"{module_key}.pre_norm_mha.1",
                perturb_config=perturb_config,
            ),
            nn_module.Identity(),
        ]
        self.pre_norm_ffn = [
            MLXLayerNorm(
                embed_dim,
                eps=1e-5,
                module_key=f"{module_key}.pre_norm_ffn.0",
                perturb_config=perturb_config,
            ),
            MLXLinearLayer(
                in_features=embed_dim,
                out_features=ffn_latent_dim,
                bias=True,
                module_key=f"{module_key}.pre_norm_ffn.1",
                perturb_config=perturb_config,
            ),
            MLXActivation(
                activation_kind="silu",
                module_key=f"{module_key}.pre_norm_ffn.2",
                perturb_config=perturb_config,
            ),
            nn_module.Identity(),
            MLXLinearLayer(
                in_features=ffn_latent_dim,
                out_features=embed_dim,
                bias=True,
                module_key=f"{module_key}.pre_norm_ffn.4",
                perturb_config=perturb_config,
            ),
            nn_module.Identity(),
        ]

    def __call__(self, x, x_prev=None, key_padding_mask=None, attn_mask=None):
        res = x
        x_norm = self.pre_norm_mha[0](x)
        x_attn = self.pre_norm_mha[1](
            x_q=x_norm,
            x_kv=x_prev,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        x = res + self.pre_norm_mha[2](x_attn)

        y = self.pre_norm_ffn[0](x)
        y = self.pre_norm_ffn[1](y)
        y = self.pre_norm_ffn[2](y)
        y = self.pre_norm_ffn[3](y)
        y = self.pre_norm_ffn[4](y)
        y = self.pre_norm_ffn[5](y)
        return x + y


class MLXInvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block in NHWC layout."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
        module_key: str,
        perturb_config: MLXPerturbationConfig | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = int(math.ceil((in_channels * expand_ratio) / 8.0) * 8)
        self.block: dict[str, Any] = {}
        if expand_ratio != 1:
            self.block["exp_1x1"] = MLXConvNormAct(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=1,
                stride=1,
                use_norm=True,
                use_act=True,
                module_key=f"{module_key}.block.exp_1x1",
                perturb_config=perturb_config,
            )
        self.block["conv_3x3"] = MLXConvNormAct(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=stride,
            groups=hidden_dim,
            use_norm=True,
            use_act=True,
            module_key=f"{module_key}.block.conv_3x3",
            perturb_config=perturb_config,
        )
        self.block["red_1x1"] = MLXConvNormAct(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=False,
            module_key=f"{module_key}.block.red_1x1",
            perturb_config=perturb_config,
        )
        self.use_res_connect = stride == 1 and in_channels == out_channels

    def __call__(self, x):
        y = x
        if "exp_1x1" in self.block:
            y = self.block["exp_1x1"](y)
        y = self.block["conv_3x3"](y)
        y = self.block["red_1x1"](y)
        return x + y if self.use_res_connect else y


class MLXMobileViTBlock(nn.Module):
    """MobileViT block with CVNets-compatible parameter naming."""

    def __init__(
        self,
        *,
        in_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        n_transformer_blocks: int,
        head_dim: int,
        patch_h: int,
        patch_w: int,
        module_key: str,
        perturb_config: MLXPerturbationConfig | None = None,
        no_fusion: bool = False,
    ) -> None:
        super().__init__()
        self.local_rep: dict[str, Any] = {
            "conv_3x3": MLXConvNormAct(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                use_norm=True,
                use_act=True,
                module_key=f"{module_key}.local_rep.conv_3x3",
                perturb_config=perturb_config,
            ),
            "conv_1x1": MLXConvNormAct(
                in_channels=in_channels,
                out_channels=transformer_dim,
                kernel_size=1,
                stride=1,
                use_norm=False,
                use_act=False,
                module_key=f"{module_key}.local_rep.conv_1x1",
                perturb_config=perturb_config,
            ),
        }
        self.global_rep: list[Any] = [
            MLXTransformerEncoder(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=transformer_dim // head_dim,
                module_key=f"{module_key}.global_rep.{idx}",
                perturb_config=perturb_config,
            )
            for idx in range(n_transformer_blocks)
        ]
        self.global_rep.append(
            MLXLayerNorm(
                transformer_dim,
                eps=1e-5,
                module_key=f"{module_key}.global_rep.{len(self.global_rep)}",
                perturb_config=perturb_config,
            )
        )
        self.conv_proj = MLXConvNormAct(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=True,
            module_key=f"{module_key}.conv_proj",
            perturb_config=perturb_config,
        )
        self.fusion = None
        if not no_fusion:
            self.fusion = MLXConvNormAct(
                in_channels=2 * in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                use_norm=True,
                use_act=True,
                module_key=f"{module_key}.fusion",
                perturb_config=perturb_config,
            )
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = patch_h * patch_w

    def unfolding(self, feature_map):
        mx_module, _ = _ensure_mlx_modules()
        batch_size, orig_h, orig_w, in_channels = feature_map.shape
        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
        interpolate = False
        if new_h != orig_h or new_w != orig_w:
            feature_map = _resize_image_nhwc(feature_map, new_h, new_w)
            interpolate = True

        num_patch_h = new_h // self.patch_h
        num_patch_w = new_w // self.patch_w
        num_patches = num_patch_h * num_patch_w
        patches = feature_map.reshape(
            batch_size,
            num_patch_h,
            self.patch_h,
            num_patch_w,
            self.patch_w,
            in_channels,
        )
        patches = mx_module.transpose(patches, (0, 2, 4, 1, 3, 5))
        patches = patches.reshape(batch_size, self.patch_area, num_patches, in_channels)
        patches = patches.reshape(batch_size * self.patch_area, num_patches, in_channels)
        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }
        return patches, info_dict

    def folding(self, patches, info_dict: dict[str, Any]):
        mx_module, _ = _ensure_mlx_modules()
        batch_size = int(info_dict["batch_size"])
        num_patches = int(info_dict["total_patches"])
        num_patch_h = int(info_dict["num_patches_h"])
        num_patch_w = int(info_dict["num_patches_w"])
        channels = int(patches.shape[-1])
        patches = patches.reshape(batch_size, self.patch_area, num_patches, channels)
        patches = patches.reshape(
            batch_size,
            self.patch_h,
            self.patch_w,
            num_patch_h,
            num_patch_w,
            channels,
        )
        patches = mx_module.transpose(patches, (0, 3, 1, 4, 2, 5))
        feature_map = patches.reshape(
            batch_size,
            num_patch_h * self.patch_h,
            num_patch_w * self.patch_w,
            channels,
        )
        if info_dict["interpolate"]:
            orig_h, orig_w = info_dict["orig_size"]
            feature_map = _resize_image_nhwc(feature_map, int(orig_h), int(orig_w))
        return feature_map

    def __call__(self, x):
        mx_module, _ = _ensure_mlx_modules()
        res = x
        fm = self.local_rep["conv_3x3"](x)
        fm = self.local_rep["conv_1x1"](fm)
        patches, info_dict = self.unfolding(fm)
        for layer in self.global_rep:
            patches = layer(patches)
        fm = self.folding(patches, info_dict)
        fm = self.conv_proj(fm)
        if self.fusion is not None:
            fm = self.fusion(mx_module.concatenate([res, fm], axis=-1))
        return fm


class MLXMobileViT(nn.Module):
    """Inference-only MobileViT model compatible with exported CVNets weights."""

    def __init__(
        self,
        *,
        model_key: str,
        perturb_config: MLXPerturbationConfig | None = None,
    ) -> None:
        super().__init__()
        if model_key not in MODEL_SPECS:
            raise ValueError(f"Unsupported model key: {model_key}")
        spec = MODEL_SPECS[model_key]
        mode = str(spec.get("mode") or "")
        if mode == "xx_small":
            mv2_exp_mult = 2
            cfg = {
                "layer1": {"out_channels": 16, "expand_ratio": mv2_exp_mult, "num_blocks": 1, "stride": 1, "block_type": "mv2"},
                "layer2": {"out_channels": 24, "expand_ratio": mv2_exp_mult, "num_blocks": 3, "stride": 2, "block_type": "mv2"},
                "layer3": {"out_channels": 48, "transformer_channels": 64, "ffn_dim": 128, "transformer_blocks": 2, "patch_h": 2, "patch_w": 2, "stride": 2, "mv_expand_ratio": mv2_exp_mult, "head_dim": 16},
                "layer4": {"out_channels": 64, "transformer_channels": 80, "ffn_dim": 160, "transformer_blocks": 4, "patch_h": 2, "patch_w": 2, "stride": 2, "mv_expand_ratio": mv2_exp_mult, "head_dim": 20},
                "layer5": {"out_channels": 80, "transformer_channels": 96, "ffn_dim": 192, "transformer_blocks": 3, "patch_h": 2, "patch_w": 2, "stride": 2, "mv_expand_ratio": mv2_exp_mult, "head_dim": 24},
                "last_layer_exp_factor": 4,
            }
        elif mode == "x_small":
            mv2_exp_mult = 4
            cfg = {
                "layer1": {"out_channels": 32, "expand_ratio": mv2_exp_mult, "num_blocks": 1, "stride": 1, "block_type": "mv2"},
                "layer2": {"out_channels": 48, "expand_ratio": mv2_exp_mult, "num_blocks": 3, "stride": 2, "block_type": "mv2"},
                "layer3": {"out_channels": 64, "transformer_channels": 96, "ffn_dim": 192, "transformer_blocks": 2, "patch_h": 2, "patch_w": 2, "stride": 2, "mv_expand_ratio": mv2_exp_mult, "head_dim": 24},
                "layer4": {"out_channels": 80, "transformer_channels": 120, "ffn_dim": 240, "transformer_blocks": 4, "patch_h": 2, "patch_w": 2, "stride": 2, "mv_expand_ratio": mv2_exp_mult, "head_dim": 30},
                "layer5": {"out_channels": 96, "transformer_channels": 144, "ffn_dim": 288, "transformer_blocks": 3, "patch_h": 2, "patch_w": 2, "stride": 2, "mv_expand_ratio": mv2_exp_mult, "head_dim": 36},
                "last_layer_exp_factor": 4,
            }
        elif mode == "small":
            mv2_exp_mult = 4
            cfg = {
                "layer1": {"out_channels": 32, "expand_ratio": mv2_exp_mult, "num_blocks": 1, "stride": 1, "block_type": "mv2"},
                "layer2": {"out_channels": 64, "expand_ratio": mv2_exp_mult, "num_blocks": 3, "stride": 2, "block_type": "mv2"},
                "layer3": {"out_channels": 96, "transformer_channels": 144, "ffn_dim": 288, "transformer_blocks": 2, "patch_h": 2, "patch_w": 2, "stride": 2, "mv_expand_ratio": mv2_exp_mult, "head_dim": 36},
                "layer4": {"out_channels": 128, "transformer_channels": 192, "ffn_dim": 384, "transformer_blocks": 4, "patch_h": 2, "patch_w": 2, "stride": 2, "mv_expand_ratio": mv2_exp_mult, "head_dim": 48},
                "layer5": {"out_channels": 160, "transformer_channels": 240, "ffn_dim": 480, "transformer_blocks": 3, "patch_h": 2, "patch_w": 2, "stride": 2, "mv_expand_ratio": mv2_exp_mult, "head_dim": 60},
                "last_layer_exp_factor": 4,
            }
        else:
            raise ValueError(f"Unsupported MobileViT mode: {mode}")

        self.model_key = model_key
        self.input_size = int(spec["input_size"])
        self.perturb_config = perturb_config
        self.conv_1 = MLXConvNormAct(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
            module_key="conv_1",
            perturb_config=perturb_config,
        )
        in_channels = 16
        self.layer_1, out_channels = self._make_layer(
            input_channel=in_channels,
            cfg=cfg["layer1"],
            layer_name="layer_1",
        )
        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            input_channel=in_channels,
            cfg=cfg["layer2"],
            layer_name="layer_2",
        )
        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            input_channel=in_channels,
            cfg=cfg["layer3"],
            layer_name="layer_3",
        )
        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            input_channel=in_channels,
            cfg=cfg["layer4"],
            layer_name="layer_4",
        )
        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            input_channel=in_channels,
            cfg=cfg["layer5"],
            layer_name="layer_5",
        )
        in_channels = out_channels
        exp_channels = min(cfg["last_layer_exp_factor"] * in_channels, 960)
        self.conv_1x1_exp = MLXConvNormAct(
            in_channels=in_channels,
            out_channels=exp_channels,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=True,
            module_key="conv_1x1_exp",
            perturb_config=perturb_config,
        )
        self.classifier: dict[str, Any] = {
            "fc": MLXLinearLayer(
                in_features=exp_channels,
                out_features=1000,
                bias=True,
                module_key="classifier.fc",
                perturb_config=perturb_config,
            )
        }

    def _make_mobilenet_layer(
        self,
        *,
        input_channel: int,
        cfg: dict[str, Any],
        layer_name: str,
    ) -> tuple[list[Any], int]:
        output_channels = int(cfg["out_channels"])
        num_blocks = int(cfg.get("num_blocks", 2))
        expand_ratio = int(cfg.get("expand_ratio", 4))
        blocks: list[Any] = []
        for idx in range(num_blocks):
            stride = int(cfg.get("stride", 1)) if idx == 0 else 1
            blocks.append(
                MLXInvertedResidual(
                    in_channels=input_channel,
                    out_channels=output_channels,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    module_key=f"{layer_name}.{idx}",
                    perturb_config=self.perturb_config,
                )
            )
            input_channel = output_channels
        return blocks, input_channel

    def _make_mit_layer(
        self,
        *,
        input_channel: int,
        cfg: dict[str, Any],
        layer_name: str,
    ) -> tuple[list[Any], int]:
        blocks: list[Any] = []
        stride = int(cfg.get("stride", 1))
        if stride == 2:
            blocks.append(
                MLXInvertedResidual(
                    in_channels=input_channel,
                    out_channels=int(cfg["out_channels"]),
                    stride=stride,
                    expand_ratio=int(cfg.get("mv_expand_ratio", 4)),
                    module_key=f"{layer_name}.0",
                    perturb_config=self.perturb_config,
                )
            )
            input_channel = int(cfg["out_channels"])

        transformer_dim = int(cfg["transformer_channels"])
        head_dim = int(cfg.get("head_dim") or (transformer_dim // 4))
        blocks.append(
            MLXMobileViTBlock(
                in_channels=input_channel,
                transformer_dim=transformer_dim,
                ffn_dim=int(cfg["ffn_dim"]),
                n_transformer_blocks=int(cfg.get("transformer_blocks", 1)),
                head_dim=head_dim,
                patch_h=int(cfg.get("patch_h", 2)),
                patch_w=int(cfg.get("patch_w", 2)),
                module_key=f"{layer_name}.{len(blocks)}",
                perturb_config=self.perturb_config,
                no_fusion=False,
            )
        )
        return blocks, input_channel

    def _make_layer(
        self,
        *,
        input_channel: int,
        cfg: dict[str, Any],
        layer_name: str,
    ) -> tuple[list[Any], int]:
        if str(cfg.get("block_type", "mobilevit")).lower() == "mobilevit":
            return self._make_mit_layer(
                input_channel=input_channel,
                cfg=cfg,
                layer_name=layer_name,
            )
        return self._make_mobilenet_layer(
            input_channel=input_channel,
            cfg=cfg,
            layer_name=layer_name,
        )

    def forward_features(self, x):
        x = self.conv_1(x)
        for layer in self.layer_1:
            x = layer(x)
        for layer in self.layer_2:
            x = layer(x)
        for layer in self.layer_3:
            x = layer(x)
        for layer in self.layer_4:
            x = layer(x)
        for layer in self.layer_5:
            x = layer(x)
        x = self.conv_1x1_exp(x)
        return x

    def __call__(self, x):
        mx_module, _ = _ensure_mlx_modules()
        x = self.forward_features(x)
        x = mx_module.mean(x, axis=(1, 2))
        return self.classifier["fc"](x)


def resolve_default_mlx_weights_dir() -> Path:
    return ROOT_DIR / "experiments" / "results" / "generated_configs" / "mlx_weights"


def resolve_mlx_weights_cache_path(model_key: str, out_dir: str | Path | None = None) -> Path:
    base_dir = Path(out_dir) if out_dir is not None else resolve_default_mlx_weights_dir()
    return base_dir / f"{model_key}.npz"


def ensure_mlx_weights_exported(
    model_key: str,
    *,
    weights_dir: str | None = None,
    weights_override: str | None = None,
    out_dir: str | Path | None = None,
    force_reexport: bool = False,
) -> Path:
    weights_override_map = parse_weights_override(weights_override)
    output_path = resolve_mlx_weights_cache_path(model_key, out_dir=out_dir)
    if output_path.is_file() and not force_reexport:
        return output_path
    checkpoint_path = Path(
        resolve_weights_path(
            model_key,
            weights_dir,
            weights_override_map,
        )
    )
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    # Keep the active MLX inference import surface free of a hard PyTorch
    # dependency when a pre-exported NPZ already exists.
    from tools.export_mobilevit_weights_npz import export_checkpoint_to_npz

    export_checkpoint_to_npz(
        checkpoint_path,
        output_path,
        strip_prefix="module.",
        export_format="mlx_mobilevit",
    )
    return output_path


def build_mlx_mobilevit(
    model_key: str,
    *,
    perturb_config: MLXPerturbationConfig | None = None,
    weights_npz: str | Path | None = None,
    weights_dir: str | None = None,
    weights_override: str | None = None,
    mlx_weights_dir: str | Path | None = None,
    force_reexport: bool = False,
):
    _load_mlx_photonic_perturb_module().require_mlx()
    mx_module, _ = _ensure_mlx_modules()
    model = MLXMobileViT(model_key=model_key, perturb_config=perturb_config)
    weights_path = (
        Path(weights_npz)
        if weights_npz is not None
        else ensure_mlx_weights_exported(
            model_key,
            weights_dir=weights_dir,
            weights_override=weights_override,
            out_dir=mlx_weights_dir,
            force_reexport=force_reexport,
        )
    )
    model.load_weights(str(weights_path), strict=True)
    model.eval()
    mx_module.eval(model.parameters())
    return model, weights_path


__all__ = [
    "BITSTREAM_RUNTIME_FAMILY_POLICY_SOURCE",
    "DEFAULT_BITSTREAM_RUNTIME_STREAM_REUSE_POLICY",
    "BitstreamExecutionConfig",
    "BitstreamSliceRecorder",
    "MLXActivation",
    "MLXMobileViT",
    "MLXPerturbationConfig",
    "MLXLayerNorm",
    "SparseActivityRecorder",
    "build_mlx_mobilevit",
    "capture_mlx_bitstream_slices",
    "ensure_mlx_weights_exported",
    "recommended_bitstream_slice_targets",
    "resolve_bitstream_runtime_family_policy",
    "resolve_default_mlx_weights_dir",
    "resolve_mlx_weights_cache_path",
]
