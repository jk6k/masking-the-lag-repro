#!/usr/bin/env python3
"""Bounded Phase 4 validator for the repaired True-SC MLX runtime."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = ROOT / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from accuracy.mlx_mobilevit import (  # noqa: E402
    BitstreamExecutionConfig,
    MLXPerturbationConfig,
    _execute_bitstream_activation,
    _execute_bitstream_batch_norm2d,
    _execute_bitstream_conv2d_nhwc,
    _execute_bitstream_layer_norm,
    _execute_bitstream_matmul,
    _ensure_mlx_modules,
    build_mlx_mobilevit,
    recommended_bitstream_slice_targets,
)


DEFAULT_WEIGHTS_NPZ = (
    ROOT / "experiments" / "results" / "generated_configs" / "mlx_weights" / "mobilevit_s.npz"
)


@dataclass
class DiffStats:
    shape: list[int]
    max_abs: float
    mean_abs: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--weights_npz", default=str(DEFAULT_WEIGHTS_NPZ))
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--stream_length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--helper_max_abs_tolerance", type=float, default=1e-6)
    parser.add_argument("--helper_mean_abs_tolerance", type=float, default=1e-6)
    parser.add_argument("--model_compile_max_abs_tolerance", type=float, default=1e-5)
    parser.add_argument("--model_compile_mean_abs_tolerance", type=float, default=1e-6)
    return parser.parse_args()


def _require_mps(args: argparse.Namespace, mx_module) -> None:
    requested = str(args.device).strip().lower()
    if requested != "mps":
        raise SystemExit(
            f"Phase 4 validation only supports --device mps on this host, received: {args.device!r}"
        )
    mx_module.set_default_device(mx_module.gpu)
    default_device = str(mx_module.default_device()).lower()
    if "gpu" not in default_device:
        raise SystemExit(
            f"Requested MPS/GPU execution, but MLX default_device is {mx_module.default_device()}."
        )


def _diff_stats(reference, candidate) -> DiffStats:
    ref_np = np.asarray(reference, dtype=np.float32)
    cand_np = np.asarray(candidate, dtype=np.float32)
    diff = np.abs(ref_np - cand_np)
    return DiffStats(
        shape=[int(dim) for dim in ref_np.shape],
        max_abs=float(diff.max(initial=0.0)),
        mean_abs=float(diff.mean() if diff.size else 0.0),
    )


def _assert_within_tolerance(
    *,
    label: str,
    stats: DiffStats,
    max_abs_tolerance: float,
    mean_abs_tolerance: float,
) -> None:
    if stats.max_abs > float(max_abs_tolerance) or stats.mean_abs > float(mean_abs_tolerance):
        raise SystemExit(
            f"{label} exceeded tolerance: "
            f"max_abs={stats.max_abs:.8f} (limit {max_abs_tolerance:.8f}), "
            f"mean_abs={stats.mean_abs:.8f} (limit {mean_abs_tolerance:.8f})"
        )


def _runtime_cfg(*, stream_length: int, seed: int) -> BitstreamExecutionConfig:
    return BitstreamExecutionConfig(
        enabled=True,
        stream_length=int(stream_length),
        generator="low_discrepancy",
        stream_reuse_policy="operand_factored_module_call_reuse",
        seed=int(seed),
    )


def _helper_parity_report(mx_module, *, stream_length: int, seed: int) -> dict[str, DiffStats]:
    lhs_np = np.array(
        [[[0.75, -0.5, 0.25], [0.1, 0.6, -0.8]]],
        dtype=np.float32,
    )
    rhs_np = np.array(
        [[0.25, -0.5], [0.75, 0.1], [-0.4, 0.6]],
        dtype=np.float32,
    )
    matmul_ref = _execute_bitstream_matmul(
        lhs_np,
        rhs_np,
        runtime_cfg=_runtime_cfg(stream_length=stream_length, seed=seed),
        module_key="phase4.matmul",
    )
    matmul_mx = _execute_bitstream_matmul(
        mx_module.array(lhs_np),
        mx_module.array(rhs_np),
        runtime_cfg=_runtime_cfg(stream_length=stream_length, seed=seed),
        module_key="phase4.matmul",
    )

    conv_x_np = np.array(
        [[
            [[0.2], [0.4], [-0.1]],
            [[0.6], [-0.8], [0.5]],
            [[0.1], [0.3], [0.9]],
        ]],
        dtype=np.float32,
    )
    conv_weight_np = np.array(
        [[[[0.25], [-0.75]], [[0.5], [0.125]]]],
        dtype=np.float32,
    )
    conv_bias_np = np.array([0.2], dtype=np.float32)
    conv_ref = _execute_bitstream_conv2d_nhwc(
        conv_x_np,
        weight=conv_weight_np,
        bias=conv_bias_np,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        runtime_cfg=_runtime_cfg(stream_length=stream_length, seed=seed),
        module_key="phase4.conv2d",
    )
    conv_mx = _execute_bitstream_conv2d_nhwc(
        mx_module.array(conv_x_np),
        weight=mx_module.array(conv_weight_np),
        bias=mx_module.array(conv_bias_np),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        runtime_cfg=_runtime_cfg(stream_length=stream_length, seed=seed),
        module_key="phase4.conv2d",
    )

    attention_query_np = np.array(
        [[[[0.25, -0.4], [0.6, 0.1]]]],
        dtype=np.float32,
    )
    attention_key_np = np.array(
        [[[[0.2, -0.7], [0.5, 0.3]]]],
        dtype=np.float32,
    )
    attention_value_np = np.array(
        [[[[0.1, 0.9], [-0.2, 0.4]]]],
        dtype=np.float32,
    )
    attention_scores_ref = _execute_bitstream_matmul(
        attention_query_np,
        np.transpose(attention_key_np, (0, 1, 3, 2)),
        runtime_cfg=_runtime_cfg(stream_length=stream_length, seed=seed),
        module_key="phase4.attn_scores",
    )
    attention_scores_mx = _execute_bitstream_matmul(
        mx_module.array(attention_query_np),
        mx_module.transpose(mx_module.array(attention_key_np), (0, 1, 3, 2)),
        runtime_cfg=_runtime_cfg(stream_length=stream_length, seed=seed),
        module_key="phase4.attn_scores",
    )
    attention_weights_ref = _softmax_numpy(np.asarray(attention_scores_ref), axis=-1)
    attention_weights_mx = mx_module.softmax(attention_scores_mx, axis=-1)
    attention_output_ref = _execute_bitstream_matmul(
        attention_weights_ref,
        attention_value_np,
        runtime_cfg=_runtime_cfg(stream_length=stream_length, seed=seed),
        module_key="phase4.attn_output",
    )
    attention_output_mx = _execute_bitstream_matmul(
        attention_weights_mx,
        mx_module.array(attention_value_np),
        runtime_cfg=_runtime_cfg(stream_length=stream_length, seed=seed),
        module_key="phase4.attn_output",
    )

    activation_input_np = np.array([-0.8, 0.0, 0.8], dtype=np.float32)
    activation_ref = _execute_bitstream_activation(
        activation_input_np,
        activation_kind="silu",
        runtime_cfg=_runtime_cfg(stream_length=stream_length, seed=seed),
        module_key="phase4.activation",
    )
    activation_mx = _execute_bitstream_activation(
        mx_module.array(activation_input_np),
        activation_kind="silu",
        runtime_cfg=_runtime_cfg(stream_length=stream_length, seed=seed),
        module_key="phase4.activation",
    )

    layer_norm_input_np = np.array([[-0.8, 0.0, 0.8]], dtype=np.float32)
    layer_norm_weight_np = np.array([1.0, 0.75, 1.25], dtype=np.float32)
    layer_norm_bias_np = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    layer_norm_ref = _execute_bitstream_layer_norm(
        layer_norm_input_np,
        weight=layer_norm_weight_np,
        bias=layer_norm_bias_np,
        eps=1e-5,
        runtime_cfg=_runtime_cfg(stream_length=stream_length, seed=seed),
        module_key="phase4.layer_norm",
    )
    layer_norm_mx = _execute_bitstream_layer_norm(
        mx_module.array(layer_norm_input_np),
        weight=mx_module.array(layer_norm_weight_np),
        bias=mx_module.array(layer_norm_bias_np),
        eps=1e-5,
        runtime_cfg=_runtime_cfg(stream_length=stream_length, seed=seed),
        module_key="phase4.layer_norm",
    )

    batch_norm_input_np = np.array(
        [[
            [[-0.8, 0.25], [0.0, -0.5]],
            [[0.8, 0.75], [0.4, 0.0]],
        ]],
        dtype=np.float32,
    )
    batch_norm_weight_np = np.array([1.0, 0.5], dtype=np.float32)
    batch_norm_bias_np = np.array([0.1, -0.2], dtype=np.float32)
    batch_norm_mean_np = np.array([0.25, -0.1], dtype=np.float32)
    batch_norm_var_np = np.array([0.5, 0.75], dtype=np.float32)
    batch_norm_ref = _execute_bitstream_batch_norm2d(
        batch_norm_input_np,
        weight=batch_norm_weight_np,
        bias=batch_norm_bias_np,
        running_mean=batch_norm_mean_np,
        running_var=batch_norm_var_np,
        eps=1e-5,
        runtime_cfg=_runtime_cfg(stream_length=stream_length, seed=seed),
        module_key="phase4.batch_norm",
    )
    batch_norm_mx = _execute_bitstream_batch_norm2d(
        mx_module.array(batch_norm_input_np),
        weight=mx_module.array(batch_norm_weight_np),
        bias=mx_module.array(batch_norm_bias_np),
        running_mean=mx_module.array(batch_norm_mean_np),
        running_var=mx_module.array(batch_norm_var_np),
        eps=1e-5,
        runtime_cfg=_runtime_cfg(stream_length=stream_length, seed=seed),
        module_key="phase4.batch_norm",
    )

    return {
        "matmul": _diff_stats(matmul_ref, matmul_mx),
        "conv2d": _diff_stats(conv_ref, conv_mx),
        "attention_scores": _diff_stats(attention_scores_ref, attention_scores_mx),
        "attention_output": _diff_stats(attention_output_ref, attention_output_mx),
        "activation": _diff_stats(activation_ref, activation_mx),
        "layer_norm": _diff_stats(layer_norm_ref, layer_norm_mx),
        "batch_norm": _diff_stats(batch_norm_ref, batch_norm_mx),
    }


def _softmax_numpy(values: np.ndarray, *, axis: int) -> np.ndarray:
    stable = values - np.max(values, axis=axis, keepdims=True)
    numer = np.exp(stable)
    return numer / np.sum(numer, axis=axis, keepdims=True)


def _compiled_helper_report(mx_module, *, stream_length: int, seed: int) -> dict[str, DiffStats]:
    lhs = mx_module.array(
        [[[0.5, -0.25, 0.75], [0.1, 0.2, -0.3]]],
        dtype=mx_module.float32,
    )
    rhs = mx_module.array(
        [[0.3, -0.7], [0.4, 0.25], [-0.6, 0.5]],
        dtype=mx_module.float32,
    )

    def eager_matmul(lhs_arg, rhs_arg):
        return _execute_bitstream_matmul(
            lhs_arg,
            rhs_arg,
            runtime_cfg=_runtime_cfg(stream_length=stream_length, seed=seed),
            module_key="phase4.compiled.matmul",
        )

    compiled_matmul = mx_module.compile(
        lambda lhs_arg, rhs_arg: _execute_bitstream_matmul(
            lhs_arg,
            rhs_arg,
            runtime_cfg=_runtime_cfg(stream_length=stream_length, seed=seed),
            module_key="phase4.compiled.matmul",
        )
    )
    eager_matmul_out = eager_matmul(lhs, rhs)
    compiled_matmul_out = compiled_matmul(lhs, rhs)
    mx_module.eval(compiled_matmul_out)

    conv_x = mx_module.array(
        [[
            [[0.2], [0.4], [-0.1]],
            [[0.6], [-0.8], [0.5]],
            [[0.1], [0.3], [0.9]],
        ]],
        dtype=mx_module.float32,
    )
    conv_weight = mx_module.array(
        [[[[0.25], [-0.75]], [[0.5], [0.125]]]],
        dtype=mx_module.float32,
    )
    conv_bias = mx_module.array([0.2], dtype=mx_module.float32)

    def eager_conv(x_arg, weight_arg, bias_arg):
        return _execute_bitstream_conv2d_nhwc(
            x_arg,
            weight=weight_arg,
            bias=bias_arg,
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            runtime_cfg=_runtime_cfg(stream_length=stream_length, seed=seed),
            module_key="phase4.compiled.conv2d",
        )

    compiled_conv = mx_module.compile(
        lambda x_arg, weight_arg, bias_arg: _execute_bitstream_conv2d_nhwc(
            x_arg,
            weight=weight_arg,
            bias=bias_arg,
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            runtime_cfg=_runtime_cfg(stream_length=stream_length, seed=seed),
            module_key="phase4.compiled.conv2d",
        )
    )
    eager_conv_out = eager_conv(conv_x, conv_weight, conv_bias)
    compiled_conv_out = compiled_conv(conv_x, conv_weight, conv_bias)
    mx_module.eval(compiled_conv_out)

    return {
        "matmul": _diff_stats(eager_matmul_out, compiled_matmul_out),
        "conv2d": _diff_stats(eager_conv_out, compiled_conv_out),
    }


def _model_compile_report(
    mx_module,
    *,
    stream_length: int,
    seed: int,
    weights_npz: Path,
) -> dict[str, object]:
    if not weights_npz.is_file():
        raise SystemExit(f"Expected MLX weights NPZ for Phase 4 model compile smoke: {weights_npz}")
    target_keys = recommended_bitstream_slice_targets("mobilevit_s")
    target_module_keys = set(target_keys.values())
    lane_module_key = target_keys["primary"].removesuffix(".attn_scores")

    def build_lane():
        runtime_cfg = BitstreamExecutionConfig(
            enabled=True,
            target_module_keys=set(target_module_keys),
            stream_length=int(stream_length),
            generator="low_discrepancy",
            stream_reuse_policy="operand_factored_module_call_reuse",
            seed=int(seed),
            surface_scope="phase4_compile_attention_limited",
            measurement_truth_class="bitstream_model_level_measured",
            contract_note="phase4_compile_smoke",
        )
        perturb_config = MLXPerturbationConfig(
            enabled=True,
            bitstream_execution_config=runtime_cfg,
        )
        return build_mlx_mobilevit(
            "mobilevit_s",
            perturb_config=perturb_config,
            weights_npz=str(weights_npz),
        )

    eager_model, resolved_weights = build_lane()
    compiled_model_ref, _ = build_lane()
    eager_attention = eager_model.layer_4[1].global_rep[0].pre_norm_mha[1]
    compiled_attention = compiled_model_ref.layer_4[1].global_rep[0].pre_norm_mha[1]
    token_count = 4
    inputs = mx_module.random.uniform(
        low=-1.0,
        high=1.0,
        shape=(1, token_count, eager_attention.embed_dim),
    ).astype(mx_module.float32)
    eager_output = eager_attention(x_q=inputs)
    compiled_model = mx_module.compile(lambda tokens: compiled_attention(x_q=tokens))
    compiled_output = compiled_model(inputs)
    mx_module.eval(eager_output, compiled_output)
    return {
        "weights_npz": str(resolved_weights),
        "target_module_keys": sorted(target_module_keys),
        "lane_module_key": lane_module_key,
        "lane_input_shape": [int(dim) for dim in inputs.shape],
        "output_shape": [int(dim) for dim in compiled_output.shape],
        "diff": asdict(_diff_stats(eager_output, compiled_output)),
    }


def main() -> int:
    args = _parse_args()
    mx_module, _ = _ensure_mlx_modules()
    _require_mps(args, mx_module)

    helper_report = _helper_parity_report(
        mx_module,
        stream_length=int(args.stream_length),
        seed=int(args.seed),
    )
    for label, stats in helper_report.items():
        _assert_within_tolerance(
            label=label,
            stats=stats,
            max_abs_tolerance=float(args.helper_max_abs_tolerance),
            mean_abs_tolerance=float(args.helper_mean_abs_tolerance),
        )

    compiled_helper_report = _compiled_helper_report(
        mx_module,
        stream_length=int(args.stream_length),
        seed=int(args.seed),
    )
    for label, stats in compiled_helper_report.items():
        _assert_within_tolerance(
            label=f"compiled_{label}",
            stats=stats,
            max_abs_tolerance=float(args.helper_max_abs_tolerance),
            mean_abs_tolerance=float(args.helper_mean_abs_tolerance),
        )

    model_compile_report = _model_compile_report(
        mx_module,
        stream_length=int(args.stream_length),
        seed=int(args.seed),
        weights_npz=Path(args.weights_npz),
    )
    _assert_within_tolerance(
        label="compiled_mobilevit_s_lane",
        stats=DiffStats(**model_compile_report["diff"]),
        max_abs_tolerance=float(args.model_compile_max_abs_tolerance),
        mean_abs_tolerance=float(args.model_compile_mean_abs_tolerance),
    )

    report = {
        "status": "phase4_validated",
        "device": str(args.device),
        "mlx_default_device": str(mx_module.default_device()),
        "stream_length": int(args.stream_length),
        "seed": int(args.seed),
        "helper_parity": {key: asdict(value) for key, value in helper_report.items()},
        "compiled_helper_parity": {
            key: asdict(value) for key, value in compiled_helper_report.items()
        },
        "model_compile_smoke": model_compile_report,
        "runtime_operational_ready": True,
    }
    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
