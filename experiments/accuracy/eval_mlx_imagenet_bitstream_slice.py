#!/usr/bin/env python3
"""Materialize or execute a bounded MLX bitstream-slice capture/replay run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from accuracy.bitstream_capture import parse_operand_values, write_capture_manifest  # noqa: E402
from accuracy.bitstream_semantics import (  # noqa: E402
    BitstreamSemanticsConfig,
    normalize_bitstream_semantics,
)
from accuracy.eval_mlx_imagenet_noise import (  # noqa: E402
    OpenCVImageNetDataset,
    _get_git_hash,
    parse_float_list,
    set_seeds,
)
from accuracy.mlx_mobilevit import (  # noqa: E402
    BitstreamSliceRecorder,
    MLXPerturbationConfig,
    build_mlx_mobilevit,
    capture_mlx_bitstream_slices,
    recommended_bitstream_slice_targets,
)
from sc_bitstream.kernels import estimate_dot_product  # noqa: E402


def build_request(args: argparse.Namespace) -> dict[str, object]:
    recommended_targets = recommended_bitstream_slice_targets(args.model)
    module_key = str(args.module_key or "").strip() or recommended_targets["primary"]
    control_module_key = (
        str(args.control_module_key or "").strip() or recommended_targets["control"]
    )
    semantics = normalize_bitstream_semantics(
        {
            "execution_semantics": "bitstream",
            "encoding_mode": args.encoding_mode,
            "multiplier_mode": args.multiplier_mode,
            "accumulator_mode": args.accumulator_mode,
            "stream_length": args.stream_length,
            "generator": args.generator,
            "calibration_source": args.calibration_source,
            "sign_mapping": args.sign_mapping,
        }
    )
    return {
        "request_kind": "mlx_bitstream_slice_replay_request",
        "mode": getattr(args, "mode", "request"),
        "model": args.model,
        "module_key": module_key,
        "control_module_key": control_module_key,
        "seed": args.seed,
        "semantics": semantics.to_dict(),
        "notes": args.notes or "",
        "status": "scaffold_only_not_executed",
    }


def _resolve_slice_targets(args: argparse.Namespace) -> tuple[str, str]:
    recommended_targets = recommended_bitstream_slice_targets(args.model)
    module_key = str(args.module_key or "").strip() or recommended_targets["primary"]
    control_module_key = (
        str(args.control_module_key or "").strip() or recommended_targets["control"]
    )
    return module_key, control_module_key


def _resolve_semantics(args: argparse.Namespace) -> BitstreamSemanticsConfig:
    return normalize_bitstream_semantics(
        {
            "execution_semantics": "bitstream",
            "encoding_mode": args.encoding_mode,
            "multiplier_mode": args.multiplier_mode,
            "accumulator_mode": args.accumulator_mode,
            "stream_length": args.stream_length,
            "generator": args.generator,
            "calibration_source": args.calibration_source,
            "sign_mapping": args.sign_mapping,
        }
    )


def _load_single_dataset_input(
    args: argparse.Namespace,
    *,
    input_size: int,
) -> tuple[np.ndarray, int, str, int]:
    if not args.imagenet_val:
        raise SystemExit("--imagenet_val is required in execute mode.")
    resize_size = args.resize_size if args.resize_size is not None else input_size + 32
    crop_size = args.center_crop_size if args.center_crop_size is not None else input_size
    if resize_size < crop_size:
        resize_size = crop_size
    dataset = OpenCVImageNetDataset(
        args.imagenet_val,
        manifest_path=args.imagenet_manifest,
        resize_size=resize_size,
        center_crop_size=crop_size,
        percentage=100.0,
        seed=int(args.seed),
        enable_mean_std=bool(args.enable_mean_std),
        mean_std_mean=parse_float_list(args.mean_std_mean),
        mean_std_std=parse_float_list(args.mean_std_std),
        input_color_order=args.data_color_order,
        model_color_order=args.input_color_order,
        input_scale=float(args.input_scale),
    )
    sample_index = int(args.sample_index)
    if sample_index < 0 or sample_index >= len(dataset):
        raise SystemExit(
            f"--sample_index={sample_index} is out of range for dataset length {len(dataset)}."
        )
    image_np, label = dataset[sample_index]
    sample_path, _ = dataset.samples[sample_index]
    return image_np, int(label), str(sample_path), len(dataset)


def _normalize_for_semantics(
    values: list[float],
    *,
    encoding_mode: str,
) -> tuple[list[float], float]:
    if not values:
        return [], 1.0
    if encoding_mode == "bipolar":
        scale = max(1.0, max(abs(float(value)) for value in values))
        return [float(value) / scale for value in values], scale
    scale = max(1.0, max(float(value) for value in values))
    return [max(0.0, float(value)) / scale for value in values], scale


def _replay_roles_for_module(module_key: str) -> tuple[str, str, str, str]:
    if module_key.endswith(".attn_scores"):
        return (
            "attn_qk_scalar",
            "lhs_query_slice",
            "rhs_key_slice",
            "output_score_slice",
        )
    return (
        "linear_scalar",
        "lhs_activation_slice",
        "rhs_weight_row_slice",
        "output_dot_slice",
    )


def _replay_module(
    module_key: str,
    module_rows: list[dict[str, object]],
    *,
    semantics: BitstreamSemanticsConfig,
    seed: int,
) -> dict[str, object]:
    replay_kind, lhs_role, rhs_role, output_role = _replay_roles_for_module(module_key)
    role_map = {
        str(row.get("operand_role") or ""): row
        for row in module_rows
    }
    missing_roles = [
        role
        for role in (lhs_role, rhs_role, output_role)
        if role not in role_map
    ]
    if missing_roles:
        return {
            "module_key": module_key,
            "replay_kind": replay_kind,
            "status": "missing_operands",
            "missing_roles": missing_roles,
        }
    lhs_raw = parse_operand_values(role_map[lhs_role])
    rhs_raw = parse_operand_values(role_map[rhs_role])
    output_raw = parse_operand_values(role_map[output_role])
    usable_length = min(len(lhs_raw), len(rhs_raw))
    if usable_length <= 0:
        return {
            "module_key": module_key,
            "replay_kind": replay_kind,
            "status": "empty_operands",
        }
    lhs_raw = lhs_raw[:usable_length]
    rhs_raw = rhs_raw[:usable_length]
    lhs_scaled, lhs_scale = _normalize_for_semantics(
        lhs_raw,
        encoding_mode=semantics.encoding_mode,
    )
    rhs_scaled, rhs_scale = _normalize_for_semantics(
        rhs_raw,
        encoding_mode=semantics.encoding_mode,
    )
    replay = estimate_dot_product(
        lhs_scaled,
        rhs_scaled,
        stream_length=semantics.stream_length,
        generator=semantics.generator,
        encoding_mode=semantics.encoding_mode,
        multiplier_mode=semantics.multiplier_mode,
        seed=seed,
    )
    rescale_factor = lhs_scale * rhs_scale
    rescaled_estimate = float(replay["estimated_value"]) * rescale_factor
    raw_exact = sum(lhs * rhs for lhs, rhs in zip(lhs_raw, rhs_raw))
    captured_output = output_raw[0] if output_raw else None
    result = {
        "module_key": module_key,
        "replay_kind": replay_kind,
        "status": "replayed",
        "operand_length": usable_length,
        "lhs_scale": lhs_scale,
        "rhs_scale": rhs_scale,
        "rescale_factor": rescale_factor,
        "normalized_exact_dot_product": float(replay["exact_value"]),
        "normalized_estimated_dot_product": float(replay["estimated_value"]),
        "raw_exact_dot_product": raw_exact,
        "rescaled_estimated_dot_product": rescaled_estimate,
        "abs_error_vs_raw_exact": abs(rescaled_estimate - raw_exact),
        "captured_output_scalar": captured_output,
        "stream_length": semantics.stream_length,
        "generator": semantics.generator,
        "encoding_mode": semantics.encoding_mode,
        "multiplier_mode": semantics.multiplier_mode,
        "accumulator_mode": semantics.accumulator_mode,
        "total_count": int(replay["total_count"]),
    }
    if captured_output is not None:
        result["abs_error_vs_captured_output"] = abs(rescaled_estimate - captured_output)
    bias_row = role_map.get("bias_scalar")
    if bias_row is not None:
        bias_values = parse_operand_values(bias_row)
        result["bias_scalar"] = bias_values[0] if bias_values else None
    return result


def build_replay_summary(
    rows: list[dict[str, object]],
    *,
    semantics: BitstreamSemanticsConfig,
    seed: int,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        module_key = str(row.get("module_key") or "")
        grouped.setdefault(module_key, []).append(row)
    return [
        _replay_module(module_key, module_rows, semantics=semantics, seed=seed)
        for module_key, module_rows in sorted(grouped.items())
    ]


def execute_bitstream_slice(args: argparse.Namespace) -> dict[str, object]:
    set_seeds(int(args.seed))
    semantics = _resolve_semantics(args)
    module_key, control_module_key = _resolve_slice_targets(args)
    perturb_config = MLXPerturbationConfig(enabled=False)
    model, weights_path = build_mlx_mobilevit(
        args.model,
        perturb_config=perturb_config,
        weights_npz=args.weights_npz,
        weights_dir=args.weights_dir,
        weights_override=args.weights_override,
        mlx_weights_dir=args.mlx_weights_dir,
        force_reexport=bool(args.force_reexport_mlx_weights),
    )
    setattr(model, "model_key", args.model)
    input_size = int(getattr(model, "input_size", 0) or 0)
    image_np, label, sample_path, dataset_length = _load_single_dataset_input(
        args,
        input_size=input_size,
    )
    recorder = BitstreamSliceRecorder(
        enabled=True,
        target_module_keys={module_key, control_module_key},
        encoding_mode=semantics.encoding_mode,
        multiplier_mode=semantics.multiplier_mode,
        accumulator_mode=semantics.accumulator_mode,
        stream_length=semantics.stream_length,
        generator=semantics.generator,
        max_values_per_capture=int(args.max_capture_values),
    )
    import mlx.core as mx

    inputs = mx.array(np.expand_dims(image_np, axis=0))
    started_at = time.perf_counter()
    rows = capture_mlx_bitstream_slices(
        model,
        inputs,
        recorder=recorder,
        row_context={"model": args.model},
    )
    measured_elapsed_s = time.perf_counter() - started_at
    capture_path = Path(args.capture_csv).resolve()
    write_capture_manifest(capture_path, rows)
    replay_rows = build_replay_summary(rows, semantics=semantics, seed=int(args.seed))
    return {
        "request_kind": "mlx_bitstream_slice_replay_execution",
        "status": "executed",
        "device": "mps",
        "model": args.model,
        "module_key": module_key,
        "control_module_key": control_module_key,
        "semantics": semantics.to_dict(),
        "seed": int(args.seed),
        "sample_index": int(args.sample_index),
        "sample_label": label,
        "sample_path": sample_path,
        "dataset_length": dataset_length,
        "capture_csv": str(capture_path),
        "capture_row_count": len(rows),
        "measured_elapsed_s": measured_elapsed_s,
        "weights_npz": str(weights_path),
        "git_hash": _get_git_hash(ROOT_DIR) or "nogit",
        "notes": args.notes or "",
        "replay": replay_rows,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize or execute a bounded MLX bitstream slice replay.",
    )
    parser.add_argument("--mode", choices=["request", "execute"], default="request")
    parser.add_argument("--model", default="mobilevit_s")
    parser.add_argument("--module_key", default="")
    parser.add_argument("--control_module_key", default="")
    parser.add_argument("--stream_length", type=int, default=64)
    parser.add_argument("--generator", default="bernoulli")
    parser.add_argument("--encoding_mode", default="bipolar")
    parser.add_argument("--multiplier_mode", default="xnor")
    parser.add_argument("--accumulator_mode", default="bitcount")
    parser.add_argument("--calibration_source", default="")
    parser.add_argument("--sign_mapping", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--notes", default="")
    parser.add_argument("--out_json", default="")
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--capture_csv", default="")
    parser.add_argument("--max_capture_values", type=int, default=256)
    parser.add_argument("--imagenet_val", default="")
    parser.add_argument("--imagenet_manifest", default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--resize_size", type=int, default=None)
    parser.add_argument("--center_crop_size", type=int, default=None)
    parser.add_argument("--input_color_order", choices=["rgb", "bgr"], default="rgb")
    parser.add_argument("--data_color_order", choices=["rgb", "bgr"], default="bgr")
    parser.add_argument("--input_scale", type=float, default=1.0)
    parser.add_argument("--enable_mean_std", action="store_true")
    parser.add_argument("--mean_std_mean", default="0.485,0.456,0.406")
    parser.add_argument("--mean_std_std", default="0.229,0.224,0.225")
    parser.add_argument("--weights_npz", default=None)
    parser.add_argument("--weights_dir", default=None)
    parser.add_argument("--weights_override", default=None)
    parser.add_argument("--mlx_weights_dir", default=None)
    parser.add_argument("--force_reexport_mlx_weights", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.mode == "request":
        payload = build_request(args)
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        if args.out_json:
            out_path = Path(args.out_json).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(rendered + "\n", encoding="utf-8")
            print(out_path)
            return
        print(rendered)
        return

    if not args.capture_csv:
        raise SystemExit("--capture_csv is required in execute mode.")
    summary = execute_bitstream_slice(args)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    if args.summary_json:
        out_path = Path(args.summary_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
        print(out_path)
        return
    print(rendered)


if __name__ == "__main__":
    main()
