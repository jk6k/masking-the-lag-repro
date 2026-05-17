#!/usr/bin/env python3
"""Audit PyTorch-versus-MLX MobileViT parity on Apple Silicon."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = ROOT / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from exp_common.cvnets_utils import (  # noqa: E402
    build_opts,
    parse_weights_override,
    resolve_config_path,
    resolve_cvnets_root,
    resolve_weights_path,
)
from exp_common.model_specs import MODEL_SPECS, classification_override_kwargs  # noqa: E402
from exp_common.runtime import apply_torch_device, prefer_vision_channels_last  # noqa: E402


STAGE_ORDER = [
    "conv_1",
    "layer_1",
    "layer_2",
    "layer_3",
    "layer_4",
    "layer_5",
    "conv_1x1_exp",
    "pool",
    "logits",
]
STAGE_PROBE_PREFIX = "stages."
DETAIL_STAGE_PROBE_PREFIX = "detail_block.stages."
TOP_DETAIL_STAGE_PROBE_PREFIX = "detail_stage.stages."
DETAIL_STAGE_CHOICES = ("conv_1x1_exp", "logits")


def _diff_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float | list[int]]:
    diff = np.abs(a - b)
    return {
        "shape": list(a.shape),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
    }


def _parse_block_ref(raw: str | None) -> tuple[str, int] | None:
    if not raw:
        return None
    value = str(raw).strip()
    parts = value.split(".")
    if len(parts) != 2:
        raise SystemExit(
            f"--detail_block must look like layer_5.1, received: {raw}"
        )
    layer_name, block_idx_raw = parts
    if layer_name not in {"layer_3", "layer_4", "layer_5"}:
        raise SystemExit(
            f"--detail_block only supports MobileViT blocks in layer_3/layer_4/layer_5, received: {raw}"
        )
    return layer_name, int(block_idx_raw)


def _parse_stage_threshold(raw: str) -> tuple[str, float]:
    value = str(raw or "").strip()
    if "=" not in value:
        raise SystemExit(
            f"--max_stage_max_abs must look like logits=0.6, received: {raw}"
        )
    stage_name, threshold_raw = value.split("=", 1)
    stage_name = stage_name.strip()
    if stage_name not in STAGE_ORDER:
        raise SystemExit(
            f"--max_stage_max_abs only supports stages {STAGE_ORDER}, received: {stage_name}"
        )
    try:
        threshold = float(threshold_raw)
    except ValueError as exc:
        raise SystemExit(
            f"--max_stage_max_abs threshold must be numeric, received: {raw}"
        ) from exc
    return stage_name, threshold


def _parse_probe_ref(raw: str | None) -> tuple[str, str] | None:
    if not raw:
        return None
    value = str(raw).strip()
    if value.startswith(STAGE_PROBE_PREFIX):
        stage_name = value[len(STAGE_PROBE_PREFIX) :]
        if stage_name not in STAGE_ORDER:
            raise SystemExit(
                f"--probe_ref stage probe must target one of {STAGE_ORDER}, received: {raw}"
            )
        return "stages", stage_name
    if value.startswith(DETAIL_STAGE_PROBE_PREFIX):
        stage_name = value[len(DETAIL_STAGE_PROBE_PREFIX) :]
        if not stage_name:
            raise SystemExit(
                f"--probe_ref detail probe must include an op name, received: {raw}"
            )
        return "detail_block.stages", stage_name
    if value.startswith(TOP_DETAIL_STAGE_PROBE_PREFIX):
        stage_name = value[len(TOP_DETAIL_STAGE_PROBE_PREFIX) :]
        if not stage_name:
            raise SystemExit(
                f"--probe_ref detail-stage probe must include an op name, received: {raw}"
            )
        return "detail_stage.stages", stage_name
    raise SystemExit(
        "--probe_ref must look like stages.logits or "
        "detail_block.stages.local_rep_conv_1x1 or detail_stage.stages.pool"
    )


def _nchw_to_nhwc(array: np.ndarray) -> np.ndarray:
    return np.transpose(array, (0, 2, 3, 1))


def _torch_local_rep_steps(pt_block, pt):
    local_rep = pt_block.local_rep
    modules = []
    if hasattr(local_rep, "__getitem__"):
        try:
            modules = [local_rep[0], local_rep[1]]
        except Exception:
            modules = []
    if not modules and hasattr(local_rep, "children"):
        modules = list(local_rep.children())[:2]
    if len(modules) < 2:
        raise SystemExit(
            "Unable to resolve PyTorch local_rep conv_3x3/conv_1x1 modules for parity audit."
        )
    pt_local_3x3 = modules[0](pt)
    pt_local_1x1 = modules[1](pt_local_3x3)
    return pt_local_3x3, pt_local_1x1


def _torch_to_numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _mlx_to_numpy(tensor) -> np.ndarray:
    return np.array(tensor)


def _torch_attention_steps(torch, attn_module, x_q):
    batch_size, src_len, _ = x_q.shape
    qkv_proj = attn_module.qkv_proj(x_q)
    qkv = qkv_proj.reshape(batch_size, src_len, 3, attn_module.num_heads, -1)
    qkv = qkv.transpose(1, 3).contiguous()
    query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    query = query * attn_module.scaling
    attn_scores = torch.matmul(query, key.transpose(-1, -2))
    attn_softmax = attn_module.softmax(attn_scores.float()).to(attn_scores.dtype)
    attn_softmax = attn_module.attn_dropout(attn_softmax)
    attn_ctx = torch.matmul(attn_softmax, value)
    attn_ctx = attn_ctx.transpose(1, 2).reshape(batch_size, src_len, -1)
    attn_out_proj = attn_module.out_proj(attn_ctx)
    return {
        "qkv_proj": qkv_proj,
        "attn_scores": attn_scores,
        "attn_softmax": attn_softmax,
        "attn_ctx": attn_ctx,
        "attn_out_proj": attn_out_proj,
    }


def _mlx_attention_steps(mx, attn_module, x_q):
    batch_size, src_len, _ = x_q.shape
    qkv_proj = attn_module.qkv_proj(x_q)
    qkv = qkv_proj.reshape(
        batch_size,
        src_len,
        3,
        attn_module.num_heads,
        attn_module.head_dim,
    )
    qkv = mx.transpose(qkv, (0, 3, 2, 1, 4))
    query = qkv[:, :, 0]
    key = qkv[:, :, 1]
    value = qkv[:, :, 2]
    query = query * attn_module.scaling
    attn_scores = mx.matmul(query, mx.transpose(key, (0, 1, 3, 2)))
    attn_softmax = mx.softmax(attn_scores, axis=-1)
    attn_ctx = mx.matmul(attn_softmax, value)
    attn_ctx = mx.transpose(attn_ctx, (0, 2, 1, 3)).reshape(
        batch_size,
        src_len,
        attn_module.embed_dim,
    )
    attn_out_proj = attn_module.out_proj(attn_ctx)
    return {
        "qkv_proj": qkv_proj,
        "attn_scores": attn_scores,
        "attn_softmax": attn_softmax,
        "attn_ctx": attn_ctx,
        "attn_out_proj": attn_out_proj,
    }


def _transformer_encoder_step_parity(
    *,
    torch,
    mx,
    torch_encoder,
    mlx_encoder,
    torch_tokens,
    mlx_tokens,
) -> tuple[dict[str, dict[str, float | list[int]]], Any, Any]:
    residual_pt = torch_tokens
    residual_mlx = mlx_tokens

    pt_ln1 = torch_encoder.pre_norm_mha[0](torch_tokens)
    mlx_ln1 = mlx_encoder.pre_norm_mha[0](mlx_tokens)
    pt_attn = _torch_attention_steps(torch, torch_encoder.pre_norm_mha[1], pt_ln1)
    mlx_attn = _mlx_attention_steps(mx, mlx_encoder.pre_norm_mha[1], mlx_ln1)
    pt_after_attn = residual_pt + torch_encoder.pre_norm_mha[2](pt_attn["attn_out_proj"])
    mlx_after_attn = residual_mlx + mlx_encoder.pre_norm_mha[2](mlx_attn["attn_out_proj"])

    pt_ln2 = torch_encoder.pre_norm_ffn[0](pt_after_attn)
    mlx_ln2 = mlx_encoder.pre_norm_ffn[0](mlx_after_attn)
    pt_ffn_fc1 = torch_encoder.pre_norm_ffn[1](pt_ln2)
    mlx_ffn_fc1 = mlx_encoder.pre_norm_ffn[1](mlx_ln2)
    pt_ffn_act = torch_encoder.pre_norm_ffn[2](pt_ffn_fc1)
    mlx_ffn_act = mlx_encoder.pre_norm_ffn[2](mlx_ffn_fc1)
    pt_ffn_dropout1 = torch_encoder.pre_norm_ffn[3](pt_ffn_act)
    mlx_ffn_dropout1 = mlx_encoder.pre_norm_ffn[3](mlx_ffn_act)
    pt_ffn_fc2 = torch_encoder.pre_norm_ffn[4](pt_ffn_dropout1)
    mlx_ffn_fc2 = mlx_encoder.pre_norm_ffn[4](mlx_ffn_dropout1)
    pt_ffn_dropout2 = torch_encoder.pre_norm_ffn[5](pt_ffn_fc2)
    mlx_ffn_dropout2 = mlx_encoder.pre_norm_ffn[5](mlx_ffn_fc2)
    pt_after_ffn = pt_after_attn + pt_ffn_dropout2
    mlx_after_ffn = mlx_after_attn + mlx_ffn_dropout2

    report = {
        "ln1": _diff_stats(_torch_to_numpy(pt_ln1), _mlx_to_numpy(mlx_ln1)),
        "qkv_proj": _diff_stats(
            _torch_to_numpy(pt_attn["qkv_proj"]),
            _mlx_to_numpy(mlx_attn["qkv_proj"]),
        ),
        "attn_scores": _diff_stats(
            _torch_to_numpy(pt_attn["attn_scores"]),
            _mlx_to_numpy(mlx_attn["attn_scores"]),
        ),
        "attn_softmax": _diff_stats(
            _torch_to_numpy(pt_attn["attn_softmax"]),
            _mlx_to_numpy(mlx_attn["attn_softmax"]),
        ),
        "attn_ctx": _diff_stats(
            _torch_to_numpy(pt_attn["attn_ctx"]),
            _mlx_to_numpy(mlx_attn["attn_ctx"]),
        ),
        "attn_out_proj": _diff_stats(
            _torch_to_numpy(pt_attn["attn_out_proj"]),
            _mlx_to_numpy(mlx_attn["attn_out_proj"]),
        ),
        "after_attn_residual": _diff_stats(
            _torch_to_numpy(pt_after_attn),
            _mlx_to_numpy(mlx_after_attn),
        ),
        "ln2": _diff_stats(_torch_to_numpy(pt_ln2), _mlx_to_numpy(mlx_ln2)),
        "ffn_fc1": _diff_stats(
            _torch_to_numpy(pt_ffn_fc1),
            _mlx_to_numpy(mlx_ffn_fc1),
        ),
        "ffn_act": _diff_stats(
            _torch_to_numpy(pt_ffn_act),
            _mlx_to_numpy(mlx_ffn_act),
        ),
        "ffn_fc2": _diff_stats(
            _torch_to_numpy(pt_ffn_fc2),
            _mlx_to_numpy(mlx_ffn_fc2),
        ),
        "after_ffn_residual": _diff_stats(
            _torch_to_numpy(pt_after_ffn),
            _mlx_to_numpy(mlx_after_ffn),
        ),
    }
    return report, pt_after_ffn, mlx_after_ffn


def _stats_max_abs(stats: Any, *, ref: str) -> float:
    if not isinstance(stats, dict) or "max_abs" not in stats:
        raise SystemExit(f"Probe {ref} does not resolve to a max_abs-bearing stats block.")
    return float(stats["max_abs"])


def _lookup_probe_stats(
    *,
    stage_report: dict[str, dict[str, float | list[int]]],
    detail_block_report: dict[str, Any] | None,
    detail_stage_report: dict[str, Any] | None,
    probe_ref: tuple[str, str],
) -> dict[str, float | list[int]]:
    domain, key = probe_ref
    if domain == "stages":
        stats = stage_report.get(key)
    elif domain == "detail_block.stages":
        if detail_block_report is None:
            raise SystemExit(
                f"Probe {DETAIL_STAGE_PROBE_PREFIX}{key} requires --detail_block."
            )
        stats = detail_block_report.get(key)
    elif domain == "detail_stage.stages":
        if detail_stage_report is None:
            raise SystemExit(
                f"Probe {TOP_DETAIL_STAGE_PROBE_PREFIX}{key} requires --detail_stage."
            )
        stats = detail_stage_report.get(key)
    else:
        raise SystemExit(f"Unsupported probe domain: {domain}")
    if not isinstance(stats, dict) or "max_abs" not in stats:
        raise SystemExit(f"Probe target not found or not numeric: {domain}.{key}")
    return stats


def _evaluate_thresholds(
    *,
    stage_report: dict[str, dict[str, float | list[int]]],
    stage_thresholds: dict[str, float],
    detail_block_report: dict[str, Any] | None = None,
    detail_stage_report: dict[str, Any] | None = None,
    probe_ref: tuple[str, str] | None = None,
    max_probe_max_abs: float | None = None,
) -> dict[str, Any]:
    if max_probe_max_abs is not None and probe_ref is None:
        raise SystemExit("--max_probe_max_abs requires --probe_ref.")

    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for stage_name, threshold in stage_thresholds.items():
        observed = _stats_max_abs(
            stage_report.get(stage_name),
            ref=f"{STAGE_PROBE_PREFIX}{stage_name}",
        )
        passed = observed <= float(threshold)
        check = {
            "ref": f"{STAGE_PROBE_PREFIX}{stage_name}",
            "observed_max_abs": observed,
            "max_allowed": float(threshold),
            "passed": passed,
        }
        checks.append(check)
        if not passed:
            failures.append(check)

    if probe_ref is not None:
        probe_stats = _lookup_probe_stats(
            stage_report=stage_report,
            detail_block_report=detail_block_report,
            detail_stage_report=detail_stage_report,
            probe_ref=probe_ref,
        )
        if max_probe_max_abs is not None:
            domain, key = probe_ref
            ref = f"{domain}.{key}"
            observed = _stats_max_abs(probe_stats, ref=ref)
            passed = observed <= float(max_probe_max_abs)
            check = {
                "ref": ref,
                "observed_max_abs": observed,
                "max_allowed": float(max_probe_max_abs),
                "passed": passed,
            }
            checks.append(check)
            if not passed:
                failures.append(check)

    return {
        "passed": not failures,
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
    }


def _resolve_stage_thresholds(args: argparse.Namespace) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    explicit_thresholds = {
        "logits": args.max_logits_max_abs,
        "conv_1x1_exp": args.max_conv1x1_exp_max_abs,
        "pool": args.max_pool_max_abs,
        "layer_5": args.max_layer_5_max_abs,
    }
    for stage_name, threshold in explicit_thresholds.items():
        if threshold is None:
            continue
        thresholds[stage_name] = float(threshold)
    for raw in args.max_stage_max_abs or []:
        stage_name, threshold = _parse_stage_threshold(raw)
        thresholds[stage_name] = threshold
    return thresholds


def _setup_runtime_imports():
    os.environ.setdefault("CVNETS_TOLERATE_IMPORT_ERRORS", "1")
    os.environ.setdefault("CVNETS_SUPPRESS_OPTIONAL_IMPORT_WARNINGS", "1")
    os.environ.setdefault("CVNETS_LOG_TRAINABLE_PARAMETERS", "0")

    import torch
    import mlx.core as mx
    from accuracy.mlx_mobilevit import build_mlx_mobilevit

    return torch, mx, build_mlx_mobilevit


def _build_torch_model(
    *,
    model_key: str,
    weights_dir: str | None,
    weights_override: str | None,
):
    torch, _mx, _build_mlx_mobilevit = _setup_runtime_imports()
    cvnets_root = resolve_cvnets_root(None)
    from cvnets.models import get_model

    config_path = resolve_config_path(None, cvnets_root)
    weights_override_map = parse_weights_override(weights_override)
    weights_path = resolve_weights_path(
        model_key,
        weights_dir,
        weights_override_map,
    )
    overrides = classification_override_kwargs(model_key, weights_path=weights_path)
    opts = build_opts(config_path, overrides)
    apply_torch_device(opts, "mps")
    model = get_model(opts)
    use_channels_last = prefer_vision_channels_last("mps", False)
    memory_format = (
        torch.channels_last if use_channels_last else torch.contiguous_format
    )
    model = model.to(device=torch.device("mps"), memory_format=memory_format)
    model.eval()
    return model, use_channels_last


def _build_mlx_model(
    *,
    model_key: str,
    weights_dir: str | None,
    weights_override: str | None,
    weights_npz: str | None,
    mlx_weights_dir: str | None,
    force_reexport: bool,
):
    _torch, _mx, build_mlx_mobilevit = _setup_runtime_imports()
    return build_mlx_mobilevit(
        model_key,
        weights_npz=weights_npz,
        weights_dir=weights_dir,
        weights_override=weights_override,
        mlx_weights_dir=mlx_weights_dir,
        force_reexport=force_reexport,
    )


def _stage_parity(
    *,
    model_key: str,
    batch_size: int,
    seed: int,
    weights_dir: str | None,
    weights_override: str | None,
    weights_npz: str | None,
    mlx_weights_dir: str | None,
    force_reexport: bool,
) -> tuple[dict[str, dict[str, float | list[int]]], dict[str, Any]]:
    torch, mx, _build_mlx_mobilevit = _setup_runtime_imports()
    torch_model, use_channels_last = _build_torch_model(
        model_key=model_key,
        weights_dir=weights_dir,
        weights_override=weights_override,
    )
    mlx_model, resolved_weights_npz = _build_mlx_model(
        model_key=model_key,
        weights_dir=weights_dir,
        weights_override=weights_override,
        weights_npz=weights_npz,
        mlx_weights_dir=mlx_weights_dir,
        force_reexport=force_reexport,
    )

    input_size = int(MODEL_SPECS[model_key]["input_size"])
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal((batch_size, 3, input_size, input_size), dtype=np.float32)
    x_pt = torch.from_numpy(x_np).to(torch.device("mps"))
    if use_channels_last:
        x_pt = x_pt.contiguous(memory_format=torch.channels_last)
    x_mlx = mx.array(_nchw_to_nhwc(x_np))

    stage_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with torch.inference_mode():
        pt = torch_model.conv_1(x_pt)
        mlx = mlx_model.conv_1(x_mlx)
        stage_outputs["conv_1"] = (_nchw_to_nhwc(pt.detach().cpu().numpy()), np.array(mlx))

        for stage_name in ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5"]:
            for layer in getattr(torch_model, stage_name):
                pt = layer(pt)
            for layer in getattr(mlx_model, stage_name):
                mlx = layer(mlx)
            stage_outputs[stage_name] = (
                _nchw_to_nhwc(pt.detach().cpu().numpy()),
                np.array(mlx),
            )

        pt = torch_model.conv_1x1_exp(pt)
        mlx = mlx_model.conv_1x1_exp(mlx)
        stage_outputs["conv_1x1_exp"] = (
            _nchw_to_nhwc(pt.detach().cpu().numpy()),
            np.array(mlx),
        )

        pt_pool = pt.mean(dim=(2, 3))
        mlx_pool = mx.mean(mlx, axis=(1, 2))
        stage_outputs["pool"] = (
            pt_pool.detach().cpu().numpy(),
            np.array(mlx_pool),
        )

        pt_logits = torch_model(x_pt)
        mlx_logits = mlx_model(x_mlx)
        stage_outputs["logits"] = (
            pt_logits.detach().cpu().numpy(),
            np.array(mlx_logits),
        )

    report = {
        stage_name: _diff_stats(*stage_outputs[stage_name]) for stage_name in STAGE_ORDER
    }
    metadata = {
        "model": model_key,
        "batch_size": batch_size,
        "seed": seed,
        "input_size": input_size,
        "weights_npz": str(resolved_weights_npz),
    }
    return report, metadata


def _detail_block_parity(
    *,
    model_key: str,
    block_ref: tuple[str, int],
    batch_size: int,
    seed: int,
    weights_dir: str | None,
    weights_override: str | None,
    weights_npz: str | None,
    mlx_weights_dir: str | None,
    force_reexport: bool,
) -> dict[str, dict[str, float | list[int]] | bool]:
    torch, mx, _build_mlx_mobilevit = _setup_runtime_imports()
    torch_model, use_channels_last = _build_torch_model(
        model_key=model_key,
        weights_dir=weights_dir,
        weights_override=weights_override,
    )
    mlx_model, _resolved_weights_npz = _build_mlx_model(
        model_key=model_key,
        weights_dir=weights_dir,
        weights_override=weights_override,
        weights_npz=weights_npz,
        mlx_weights_dir=mlx_weights_dir,
        force_reexport=force_reexport,
    )

    input_size = int(MODEL_SPECS[model_key]["input_size"])
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal((batch_size, 3, input_size, input_size), dtype=np.float32)
    x_pt = torch.from_numpy(x_np).to(torch.device("mps"))
    if use_channels_last:
        x_pt = x_pt.contiguous(memory_format=torch.channels_last)
    x_mlx = mx.array(_nchw_to_nhwc(x_np))

    layer_name, block_idx = block_ref
    with torch.inference_mode():
        pt = x_pt
        mlx = x_mlx
        for stage_name in ["conv_1", "layer_1", "layer_2", "layer_3", "layer_4", "layer_5"]:
            if stage_name == "conv_1":
                pt = torch_model.conv_1(pt)
                mlx = mlx_model.conv_1(mlx)
            else:
                layers_pt = getattr(torch_model, stage_name)
                layers_mlx = getattr(mlx_model, stage_name)
                if stage_name == layer_name:
                    for idx in range(block_idx):
                        pt = layers_pt[idx](pt)
                        mlx = layers_mlx[idx](mlx)
                    break
                for idx in range(len(layers_pt)):
                    pt = layers_pt[idx](pt)
                    mlx = layers_mlx[idx](mlx)

        pt_block = getattr(torch_model, layer_name)[block_idx]
        mlx_block = getattr(mlx_model, layer_name)[block_idx]
        res_pt = pt
        res_mlx = mlx

        pt_local_3x3, pt_local_1x1 = _torch_local_rep_steps(pt_block, pt)
        mlx_local_3x3 = mlx_block.local_rep["conv_3x3"](mlx)
        mlx_local_1x1 = mlx_block.local_rep["conv_1x1"](mlx_local_3x3)
        pt_patches, pt_info = pt_block.unfolding(pt_local_1x1)
        mlx_patches, mlx_info = mlx_block.unfolding(mlx_local_1x1)

        report: dict[str, dict[str, float | list[int]] | bool] = {
            "residual_input": _diff_stats(
                _nchw_to_nhwc(_torch_to_numpy(res_pt)),
                _mlx_to_numpy(res_mlx),
            ),
            "local_rep_conv_3x3": _diff_stats(
                _nchw_to_nhwc(_torch_to_numpy(pt_local_3x3)),
                _mlx_to_numpy(mlx_local_3x3),
            ),
            "local_rep_conv_1x1": _diff_stats(
                _nchw_to_nhwc(_torch_to_numpy(pt_local_1x1)),
                _mlx_to_numpy(mlx_local_1x1),
            ),
            "local_rep": _diff_stats(
                _nchw_to_nhwc(_torch_to_numpy(pt_local_1x1)),
                _mlx_to_numpy(mlx_local_1x1),
            ),
            "patches0": _diff_stats(
                _torch_to_numpy(pt_patches),
                _mlx_to_numpy(mlx_patches),
            ),
            "info_equal": pt_info == mlx_info,
        }

        for idx, layer in enumerate(pt_block.global_rep):
            mlx_layer = mlx_block.global_rep[idx]
            if hasattr(layer, "pre_norm_mha") and hasattr(layer, "pre_norm_ffn"):
                layer_report, pt_patches, mlx_patches = _transformer_encoder_step_parity(
                    torch=torch,
                    mx=mx,
                    torch_encoder=layer,
                    mlx_encoder=mlx_layer,
                    torch_tokens=pt_patches,
                    mlx_tokens=mlx_patches,
                )
                for step_name, stats in layer_report.items():
                    report[f"global_rep_{idx}.{step_name}"] = stats
            else:
                pt_patches = layer(pt_patches)
                mlx_patches = mlx_layer(mlx_patches)
            report[f"global_rep_{idx}"] = _diff_stats(
                _torch_to_numpy(pt_patches),
                _mlx_to_numpy(mlx_patches),
            )

        pt_fold = pt_block.folding(pt_patches, pt_info)
        mlx_fold = mlx_block.folding(mlx_patches, mlx_info)
        report["fold"] = _diff_stats(
            _nchw_to_nhwc(_torch_to_numpy(pt_fold)),
            _mlx_to_numpy(mlx_fold),
        )

        pt_proj = pt_block.conv_proj(pt_fold)
        mlx_proj = mlx_block.conv_proj(mlx_fold)
        report["conv_proj"] = _diff_stats(
            _nchw_to_nhwc(_torch_to_numpy(pt_proj)),
            _mlx_to_numpy(mlx_proj),
        )

        if pt_block.fusion is not None and mlx_block.fusion is not None:
            pt_fused = pt_block.fusion(torch.cat((res_pt, pt_proj), dim=1))
            mlx_fused = mlx_block.fusion(mx.concatenate([res_mlx, mlx_proj], axis=-1))
            report["fusion_out"] = _diff_stats(
                _nchw_to_nhwc(_torch_to_numpy(pt_fused)),
                _mlx_to_numpy(mlx_fused),
            )

    return report


def _detail_stage_parity(
    *,
    model_key: str,
    detail_stage: str,
    batch_size: int,
    seed: int,
    weights_dir: str | None,
    weights_override: str | None,
    weights_npz: str | None,
    mlx_weights_dir: str | None,
    force_reexport: bool,
) -> dict[str, dict[str, float | list[int]]]:
    torch, mx, _build_mlx_mobilevit = _setup_runtime_imports()
    torch_model, use_channels_last = _build_torch_model(
        model_key=model_key,
        weights_dir=weights_dir,
        weights_override=weights_override,
    )
    mlx_model, _resolved_weights_npz = _build_mlx_model(
        model_key=model_key,
        weights_dir=weights_dir,
        weights_override=weights_override,
        weights_npz=weights_npz,
        mlx_weights_dir=mlx_weights_dir,
        force_reexport=force_reexport,
    )

    input_size = int(MODEL_SPECS[model_key]["input_size"])
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal((batch_size, 3, input_size, input_size), dtype=np.float32)
    x_pt = torch.from_numpy(x_np).to(torch.device("mps"))
    if use_channels_last:
        x_pt = x_pt.contiguous(memory_format=torch.channels_last)
    x_mlx = mx.array(_nchw_to_nhwc(x_np))

    with torch.inference_mode():
        pt = torch_model.conv_1(x_pt)
        mlx = mlx_model.conv_1(x_mlx)
        for stage_name in ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5"]:
            for layer in getattr(torch_model, stage_name):
                pt = layer(pt)
            for layer in getattr(mlx_model, stage_name):
                mlx = layer(mlx)

        if detail_stage == "conv_1x1_exp":
            pt_conv = torch_model.conv_1x1_exp.block.conv(pt)
            mlx_conv = mlx_model.conv_1x1_exp.block["conv"](mlx)
            pt_norm = torch_model.conv_1x1_exp.block.norm(pt_conv)
            mlx_norm = mlx_model.conv_1x1_exp.block["norm"](mlx_conv)
            pt_act = torch_model.conv_1x1_exp.block.act(pt_norm)
            mlx_act = mlx_model.conv_1x1_exp.act(mlx_norm)
            return {
                "input": _diff_stats(_nchw_to_nhwc(_torch_to_numpy(pt)), _mlx_to_numpy(mlx)),
                "conv": _diff_stats(_nchw_to_nhwc(_torch_to_numpy(pt_conv)), _mlx_to_numpy(mlx_conv)),
                "norm": _diff_stats(_nchw_to_nhwc(_torch_to_numpy(pt_norm)), _mlx_to_numpy(mlx_norm)),
                "act": _diff_stats(_nchw_to_nhwc(_torch_to_numpy(pt_act)), _mlx_to_numpy(mlx_act)),
            }

        pt_features = torch_model.conv_1x1_exp(pt)
        mlx_features = mlx_model.conv_1x1_exp(mlx)
        pt_pool = torch_model.classifier.global_pool(pt_features)
        mlx_pool = mx.mean(mlx_features, axis=(1, 2))
        pt_logits = torch_model.classifier.fc(pt_pool)
        mlx_logits = mlx_model.classifier["fc"](mlx_pool)
        return {
            "pre_pool": _diff_stats(
                _nchw_to_nhwc(_torch_to_numpy(pt_features)),
                _mlx_to_numpy(mlx_features),
            ),
            "pool": _diff_stats(_torch_to_numpy(pt_pool), _mlx_to_numpy(mlx_pool)),
            "classifier_fc": _diff_stats(
                _torch_to_numpy(pt_logits),
                _mlx_to_numpy(mlx_logits),
            ),
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit PyTorch-versus-MLX MobileViT parity on MPS.",
    )
    parser.add_argument(
        "--model",
        default="mobilevit_xxs",
        choices=tuple(MODEL_SPECS),
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights_dir", default=None)
    parser.add_argument("--weights_override", default=None)
    parser.add_argument("--weights_npz", default=None)
    parser.add_argument("--mlx_weights_dir", default=None)
    parser.add_argument("--force_reexport_mlx_weights", action="store_true")
    parser.add_argument(
        "--detail_block",
        default=None,
        help="Optional MobileViT block reference like layer_5.1 for internal-stage diffs.",
    )
    parser.add_argument(
        "--detail_stage",
        default=None,
        choices=DETAIL_STAGE_CHOICES,
        help="Optional top-level stage detail like conv_1x1_exp or logits.",
    )
    parser.add_argument(
        "--probe_ref",
        default=None,
        help=(
            "Optional probe reference like stages.logits or "
            "detail_block.stages.local_rep_conv_1x1 or detail_stage.stages.pool."
        ),
    )
    parser.add_argument(
        "--max-stage-max-abs",
        action="append",
        default=[],
        help="Repeatable stage guardrail in the form stage=value, e.g. logits=0.6.",
    )
    parser.add_argument("--max-logits-max-abs", type=float, default=None)
    parser.add_argument("--max-conv1x1-exp-max-abs", type=float, default=None)
    parser.add_argument("--max-pool-max-abs", type=float, default=None)
    parser.add_argument("--max-layer-5-max-abs", type=float, default=None)
    parser.add_argument("--max-probe-max-abs", type=float, default=None)
    parser.add_argument("--report_json", default=None)
    args = parser.parse_args()

    if args.model not in MODEL_SPECS:
        raise SystemExit(f"Unsupported model: {args.model}")

    stage_report, metadata = _stage_parity(
        model_key=args.model,
        batch_size=max(1, int(args.batch_size)),
        seed=int(args.seed),
        weights_dir=args.weights_dir,
        weights_override=args.weights_override,
        weights_npz=args.weights_npz,
        mlx_weights_dir=args.mlx_weights_dir,
        force_reexport=bool(args.force_reexport_mlx_weights),
    )
    payload: dict[str, Any] = {
        "metadata": metadata,
        "stages": stage_report,
    }

    block_ref = _parse_block_ref(args.detail_block)
    detail_block_report: dict[str, Any] | None = None
    if block_ref is not None:
        detail_block_report = _detail_block_parity(
            model_key=args.model,
            block_ref=block_ref,
            batch_size=max(1, int(args.batch_size)),
            seed=int(args.seed),
            weights_dir=args.weights_dir,
            weights_override=args.weights_override,
            weights_npz=args.weights_npz,
            mlx_weights_dir=args.mlx_weights_dir,
            force_reexport=bool(args.force_reexport_mlx_weights),
        )
        payload["detail_block"] = {
            "ref": args.detail_block,
            "stages": detail_block_report,
        }

    detail_stage_report: dict[str, Any] | None = None
    if args.detail_stage is not None:
        detail_stage_report = _detail_stage_parity(
            model_key=args.model,
            detail_stage=args.detail_stage,
            batch_size=max(1, int(args.batch_size)),
            seed=int(args.seed),
            weights_dir=args.weights_dir,
            weights_override=args.weights_override,
            weights_npz=args.weights_npz,
            mlx_weights_dir=args.mlx_weights_dir,
            force_reexport=bool(args.force_reexport_mlx_weights),
        )
        payload["detail_stage"] = {
            "ref": args.detail_stage,
            "stages": detail_stage_report,
        }

    probe_ref = _parse_probe_ref(args.probe_ref)
    if probe_ref is not None:
        payload["probe"] = {
            "ref": args.probe_ref,
            "stats": _lookup_probe_stats(
                stage_report=stage_report,
                detail_block_report=detail_block_report,
                detail_stage_report=detail_stage_report,
                probe_ref=probe_ref,
            ),
        }

    stage_thresholds = _resolve_stage_thresholds(args)
    if stage_thresholds or args.max_probe_max_abs is not None:
        payload["guardrail"] = _evaluate_thresholds(
            stage_report=stage_report,
            stage_thresholds=stage_thresholds,
            detail_block_report=detail_block_report,
            detail_stage_report=detail_stage_report,
            probe_ref=probe_ref,
            max_probe_max_abs=args.max_probe_max_abs,
        )

    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.report_json:
        path = Path(args.report_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    guardrail = payload.get("guardrail")
    if isinstance(guardrail, dict) and guardrail.get("passed") is False:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
