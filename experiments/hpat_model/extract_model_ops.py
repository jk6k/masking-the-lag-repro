"""Extract per-layer GEMM and elementwise shapes for HPAT modeling."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from exp_common.cvnets_utils import build_opts, resolve_config_path, resolve_cvnets_root  # noqa: E402
from exp_common.io_utils import backup_existing_file  # noqa: E402
from exp_common.model_specs import (  # noqa: E402
    MODEL_SPECS,
    classification_override_kwargs,
    parse_model_keys,
)

os.environ["CVNETS_TOLERATE_IMPORT_ERRORS"] = "1"


def build_model(
    model_key: str,
    batch_size: int,
    cvnets_root: Path,
    config_path: str | None,
) -> tuple[torch.nn.Module, torch.Tensor, int]:
    """Build a supported classification model + dummy input for shape tracing."""
    spec = MODEL_SPECS[model_key]
    cfg_path = str(config_path) if config_path else None
    overrides = classification_override_kwargs(model_key, weights_path=None)
    overrides["dev.device"] = "cpu"
    opts = build_opts(cfg_path, overrides)

    from cvnets.models import get_model

    model = get_model(opts)
    model.eval()

    input_size = spec["input_size"]
    dummy = torch.randn(batch_size, 3, input_size, input_size)
    return model, dummy, input_size


def collect_ops(model: torch.nn.Module, dummy: torch.Tensor) -> list[dict]:
    """Collect GEMM and elementwise ops by hooking layer forward passes."""
    ops = []
    hooks = []
    try:
        from cvnets.layers.linear_layer import LinearLayer
    except Exception:
        LinearLayer = None
    try:
        from cvnets.layers.multi_head_attention import MultiHeadAttention
    except Exception:
        MultiHeadAttention = None

    norm_classes = (
        torch.nn.LayerNorm,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
    )
    activation_classes = (
        torch.nn.SiLU,
        torch.nn.ReLU,
        torch.nn.ReLU6,
        torch.nn.GELU,
        torch.nn.Sigmoid,
        torch.nn.Tanh,
        torch.nn.Hardswish,
        torch.nn.Hardtanh,
        torch.nn.LeakyReLU,
    )

    def _append_elementwise(op_type: str, output: object, name: str) -> None:
        # Count total elements for per-element electronic estimates.
        if torch.is_tensor(output):
            elements = output.numel()
        elif isinstance(output, (list, tuple)):
            elements = sum(t.numel() for t in output if torch.is_tensor(t))
        else:
            return
        if elements <= 0:
            return
        ops.append({"name": name, "type": op_type, "elements": int(elements)})

    def register_hook(module: torch.nn.Module, name: str) -> None:
        if isinstance(module, torch.nn.Conv2d):

            def conv_hook(mod, inputs, output) -> None:
                x = inputs[0]
                y = output
                if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                    return
                # Map Conv2d to GEMM with shape (m x d) * (d x n)
                # m = batch * H_out * W_out, d = (C_in/groups)*K_h*K_w, n = C_out
                batch, c_in, _, _ = x.shape
                _, c_out, h_out, w_out = y.shape
                k_h, k_w = mod.kernel_size
                groups = mod.groups or 1
                m = batch * h_out * w_out
                d = (c_in // groups) * k_h * k_w
                n = c_out
                ops.append(
                    {
                        "name": name,
                        "type": "conv2d",
                        "m": int(m),
                        "d": int(d),
                        "n": int(n),
                        "kernel": [int(k_h), int(k_w)],
                        "stride": list(mod.stride),
                        "groups": int(groups),
                    }
                )

            hooks.append(module.register_forward_hook(conv_hook))

        if LinearLayer is not None and isinstance(module, LinearLayer):

            def cvnets_linear_hook(mod, inputs, output) -> None:
                x = inputs[0]
                y = output
                if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                    return
                # LinearLayer maps to GEMM with m = batch*..., d = in_features, n = out_features
                in_features = mod.in_features
                out_features = mod.out_features
                m = int(x.numel() / in_features)
                d = int(in_features)
                n = int(out_features)
                ops.append(
                    {
                        "name": name,
                        "type": "linear",
                        "m": m,
                        "d": d,
                        "n": n,
                    }
                )

            hooks.append(module.register_forward_hook(cvnets_linear_hook))

        if isinstance(module, torch.nn.Linear):

            def linear_hook(mod, inputs, output) -> None:
                x = inputs[0]
                y = output
                if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                    return
                # Standard Linear GEMM: (m x d) * (d x n)
                in_features = x.shape[-1]
                out_features = y.shape[-1]
                m = int(x.numel() / in_features)
                d = int(in_features)
                n = int(out_features)
                ops.append(
                    {
                        "name": name,
                        "type": "linear",
                        "m": m,
                        "d": d,
                        "n": n,
                    }
                )

            hooks.append(module.register_forward_hook(linear_hook))

        if MultiHeadAttention is not None and isinstance(module, MultiHeadAttention):

            def attn_hook(mod, inputs, output) -> None:
                if not inputs:
                    return
                x_q = inputs[0]
                x_kv = inputs[1] if len(inputs) > 1 else None
                if not torch.is_tensor(x_q):
                    return
                batch, s_len, _ = x_q.shape
                t_len = x_kv.shape[1] if torch.is_tensor(x_kv) else s_len
                num_heads = int(mod.num_heads)
                head_dim = int(mod.head_dim)
                m = batch * num_heads * s_len
                # QK^T: (m x head_dim) * (head_dim x t_len)
                ops.append(
                    {
                        "name": f"{name}.attn_qk",
                        "type": "attn_qk",
                        "m": int(m),
                        "d": head_dim,
                        "n": int(t_len),
                    }
                )
                # AV: (m x t_len) * (t_len x head_dim)
                ops.append(
                    {
                        "name": f"{name}.attn_av",
                        "type": "attn_av",
                        "m": int(m),
                        "d": int(t_len),
                        "n": head_dim,
                    }
                )
                # Softmax is modeled as elementwise over all attention scores.
                softmax_elements = batch * num_heads * s_len * t_len
                ops.append(
                    {
                        "name": f"{name}.softmax",
                        "type": "softmax",
                        "elements": int(softmax_elements),
                    }
                )

            hooks.append(module.register_forward_hook(attn_hook))

        if isinstance(module, norm_classes):

            def norm_hook(mod, inputs, output) -> None:
                _append_elementwise("norm", output, name)

            hooks.append(module.register_forward_hook(norm_hook))

        if isinstance(module, activation_classes):

            def act_hook(mod, inputs, output) -> None:
                _append_elementwise("activation", output, name)

            hooks.append(module.register_forward_hook(act_hook))

    for name, module in model.named_modules():
        register_hook(module, name)

    with torch.inference_mode():
        _ = model(dummy)

    for h in hooks:
        h.remove()

    return ops


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Conv/Linear GEMM shapes for HPAT modeling."
    )
    parser.add_argument(
        "--cvnets_dir",
        default=None,
        help="Path to a local ml-cvnets clone.",
    )
    parser.add_argument(
        "--models",
        default="mobilevit_xxs,mobilevit_xs,mobilevit_s",
        help="Comma-separated MobileViT model keys.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config YAML (defaults to ml-cvnets config when available).",
    )
    parser.add_argument(
        "--out_dir",
        default="ops",
        help="Output directory for ops JSON files.",
    )
    args = parser.parse_args()

    cvnets_root = resolve_cvnets_root(args.cvnets_dir)
    config_path = resolve_config_path(args.config, cvnets_root)

    models = parse_model_keys(args.models)
    for model_key in models:
        if model_key not in MODEL_SPECS:
            raise SystemExit(f"Unsupported model: {model_key}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_key in models:
        model, dummy, input_size = build_model(
            model_key, args.batch_size, cvnets_root, config_path
        )
        ops = collect_ops(model, dummy)
        payload = {
            "model": model_key,
            "input_size": input_size,
            "batch_size": args.batch_size,
            "note": "Includes Conv2d/Linear, attention QK/AV, softmax, norm, and activation estimates.",
            "ops": ops,
        }
        out_path = out_dir / f"ops_{model_key}.json"
        backup = backup_existing_file(out_path)
        if backup:
            print(f"Existing ops file moved to {backup}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
