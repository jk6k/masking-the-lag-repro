"""Hook utilities to inject photonic noise into MobileViT layers."""

from __future__ import annotations

import re
import types
from dataclasses import dataclass, field
from typing import Any

import torch

from accuracy.photonic_perturb import apply_perturb, compute_symmetric_scale


@dataclass
class PerturbationConfig:
    bits: int = 8
    sigma_lsb: float = 0.0  # Gaussian-noise sigma from photonic device mapping, in LSB.
    crosstalk_alpha: float = 0.0
    drift_lsb: float = 0.0
    noise_correlation: float = 0.0
    burst_error_prob: float = 0.0
    burst_error_scale_lsb: float = 0.0
    burst_span: int = 1
    enabled: bool = True
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
    calibration_mode: bool = False
    use_calibrated_scale: bool = False
    strict_calibrated_scale: bool = False
    conv_channel_dim: int = 1
    linear_channel_dim: int = -1
    attention_channel_dim: int = -1
    attention_output_channel_dim: int = -1
    scale_cache: dict[str, torch.Tensor] = field(default_factory=dict)
    sparse_activity_recorder: "SparseActivityRecorder | None" = None

    @property
    def gaussian_noise_std(self) -> float:
        """Backward-compatible alias used by the torch evaluator and hooks."""
        return float(self.sigma_lsb)

    @gaussian_noise_std.setter
    def gaussian_noise_std(self, value: float) -> None:
        self.sigma_lsb = float(value)

    @property
    def noise_sigma_lsb(self) -> float:
        return float(self.sigma_lsb)

    @noise_sigma_lsb.setter
    def noise_sigma_lsb(self, value: float) -> None:
        self.sigma_lsb = float(value)


@dataclass
class SparseActivityRecorder:
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

        module_entry = self.per_module.setdefault(
            module_key,
            {
                "gate_mode": gate_mode,
                "total_elements": 0,
                "active_elements": 0,
                "call_count": 0,
            },
        )
        module_entry["total_elements"] = int(module_entry["total_elements"]) + total
        module_entry["active_elements"] = int(module_entry["active_elements"]) + active
        module_entry["call_count"] = int(module_entry["call_count"]) + 1
        if gate_mode:
            existing_gate_mode = str(module_entry.get("gate_mode") or "").strip() or None
            if existing_gate_mode is None:
                module_entry["gate_mode"] = gate_mode
            elif existing_gate_mode != gate_mode:
                module_entry["gate_mode"] = "mixed"

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


def _compile_regex(value: str | None) -> re.Pattern | None:
    if not value:
        return None
    return re.compile(value)


def _name_allowed(
    module_name: str,
    allow_pattern: re.Pattern | None,
    block_pattern: re.Pattern | None,
) -> bool:
    if allow_pattern and not allow_pattern.search(module_name):
        return False
    if block_pattern and block_pattern.search(module_name):
        return False
    return True


def _update_scale_cache(
    tensor: torch.Tensor,
    perturb_config: PerturbationConfig,
    module_key: str,
    channel_dim: int | None,
) -> None:
    # Cache the max observed scale per layer to stabilize noise levels across batches.
    scale = compute_symmetric_scale(
        tensor, perturb_config.bits, channel_dim=channel_dim
    )
    if scale is None:
        return
    # Keep cached scales on-device to avoid a host sync on every calibration hook.
    scale_value = scale.detach().clone()
    previous = perturb_config.scale_cache.get(module_key)
    if previous is None:
        perturb_config.scale_cache[module_key] = scale_value
        return
    previous_tensor = (
        previous.to(device=scale_value.device, dtype=scale_value.dtype)
        if torch.is_tensor(previous)
        else torch.tensor(
            previous,
            device=scale_value.device,
            dtype=scale_value.dtype,
        )
    )
    if previous_tensor.shape != scale_value.shape:
        perturb_config.scale_cache[module_key] = torch.maximum(
            previous_tensor.reshape(-1).max(),
            scale_value.reshape(-1).max(),
        )
        return
    perturb_config.scale_cache[module_key] = torch.maximum(previous_tensor, scale_value)


def _resolve_scale_override(
    perturb_config: PerturbationConfig, module_key: str
) -> torch.Tensor | None:
    if not perturb_config.use_calibrated_scale:
        return None
    scale_value = perturb_config.scale_cache.get(module_key)
    if scale_value is None:
        if perturb_config.strict_calibrated_scale:
            raise RuntimeError(f"Missing calibrated scale for {module_key}")
        return None
    return scale_value


def _apply_perturb_to_tensor(
    tensor: torch.Tensor,
    perturb_config: PerturbationConfig,
    channel_dim: int | None,
    module_key: str,
) -> torch.Tensor:
    if perturb_config.calibration_mode:
        _update_scale_cache(tensor, perturb_config, module_key, channel_dim)
        return tensor
    if not perturb_config.enabled:
        return tensor
    scale_override = _resolve_scale_override(perturb_config, module_key)
    # Check if this layer is protected from sparse gating.
    layer_sparse_enabled = perturb_config.sparse_enabled
    if (
        layer_sparse_enabled
        and perturb_config.sparse_target_module_keys is not None
        and module_key not in perturb_config.sparse_target_module_keys
    ):
        layer_sparse_enabled = False
    if (
        layer_sparse_enabled
        and perturb_config.sparse_protected_layers is not None
        and module_key in perturb_config.sparse_protected_layers
    ):
        layer_sparse_enabled = False
    sparse_activity_callback = None
    if (
        layer_sparse_enabled
        and perturb_config.sparse_activity_recorder is not None
    ):
        sparse_activity_callback = lambda stats: perturb_config.sparse_activity_recorder.record(  # noqa: E731
            module_key,
            stats,
        )
    # Resolve per-layer DET runtime payload if available.
    det_k = perturb_config.det_k_global
    if perturb_config.det_enabled and perturb_config.det_k_by_layer is not None:
        det_k = perturb_config.det_k_by_layer.get(module_key, det_k)
    det_prefix_error = perturb_config.det_prefix_error_mean
    if (
        perturb_config.det_enabled
        and perturb_config.det_prefix_error_by_layer is not None
    ):
        det_prefix_error = perturb_config.det_prefix_error_by_layer.get(
            module_key, det_prefix_error
        )
    return apply_perturb(
        tensor,
        perturb_config.bits,
        perturb_config.crosstalk_alpha,
        channel_dim,
        gaussian_noise_std=perturb_config.gaussian_noise_std,
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


def _apply_perturb_to_output(
    output: object,
    perturb_config: PerturbationConfig,
    channel_dim: int | None,
    module_key: str,
) -> object:
    # Support modules that return tuples/lists by perturbing only tensor entries.
    if torch.is_tensor(output):
        return _apply_perturb_to_tensor(
            output, perturb_config, channel_dim, module_key
        )
    if isinstance(output, (list, tuple)):
        updated = []
        changed = False
        for idx, item in enumerate(output):
            if torch.is_tensor(item):
                entry_key = f"{module_key}[{idx}]"
                updated.append(
                    _apply_perturb_to_tensor(
                        item, perturb_config, channel_dim, entry_key
                    )
                )
                changed = True
            else:
                updated.append(item)
        if changed:
            return type(output)(updated)
    return output


def _build_forward_hook(
    perturb_config: PerturbationConfig,
    channel_dim: int | None,
    module_key: str,
):
    def hook(module, inputs, output):
        return _apply_perturb_to_output(
            output, perturb_config, channel_dim, module_key
        )

    return hook


def attach_hooks(
    model: torch.nn.Module,
    perturb_config: PerturbationConfig,
    *,
    enable_conv: bool = True,
    enable_linear: bool = True,
    enable_torch_conv: bool = False,
    enable_torch_linear: bool = False,
    name_regex_allowlist: str | None = None,
    name_regex_blocklist: str | None = None,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Attach forward hooks to conv/linear layers to inject perturbations."""
    if enable_conv or enable_linear:
        try:
            from cvnets.layers.conv_layer import ConvLayer2d
            from cvnets.layers.linear_layer import LinearLayer
        except Exception as exc:
            raise SystemExit(
                "cvnets is required to attach ConvLayer2d/LinearLayer hooks."
            ) from exc
    else:
        ConvLayer2d = None
        LinearLayer = None

    # Optional regex filtering allows selective perturbation by module name.
    allow_pattern = _compile_regex(name_regex_allowlist)
    block_pattern = _compile_regex(name_regex_blocklist)

    handles = []
    for module_name, module in model.named_modules():
        if module_name == "":
            continue
        if not _name_allowed(module_name, allow_pattern, block_pattern):
            continue

        is_conv = enable_conv and ConvLayer2d and isinstance(module, ConvLayer2d)
        is_linear = enable_linear and LinearLayer and isinstance(module, LinearLayer)
        is_torch_conv = enable_torch_conv and isinstance(module, torch.nn.Conv2d)
        is_torch_linear = enable_torch_linear and isinstance(module, torch.nn.Linear)

        if is_conv:
            handles.append(
                module.register_forward_hook(
                    _build_forward_hook(
                        perturb_config, perturb_config.conv_channel_dim, module_name
                    )
                )
            )
        elif is_linear:
            handles.append(
                module.register_forward_hook(
                    _build_forward_hook(
                        perturb_config, perturb_config.linear_channel_dim, module_name
                    )
                )
            )
        elif is_torch_conv:
            handles.append(
                module.register_forward_hook(
                    _build_forward_hook(
                        perturb_config, perturb_config.conv_channel_dim, module_name
                    )
                )
            )
        elif is_torch_linear:
            handles.append(
                module.register_forward_hook(
                    _build_forward_hook(
                        perturb_config, perturb_config.linear_channel_dim, module_name
                    )
                )
            )

    return handles


def _forward_default_with_noise(
    module: torch.nn.Module,
    x_q: torch.Tensor,
    x_kv: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
    attn_mask: torch.Tensor | None,
    perturb_config: PerturbationConfig,
    module_key: str,
) -> torch.Tensor:
    batch_size, source_length, embed_dim = x_q.shape

    # Compute Q/K/V with the same logic as cvnets MultiHeadAttention forward.
    if x_kv is None:
        query_key_value = module.qkv_proj(x_q).reshape(
            batch_size, source_length, 3, module.num_heads, -1
        )
        query_key_value = query_key_value.transpose(1, 3).contiguous()
        query = query_key_value[:, :, 0]
        key = query_key_value[:, :, 1]
        value = query_key_value[:, :, 2]
    else:
        target_length = x_kv.shape[1]
        query = torch.nn.functional.linear(
            x_q,
            weight=module.qkv_proj.weight[: module.embed_dim, ...],
            bias=module.qkv_proj.bias[: module.embed_dim]
            if module.qkv_proj.bias is not None
            else None,
        )
        query = (
            query.reshape(batch_size, source_length, module.num_heads, module.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        key_value = torch.nn.functional.linear(
            x_kv,
            weight=module.qkv_proj.weight[module.embed_dim :, ...],
            bias=module.qkv_proj.bias[module.embed_dim :]
            if module.qkv_proj.bias is not None
            else None,
        )
        key_value = key_value.reshape(
            batch_size, target_length, 2, module.num_heads, module.head_dim
        )
        key_value = key_value.transpose(1, 3).contiguous()
        key = key_value[:, :, 0]
        value = key_value[:, :, 1]

    query = query * module.scaling

    key = key.transpose(-1, -2)

    attention_scores = torch.matmul(query, key)
    # Inject noise on attention score matrix before masking/softmax.
    attention_scores = _apply_perturb_to_tensor(
        attention_scores,
        perturb_config,
        perturb_config.attention_channel_dim,
        f"{module_key}.attn_scores",
    )

    batch_size, num_heads, num_src_tokens, num_tgt_tokens = attention_scores.shape
    if attn_mask is not None:
        if list(attn_mask.shape) != [
            batch_size,
            num_src_tokens,
            num_tgt_tokens,
        ]:
            raise ValueError("Attention mask shape mismatch.")
        attn_mask = attn_mask.unsqueeze(1)
        attention_scores = attention_scores + attn_mask

    if key_padding_mask is not None:
        if key_padding_mask.dim() != 2 or list(key_padding_mask.shape) != [
            batch_size,
            num_tgt_tokens,
        ]:
            raise ValueError("Key padding mask shape mismatch.")
        attention_scores = attention_scores.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
            float("-inf"),
        )

    attention_scores_dtype = attention_scores.dtype
    attention_weights = module.softmax(attention_scores.float())
    attention_weights = attention_weights.to(attention_scores_dtype)
    attention_weights = module.attn_dropout(attention_weights)

    attention_output = torch.matmul(attention_weights, value)
    # Inject noise on attention output (AV matmul).
    attention_output = _apply_perturb_to_tensor(
        attention_output,
        perturb_config,
        perturb_config.attention_output_channel_dim,
        f"{module_key}.attn_output",
    )

    attention_output = attention_output.transpose(1, 2).reshape(
        batch_size, source_length, -1
    )
    attention_output = module.out_proj(attention_output)
    return attention_output


def _build_attention_forward(
    perturb_config: PerturbationConfig, module_key: str
):
    def forward(
        module,
        x_q,
        x_kv=None,
        key_padding_mask=None,
        attn_mask=None,
        *args,
        **kwargs
    ):
        if module.coreml_compatible:
            return module.forward_tracing(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        if kwargs.get("use_pytorch_mha", False):
            return module.forward_pytorch(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        return _forward_default_with_noise(
            module,
            x_q,
            x_kv,
            key_padding_mask,
            attn_mask,
            perturb_config,
            module_key,
        )

    return forward


def patch_multi_head_attention(
    model: torch.nn.Module, perturb_config: PerturbationConfig
) -> int:
    """Patch MultiHeadAttention forward to inject perturbations."""
    try:
        from cvnets.layers.multi_head_attention import MultiHeadAttention
    except Exception as exc:
        raise SystemExit("cvnets is required to patch MultiHeadAttention.") from exc

    patched = 0
    for module_name, module in model.named_modules():
        if module_name == "":
            continue
        if not isinstance(module, MultiHeadAttention):
            continue
        if getattr(module, "_hpat_attention_patched", False):
            continue
        # Preserve original forward in case you want to restore later.
        module._hpat_attention_patched = True
        module._hpat_attention_original_forward = module.forward
        module.forward = types.MethodType(
            _build_attention_forward(perturb_config, module_name), module
        )
        patched += 1
    return patched


__all__ = [
    "PerturbationConfig",
    "SparseActivityRecorder",
    "attach_hooks",
    "patch_multi_head_attention",
]
