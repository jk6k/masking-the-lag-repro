"""Shared utilities for resolving cvnets paths, configs, and defaults."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from exp_common.model_specs import MODEL_SPECS


DEFAULTS: dict[str, object] = {
    "dataset.category": "classification",
    "model.classification.name": "mobilevit",
    "model.classification.n_classes": 1000,
    "model.classification.classifier_dropout": 0.1,
    "model.classification.freeze_batch_norm": False,
    "model.classification.enable_layer_wise_lr_decay": False,
    "model.classification.layer_wise_lr_decay_rate": 1.0,
    "model.classification.gradient_checkpointing": False,
    "model.classification.activation.name": "swish",
    "model.classification.activation.inplace": False,
    "model.classification.activation.neg_slope": 0.1,
    "model.classification.mit.mode": "small",
    "model.classification.mit.attn_dropout": 0.0,
    "model.classification.mit.ffn_dropout": 0.0,
    "model.classification.mit.dropout": 0.1,
    "model.classification.mit.number_heads": 4,
    "model.classification.mit.no_fuse_local_global_features": False,
    "model.classification.mit.conv_kernel_size": 3,
    "model.classification.mit.head_dim": None,
    "model.classification.resnet.depth": 50,
    "model.classification.resnet.dropout": 0.0,
    "model.classification.resnet.stochastic_depth_prob": 0.0,
    "model.classification.resnet.se_resnet": False,
    "model.normalization.name": "batch_norm",
    "model.normalization.groups": 1,
    "model.normalization.momentum": 0.1,
    "model.activation.name": "swish",
    "model.activation.inplace": False,
    "model.activation.neg_slope": 0.1,
    "model.layer.global_pool": "mean",
    "model.layer.conv_init": "kaiming_normal",
    "model.layer.conv_init_std_dev": None,
    "model.layer.linear_init": "trunc_normal",
    "model.layer.linear_init_std_dev": 0.02,
    "model.layer.group_linear_init_std_dev": 0.01,
    "model.resume_exclude_scopes": "",
    "model.ignore_missing_scopes": "",
    "model.rename_scopes_map": [],
    "ddp.rank": 0,
    "ddp.start_rank": 0,
    "ddp.use_distributed": False,
}


def flatten_yaml_as_dict(
    data: dict[str, object], parent_key: str = "", sep: str = "."
) -> dict[str, object]:
    """Flatten a nested YAML dict into a dot-delimited dict."""
    items = []
    for k, v in data.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_yaml_as_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def set_default(opts: argparse.Namespace, key: str, value: object) -> None:
    """Set a default attribute on an argparse namespace if missing."""
    if not hasattr(opts, key):
        setattr(opts, key, value)


def apply_defaults(opts: argparse.Namespace) -> None:
    """Apply DEFAULTS to the options namespace."""
    for key, value in DEFAULTS.items():
        set_default(opts, key, value)


def build_opts(config_path: str | None, overrides: dict[str, object]) -> argparse.Namespace:
    """Build an options namespace from a config file and override dict."""
    opts = argparse.Namespace()

    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        flat_cfg = flatten_yaml_as_dict(cfg or {})
        for key, value in flat_cfg.items():
            setattr(opts, key, value)

    for key, value in overrides.items():
        setattr(opts, key, value)

    apply_defaults(opts)
    return opts


def resolve_cvnets_root(cvnets_dir: str | None) -> Path:
    """Locate a valid ml-cvnets root and import cvnets."""
    candidates = []
    if cvnets_dir:
        candidates.append(Path(cvnets_dir))
    env_root = os.getenv("CVNETS_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    here = Path(__file__).resolve().parent
    # Check common repo-relative locations first to avoid importing a wrong global cvnets.
    candidates.extend(
        [
            here.parent / "ml-cvnets",
            here.parent / "test_code" / "ml-cvnets",
            here.parent.parent / "ml-cvnets",
            here.parent.parent / "test_code" / "ml-cvnets",
        ]
    )

    for candidate in candidates:
        if candidate and (candidate / "cvnets").is_dir():
            sys.path.insert(0, str(candidate))
            try:
                import cvnets  # noqa: F401
            except Exception:
                sys.path.pop(0)
                sys.modules.pop("cvnets", None)
                continue
            return candidate

    try:
        import cvnets  # noqa: F401

        pkg_root = Path(cvnets.__file__).resolve().parent.parent
        return pkg_root
    except Exception:
        raise SystemExit(
            "Could not locate ml-cvnets. Set --cvnets_dir or CVNETS_ROOT, or install the cvnets package."
        )


def resolve_config_path(explicit_path: str | None, cvnets_root: Path | None) -> str | None:
    """Resolve the MobileViT config path."""
    if explicit_path:
        return explicit_path
    if cvnets_root:
        candidate = (
            Path(cvnets_root)
            / "config"
            / "classification"
            / "imagenet"
            / "mobilevit.yaml"
        )
        if candidate.is_file():
            return str(candidate)
    return None


def parse_weights_override(raw: str | None) -> dict[str, str]:
    """Parse --weights_override entries of the form key=/path."""
    if not raw:
        return {}
    mapping = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise SystemExit(f"Invalid --weights_override entry: {entry}")
        key, value = entry.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def resolve_weights_path(
    model_key: str,
    weights_dir: str | None,
    weights_override: dict[str, str],
    use_pretrained: bool = True,
) -> str | None:
    """Resolve a pretrained weights path for the model, or None."""
    if not use_pretrained:
        return None
    if model_key in weights_override:
        # Explicit override takes priority over local directory or URL.
        return weights_override[model_key]
    filename = str(MODEL_SPECS[model_key].get("weights_filename") or f"{model_key}.pt")
    if weights_dir:
        candidate = Path(weights_dir) / filename
        if not candidate.is_file():
            raise SystemExit(f"Weights not found for {model_key}: {candidate}")
        return str(candidate)
    workspace_candidate = Path(__file__).resolve().parents[2] / "weights" / filename
    if workspace_candidate.is_file():
        return str(workspace_candidate)
    return MODEL_SPECS[model_key]["weights"]


def safe_cuda_available() -> tuple[bool, str | None]:
    """Probe CUDA availability in a subprocess to avoid CUDA init crashes."""
    # Avoid CUDA init side effects in the main process by probing in a subprocess.
    code = (
        "import sys, torch; "
        "ok = torch.cuda.is_available() and torch.cuda.device_count() > 0; "
        "sys.exit(0 if ok else 1)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return False, f"cuda probe failed: {exc}"

    if result.returncode == 0:
        return True, None
    message = (result.stderr or result.stdout or "").strip()
    return False, message or "cuda probe failed"


__all__ = [
    "DEFAULTS",
    "apply_defaults",
    "build_opts",
    "flatten_yaml_as_dict",
    "parse_weights_override",
    "resolve_config_path",
    "resolve_cvnets_root",
    "resolve_weights_path",
    "safe_cuda_available",
    "set_default",
]
