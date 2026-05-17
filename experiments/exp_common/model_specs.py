"""Model specs for supported ImageNet classification variants."""

from __future__ import annotations

from typing import Any


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "mobilevit_xxs": {
        "family": "mobilevit",
        "model_name": "mobilevit",
        "mode": "xx_small",
        "input_size": 192,
        "weights_filename": "mobilevit_xxs.pt",
        "config_rel": "config/classification/imagenet/mobilevit.yaml",
        "weights": "https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt",
    },
    "mobilevit_xs": {
        "family": "mobilevit",
        "model_name": "mobilevit",
        "mode": "x_small",
        "input_size": 224,
        "weights_filename": "mobilevit_xs.pt",
        "config_rel": "config/classification/imagenet/mobilevit.yaml",
        "weights": "https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.pt",
    },
    "mobilevit_s": {
        "family": "mobilevit",
        "model_name": "mobilevit",
        "mode": "small",
        "input_size": 256,
        "weights_filename": "mobilevit_s.pt",
        "config_rel": "config/classification/imagenet/mobilevit.yaml",
        "weights": "https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt",
    },
    "resnet_50": {
        "family": "resnet",
        "model_name": "resnet",
        "input_size": 224,
        "weights_filename": "resnet-50.pt",
        "config_rel": "config/classification/imagenet/resnet.yaml",
        "override_kwargs": {
            "model.classification.resnet.depth": 50,
            "model.classification.activation.name": "relu",
            "model.activation.name": "relu",
            "model.activation.inplace": True,
        },
        "weights": "https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/resnet-50.pt",
    },
}


def parse_model_keys(raw: str | None) -> list[str]:
    """Parse a comma-separated list of model keys."""
    if not raw:
        return []
    return [entry.strip() for entry in raw.split(",") if entry.strip()]

def classification_override_kwargs(
    model_key: str,
    *,
    weights_path: str | None = None,
) -> dict[str, object]:
    """Return cvnets override kwargs for a classification model spec."""
    spec = MODEL_SPECS[model_key]
    overrides: dict[str, object] = {
        "dataset.category": "classification",
        "model.classification.name": spec.get("model_name", "mobilevit"),
    }
    mode = spec.get("mode")
    if mode is not None:
        overrides["model.classification.mit.mode"] = mode
    for key, value in (spec.get("override_kwargs") or {}).items():
        overrides[str(key)] = value
    if weights_path is not None:
        overrides["model.classification.pretrained"] = weights_path
    return overrides


__all__ = ["MODEL_SPECS", "classification_override_kwargs", "parse_model_keys"]
