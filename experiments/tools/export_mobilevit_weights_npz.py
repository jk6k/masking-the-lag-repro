#!/usr/bin/env python3
"""Export MobileViT PyTorch checkpoints into MLX-friendly NPZ bundles."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = ROOT / "experiments"

if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from exp_common.cvnets_utils import parse_weights_override, resolve_weights_path  # noqa: E402

MODEL_CHOICES = ("mobilevit_xxs", "mobilevit_xs", "mobilevit_s")


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    """Extract a tensor-only state dict from common checkpoint wrappers."""
    if isinstance(payload, dict):
        for candidate_key in ("state_dict", "model", "ema", "model_state_dict"):
            candidate = payload.get(candidate_key)
            if isinstance(candidate, dict) and candidate:
                return extract_state_dict(candidate)
        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload
    raise SystemExit("Checkpoint does not contain a tensor state_dict/model payload.")


def normalize_state_key(key: str, *, strip_prefix: str | None = None) -> str:
    """Normalize exported parameter names without changing layer semantics."""
    normalized = str(key)
    prefix = (strip_prefix or "").strip()
    if prefix and normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]
    return normalized.lstrip(".")


def export_checkpoint_to_npz(
    checkpoint_path: Path,
    output_path: Path,
    *,
    strip_prefix: str | None = None,
    export_format: str = "mlx_mobilevit",
) -> dict[str, Any]:
    """Load a PyTorch checkpoint and persist its weights in ``.npz`` format."""
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = extract_state_dict(payload)
    arrays: dict[str, np.ndarray] = {}
    skipped_keys: list[str] = []
    for key, tensor in state_dict.items():
        normalized_key = normalize_state_key(key, strip_prefix=strip_prefix)
        if not normalized_key:
            raise SystemExit(f"State key became empty after normalization: {key}")
        if normalized_key.endswith(".num_batches_tracked"):
            skipped_keys.append(normalized_key)
            continue
        array = tensor.detach().cpu().numpy()
        if export_format == "mlx_mobilevit" and array.ndim == 4:
            array = np.transpose(array, (0, 2, 3, 1))
        arrays[normalized_key] = array

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)

    manifest = {
        "checkpoint_path": str(checkpoint_path),
        "output_path": str(output_path),
        "parameter_count": len(arrays),
        "strip_prefix": strip_prefix,
        "export_format": export_format,
        "skipped_keys": skipped_keys,
        "parameter_names_preview": sorted(arrays)[:20],
    }
    return manifest


def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def _exported_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _manifest_entry(
    *,
    model: str,
    export_manifest: dict[str, Any],
    git_hash: str,
    exported_at: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "weights_npz": export_manifest["output_path"],
        "source_checkpoint": export_manifest["checkpoint_path"],
        "exported_at": exported_at,
        "git_hash": git_hash,
        "weights_npz_source": "explicit_weights_npz_arg",
        **export_manifest,
    }


def _parse_models(args: argparse.Namespace) -> list[str]:
    if args.models:
        models = [item.strip() for item in str(args.models).split(",") if item.strip()]
    elif args.model:
        models = [str(args.model)]
    else:
        raise SystemExit("Provide --model or --models.")
    invalid = sorted(set(models) - set(MODEL_CHOICES))
    if invalid:
        raise SystemExit(f"Unsupported model keys: {', '.join(invalid)}")
    return models


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export MobileViT PyTorch checkpoints to an MLX-friendly NPZ bundle.",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=MODEL_CHOICES,
        help="Single model key used to resolve the default weights path.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Optional comma-separated model keys for aggregate export mode.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Explicit checkpoint path. Defaults to the repo's managed weights path.",
    )
    parser.add_argument(
        "--weights_dir",
        default=None,
        help="Optional weights directory used when --weights is not provided.",
    )
    parser.add_argument(
        "--weights_override",
        default=None,
        help="Optional key=/path override list, matching the existing CVNets harness.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Destination NPZ file.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Destination directory for aggregate export mode. Files are written as <model>.npz.",
    )
    parser.add_argument(
        "--strip_prefix",
        default="module.",
        help="Optional prefix removed from exported state keys.",
    )
    parser.add_argument(
        "--manifest_json",
        default=None,
        help="Optional JSON manifest path. Defaults to <out>.manifest.json or <out_dir>/manifest.json.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional program/freeze tag recorded on aggregate manifests.",
    )
    parser.add_argument(
        "--format",
        default="mlx_mobilevit",
        choices=("mlx_mobilevit", "raw"),
        help="Weight export format.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    model_keys = _parse_models(args)

    weights_override = parse_weights_override(args.weights_override)
    git_hash = _git_hash()
    exported_at = _exported_at()

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if args.out:
            raise SystemExit("--out cannot be combined with --out_dir.")
        if args.weights:
            raise SystemExit("--weights cannot be combined with aggregate --out_dir mode.")
        exports: list[dict[str, Any]] = []
        for model in model_keys:
            checkpoint_path = Path(
                resolve_weights_path(
                    model,
                    args.weights_dir,
                    weights_override,
                )
            )
            if not checkpoint_path.is_file():
                raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
            output_path = out_dir / f"{model}.npz"
            manifest = export_checkpoint_to_npz(
                checkpoint_path,
                output_path,
                strip_prefix=args.strip_prefix,
                export_format=args.format,
            )
            entry = _manifest_entry(
                model=model,
                export_manifest=manifest,
                git_hash=git_hash,
                exported_at=exported_at,
            )
            exports.append(entry)
            print(f"Exported {manifest['parameter_count']} tensors to {output_path}")

        manifest_payload = {
            "tag": args.tag,
            "git_hash": git_hash,
            "exported_at": exported_at,
            "weights_npz_source": "explicit_weights_npz_arg",
            "exports": exports,
        }
        manifest_path = (
            Path(args.manifest_json)
            if args.manifest_json
            else out_dir / "manifest.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote manifest to {manifest_path}")
        return 0

    if len(model_keys) != 1:
        raise SystemExit("Single-export mode requires exactly one model and --out.")
    if not args.out:
        raise SystemExit("Single-export mode requires --out.")

    model_key = model_keys[0]
    checkpoint_path = Path(
        args.weights
        or resolve_weights_path(
            model_key,
            args.weights_dir,
            weights_override,
        )
    )
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    output_path = Path(args.out)
    manifest = export_checkpoint_to_npz(
        checkpoint_path,
        output_path,
        strip_prefix=args.strip_prefix,
        export_format=args.format,
    )
    manifest_payload = _manifest_entry(
        model=model_key,
        export_manifest=manifest,
        git_hash=git_hash,
        exported_at=exported_at,
    )
    manifest_path = (
        Path(args.manifest_json)
        if args.manifest_json
        else output_path.with_suffix(output_path.suffix + ".manifest.json")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {manifest['parameter_count']} tensors to {output_path}")
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
