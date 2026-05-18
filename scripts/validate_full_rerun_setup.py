#!/usr/bin/env python3
"""Validate external inputs required for full experiment reruns."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


REQUIRED_MODEL_KEYS = ("mobilevit_xxs", "mobilevit_xs", "mobilevit_s")
REQUIRED_PT_WEIGHTS = {
    "mobilevit_xxs": "mobilevit_xxs.pt",
    "mobilevit_xs": "mobilevit_xs.pt",
    "mobilevit_s": "mobilevit_s.pt",
}
REQUIRED_IMPORTS = (
    "yaml",
    "numpy",
    "matplotlib",
    "pandas",
    "scipy",
    "seaborn",
    "cv2",
    "sklearn",
    "tqdm",
    "psutil",
    "torch",
    "torchvision",
)
OPTIONAL_IMPORTS = ("fvcore", "mlx")


@dataclass
class CheckReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def print(self) -> None:
        for message in self.warnings:
            print(f"[full-rerun-preflight] WARN {message}")
        for message in self.errors:
            print(f"[full-rerun-preflight] ERROR {message}")
        print(
            "[full-rerun-preflight] summary: "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _check_imports(report: CheckReport, *, require_mlx: bool) -> None:
    for name in REQUIRED_IMPORTS:
        if not _module_available(name):
            report.error(f"missing Python dependency: {name}")
    for name in OPTIONAL_IMPORTS:
        if not _module_available(name):
            if name == "mlx" and require_mlx:
                report.error("missing Python dependency: mlx")
            else:
                report.warn(f"optional Python dependency missing: {name}")


def _check_mps(report: CheckReport) -> None:
    code = (
        "import sys, torch; "
        "sys.exit(0 if torch.backends.mps.is_available() else 1)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        suffix = f" ({detail})" if detail else ""
        report.error(f"torch MPS backend is not available{suffix}")


def _check_imagenet(report: CheckReport, imagenet_val: str | None, *, strict: bool) -> None:
    if not imagenet_val:
        report.error("missing --imagenet-val path")
        return
    root = Path(imagenet_val).expanduser()
    if not root.is_dir():
        report.error(f"ImageNet val directory not found: {root}")
        return
    synsets = sorted(
        child for child in root.iterdir()
        if child.is_dir() and len(child.name) == 9 and child.name.startswith("n") and child.name[1:].isdigit()
    )
    if not synsets and (root / "val").is_dir():
        report.error(f"pass the synset directory layer, for example: {root / 'val'}")
        return
    if strict and len(synsets) != 1000:
        report.error(f"expected 1000 ImageNet synset directories, found {len(synsets)} in {root}")
    elif not synsets:
        report.error(f"no ImageNet synset directories found in {root}")
    checked = synsets[: min(10, len(synsets))]
    for synset in checked:
        if not any(path.is_file() for path in synset.rglob("*")):
            report.error(f"synset directory appears empty: {synset}")


def _load_weights_manifest(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    exports = payload.get("exports")
    if isinstance(exports, list):
        mapping: dict[str, str] = {}
        for item in exports:
            if not isinstance(item, dict):
                continue
            model = str(item.get("model") or "").strip()
            weights = str(item.get("weights_npz") or item.get("output_path") or "").strip()
            if model and weights:
                mapping[model] = weights
        return mapping
    model = str(payload.get("model") or "").strip()
    weights = str(payload.get("weights_npz") or payload.get("output_path") or "").strip()
    return {model: weights} if model and weights else {}


def _check_weights(
    report: CheckReport,
    *,
    weights_dir: str | None,
    weights_npz_manifest: str | None,
) -> None:
    if weights_npz_manifest:
        manifest_path = Path(weights_npz_manifest).expanduser()
        if not manifest_path.is_file():
            if weights_dir:
                report.warn(
                    f"MLX weights manifest not found, falling back to --weights-dir: {manifest_path}"
                )
            else:
                report.error(f"MLX weights manifest not found: {manifest_path}")
                return
        else:
            try:
                mapping = _load_weights_manifest(manifest_path)
            except Exception as exc:
                report.error(f"unable to parse MLX weights manifest {manifest_path}: {exc}")
                return
            manifest_base = manifest_path.parent
            for model in REQUIRED_MODEL_KEYS:
                target = mapping.get(model)
                if not target:
                    report.error(f"MLX weights manifest missing model: {model}")
                    continue
                candidates = [Path(target).expanduser()]
                if not candidates[0].is_absolute():
                    candidates.append(manifest_base / target)
                if not any(candidate.is_file() for candidate in candidates):
                    report.error(f"MLX weights NPZ not found for {model}: {target}")
            return

    if not weights_dir:
        report.error("provide --weights-dir with MobileViT .pt files or --weights-npz-manifest")
        return
    root = Path(weights_dir).expanduser()
    if not root.is_dir():
        report.error(f"weights directory not found: {root}")
        return
    for model, filename in REQUIRED_PT_WEIGHTS.items():
        path = root / filename
        if not path.is_file():
            report.error(f"missing {model} checkpoint: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenet-val", default=None)
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--weights-npz-manifest", default=None)
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--require-mlx", action="store_true")
    parser.add_argument("--strict-imagenet", action="store_true")
    parser.add_argument(
        "--allow-missing-external",
        action="store_true",
        help="Return success after printing external-data blockers. Use only for package smoke checks.",
    )
    args = parser.parse_args()

    report = CheckReport()
    _check_imports(report, require_mlx=args.require_mlx)
    if args.require_mps:
        _check_mps(report)
    _check_imagenet(report, args.imagenet_val, strict=args.strict_imagenet)
    _check_weights(
        report,
        weights_dir=args.weights_dir,
        weights_npz_manifest=args.weights_npz_manifest,
    )
    report.print()
    if report.errors and not args.allow_missing_external:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
