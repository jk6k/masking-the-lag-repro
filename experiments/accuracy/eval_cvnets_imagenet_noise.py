"""Evaluate MobileViT accuracy under quantization noise and crosstalk."""

from __future__ import annotations

import argparse
import csv
import functools
import gc
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from exp_common.runtime import prepare_runtime_environment_for_request  # noqa: E402

prepare_runtime_environment_for_request()

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "numpy is required. Install it with: pip install numpy (or pip install -r requirements.txt)"
    ) from exc
import torch

from exp_common.cvnets_utils import (  # noqa: E402
    parse_weights_override,
    resolve_config_path,
    resolve_cvnets_root,
    resolve_weights_path,
    safe_cuda_available,
)
from exp_common.io_utils import backup_existing_file  # noqa: E402
from exp_common.model_specs import (  # noqa: E402
    MODEL_SPECS,
    classification_override_kwargs,
    parse_model_keys,
)
from exp_common.runtime import (  # noqa: E402
    apply_torch_device,
    ensure_validated_accuracy_backend,
    prefer_vision_channels_last,
    resolve_data_workers,
    resolve_dataloader_prefetch_factor,
    resolve_device_preference,
    resolve_eval_batch_size,
)

from accuracy.inject_hooks import (
    PerturbationConfig,
    SparseActivityRecorder,
    attach_hooks,
    patch_multi_head_attention,
)

os.environ["CVNETS_TOLERATE_IMPORT_ERRORS"] = "1"
os.environ.setdefault("CVNETS_SUPPRESS_OPTIONAL_IMPORT_WARNINGS", "1")
os.environ.setdefault("CVNETS_LOG_TRAINABLE_PARAMETERS", "0")


DEFAULT_NOISE_SIGMA_LSB = "0,0.25,0.5,1.0,2.0"
DEFAULT_GAUSSIAN_NOISE_STD = "0,0.25,0.5,1.0,2.0"
DEFAULT_CROSSTALK_ALPHA = "0,0.01,0.02,0.05"
PASS_MODE_PAIRED = "paired"
PASS_MODE_BASELINE_ONLY = "baseline_only"
PASS_MODE_QUANTIZED_ONLY = "quantized_only"
PASS_MODE_CHOICES = [
    PASS_MODE_PAIRED,
    PASS_MODE_BASELINE_ONLY,
    PASS_MODE_QUANTIZED_ONLY,
]
RESULT_FIELDNAMES = [
    "run_id",
    "source_run_id",
    "experiment_id",
    "baseline",
    "device",
    "workload",
    "profile",
    "sweep_resolution",
    "git_hash",
    "imagenet_val",
    "imagenet_manifest",
    "config_snapshot",
    "det_policy",
    "det_k_signature",
    "det_k_global",
    "det_prefix_error_mean",
    "det_prefix_error_p95",
    "det_perturbation",
    "sparse_tau_global",
    "sparse_active_fraction",
    "sparse_perturbation",
    "sparse_gate_mode",
    "sparse_measured_activity_fraction",
    "sparse_measured_zero_fraction",
    "sparse_stats_total_elements",
    "sparse_stats_active_elements",
    "sparse_stats_call_count",
    "sparse_stats_module_count",
    "model",
    "input_size",
    "quant_bits",
    "noise_sigma_lsb",
    "gaussian_noise_std",
    "crosstalk_alpha",
    "drift_lsb",
    "noise_correlation",
    "burst_error_prob",
    "burst_error_scale_lsb",
    "burst_span",
    "top1",
    "top5",
    "top1_delta",
    "top5_delta",
    "measured_pass_elapsed_s",
    "measured_processed_samples",
    "latency_ms_per_sample",
    "measurement_window",
    "seed",
    "notes",
]
ROW_IDENTITY_FIELDS = [
    field
    for field in RESULT_FIELDNAMES
    if field
    not in {
        "source_run_id",
        "profile",
        "sweep_resolution",
        "top1",
        "top5",
        "top1_delta",
        "top5_delta",
        "measured_pass_elapsed_s",
        "measured_processed_samples",
        "latency_ms_per_sample",
        "measurement_window",
    }
]
_STOP_REQUESTED = False


class EvaluationInterrupted(RuntimeError):
    """Raised when a graceful stop was requested during evaluation."""

SPARSE_ACTIVITY_LAYER_FIELDNAMES = [
    "run_id",
    "experiment_id",
    "baseline",
    "model",
    "seed",
    "device",
    "gaussian_noise_std",
    "crosstalk_alpha",
    "module_key",
    "sparse_gate_mode",
    "sparse_measured_activity_fraction",
    "sparse_measured_zero_fraction",
    "sparse_stats_total_elements",
    "sparse_stats_active_elements",
    "sparse_stats_call_count",
]


IMG_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".ppm",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
}

IMAGENET_SYNSET_DIR_RE = re.compile(r"^n\d{8}$")


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def _get_git_hash(repo_root: Path) -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return output.decode("utf-8", errors="ignore").strip() or None


def _planned_pass_count(
    *,
    noise_sigmas: list[float],
    crosstalk_alphas: list[float],
    pass_mode: str,
) -> int:
    quantized_pass_count = len(noise_sigmas) * len(crosstalk_alphas)
    if pass_mode == PASS_MODE_BASELINE_ONLY:
        return 1
    if pass_mode == PASS_MODE_QUANTIZED_ONLY:
        return quantized_pass_count
    return 1 + quantized_pass_count


def _load_baseline_reference_row(
    *,
    baseline_reference_csv: str,
    baseline_reference_run_id: str,
    model_key: str,
) -> dict[str, str]:
    path = Path(baseline_reference_csv)
    if not path.is_file():
        raise SystemExit(f"Missing --baseline_reference_csv: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [
        row
        for row in rows
        if str(row.get("run_id") or "").strip() == str(baseline_reference_run_id).strip()
        and str(row.get("model") or "").strip() == str(model_key).strip()
        and str(row.get("baseline") or "").strip().lower() == "true"
    ]
    if not matches:
        raise SystemExit(
            "Could not find a baseline reference row for "
            f"run_id={baseline_reference_run_id!r}, model={model_key!r} in {path}."
        )
    return matches[-1]


def _baseline_metrics_from_reference(row: dict[str, str]) -> tuple[float, float]:
    try:
        top1 = float(row["top1"])
        top5 = float(row["top5"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            "Baseline reference row is missing numeric top1/top5 values."
        ) from exc
    return top1, top5


def _find_imagenet_samples(
    root_dir: str,
) -> tuple[list[tuple[str, int]], dict[str, int]]:
    root = Path(root_dir)
    if not root.is_dir():
        raise SystemExit(f"ImageNet val directory not found: {root}")
    # ImageNet val layout: one synset folder per class at the provided root.
    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    class_dirs.sort(key=lambda p: p.name)
    synset_dirs = [d for d in class_dirs if IMAGENET_SYNSET_DIR_RE.fullmatch(d.name)]
    if len(synset_dirs) != len(class_dirs):
        nested_val = root / "val"
        nested_val_synsets = []
        if nested_val.is_dir():
            nested_val_synsets = sorted(
                [
                    d
                    for d in nested_val.iterdir()
                    if d.is_dir() and IMAGENET_SYNSET_DIR_RE.fullmatch(d.name)
                ],
                key=lambda p: p.name,
            )
        if nested_val_synsets:
            raise SystemExit(
                "ImageNet val root must point at the synset directory layer. "
                f"Received {root}, which looks like a container root; pass {nested_val} instead."
            )
        bad_dirs = [d.name for d in class_dirs if not IMAGENET_SYNSET_DIR_RE.fullmatch(d.name)]
        preview = ", ".join(bad_dirs[:5])
        raise SystemExit(
            "ImageNet val root must contain synset-named class directories "
            f"(e.g. n01440764). Received {root} with non-synset entries: {preview}."
        )
    class_dirs = synset_dirs
    class_to_idx = {cls.name: idx for idx, cls in enumerate(class_dirs)}
    samples = []
    for cls_dir in class_dirs:
        for path in sorted(cls_dir.rglob("*")):
            if path.is_file() and _is_image_file(path):
                samples.append((str(path), class_to_idx[cls_dir.name]))
    if not samples:
        raise SystemExit(f"No images found under {root}")
    return samples, class_to_idx


def _load_imagenet_manifest(manifest_path: str) -> list[tuple[str, int]]:
    """Load a CSV manifest with columns: path,label."""
    path = Path(manifest_path)
    if not path.is_file():
        raise SystemExit(f"ImageNet manifest not found: {path}")
    samples: list[tuple[str, int]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"Invalid manifest (missing header): {path}")
        if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise SystemExit(f"Manifest must have columns path,label: {path}")
        for row in reader:
            img_path = (row.get("path") or "").strip()
            label_str = (row.get("label") or "").strip()
            if not img_path:
                continue
            try:
                label = int(label_str)
            except ValueError as exc:
                raise SystemExit(f"Invalid label in manifest: {label_str}") from exc
            samples.append((img_path, label))
    if not samples:
        raise SystemExit(f"No samples in manifest: {path}")
    return samples


class OpenCVImageNetDataset(torch.utils.data.Dataset):
    """Minimal ImageNet val loader using OpenCV with optional normalization."""
    def __init__(
        self,
        root_dir: str,
        *,
        manifest_path: str | None = None,
        resize_size: int,
        center_crop_size: int,
        percentage: float,
        seed: int,
        enable_mean_std: bool,
        mean_std_mean: list[float],
        mean_std_std: list[float],
    ) -> None:
        super().__init__()
        try:
            import cv2  # noqa: F401
        except Exception as exc:
            raise SystemExit(
                "OpenCV (cv2) is required for --opencv_pipeline. Install opencv-python."
            ) from exc

        if manifest_path:
            samples = _load_imagenet_manifest(manifest_path)
            class_to_idx = {}
        else:
            samples, class_to_idx = _find_imagenet_samples(root_dir)
        if percentage < 100.0:
            rng = random.Random(seed)
            indices = list(range(len(samples)))
            rng.shuffle(indices)
            keep = max(1, int(len(indices) * (percentage / 100.0)))
            indices = indices[:keep]
            samples = [samples[i] for i in indices]

        self.samples = samples
        self.class_to_idx = class_to_idx
        self.resize_size = resize_size
        self.center_crop_size = center_crop_size
        self.enable_mean_std = enable_mean_std
        self.mean_std_mean = mean_std_mean
        self.mean_std_std = mean_std_std
        self._mean = None
        self._std = None
        if self.enable_mean_std:
            self._mean = np.array(self.mean_std_mean, dtype="float32").reshape(1, 1, 3)
            self._std = np.array(self.mean_std_std, dtype="float32").reshape(1, 1, 3)

    def __len__(self) -> int:
        return len(self.samples)

    def _resize_shorter_side(self, img, size):
        import cv2

        h, w = img.shape[:2]
        if h == 0 or w == 0:
            raise RuntimeError("Invalid image with zero dimension.")
        if h < w:
            new_h = size
            new_w = int(round(w * size / h))
        else:
            new_w = size
            new_h = int(round(h * size / w))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _center_crop(self, img, crop_size):
        h, w = img.shape[:2]
        top = max(0, (h - crop_size) // 2)
        left = max(0, (w - crop_size) // 2)
        return img[top : top + crop_size, left : left + crop_size]

    def __getitem__(self, idx):
        import cv2

        path, target = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")

        img = self._resize_shorter_side(img, self.resize_size)
        img = self._center_crop(img, self.center_crop_size)

        img = img.astype("float32") / 255.0
        if self.enable_mean_std:
            img = (img - self._mean) / self._std

        # BGR HWC -> CHW
        img = torch.from_numpy(img).permute(2, 0, 1)
        return {"samples": img, "targets": int(target), "sample_id": idx}


def opencv_image_collate_fn(
    batch: list[dict[str, object]],
    *,
    use_channels_last: bool = False,
) -> dict[str, torch.Tensor]:
    """Pickle-safe collate function for OpenCV ImageNet batches."""
    samples = torch.stack([b["samples"] for b in batch], dim=0)
    if use_channels_last:
        samples = samples.contiguous(memory_format=torch.channels_last)
    targets = torch.tensor([b["targets"] for b in batch], dtype=torch.long)
    return {"samples": samples, "targets": targets}


def build_opencv_loader(
    imagenet_val: str,
    imagenet_manifest: str | None,
    *,
    input_size,
    resize_size,
    center_crop_size,
    percentage,
    seed,
    batch_size,
    workers,
    enable_mean_std,
    mean_std_mean,
    mean_std_std,
    device: torch.device,
    prefetch_factor: int | None = None,
    use_channels_last: bool = False,
):
    """Build a simple OpenCV-based dataloader (single dataset class)."""
    crop_size = center_crop_size if center_crop_size is not None else input_size
    resize_dim = resize_size if resize_size is not None else input_size + 32
    if resize_dim < crop_size:
        resize_dim = crop_size

    dataset = OpenCVImageNetDataset(
        imagenet_val,
        manifest_path=imagenet_manifest,
        resize_size=resize_dim,
        center_crop_size=crop_size,
        percentage=percentage,
        seed=seed,
        enable_mean_std=enable_mean_std,
        mean_std_mean=mean_std_mean,
        mean_std_std=mean_std_std,
    )

    # Use pin_memory only for CUDA; dataloader workers default to 0 if not set.
    num_workers = max(0, int(workers or 0))
    pin_memory = device.type == "cuda"
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=functools.partial(
            opencv_image_collate_fn,
            use_channels_last=use_channels_last,
        ),
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return torch.utils.data.DataLoader(**loader_kwargs)


def parse_list(value: str | None, cast_type=float) -> list:
    """Parse a comma-separated list and cast each entry."""
    if value is None:
        return []
    items = []
    for entry in value.split(","):
        entry = entry.strip()
        if entry:
            items.append(cast_type(entry))
    return items


def parse_float_list(raw: str) -> list[float]:
    """Parse a comma-separated float list."""
    return [float(x) for x in raw.split(",") if x.strip()]


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def patch_dataloader_prefetch_for_single_worker() -> None:
    try:
        from data.loader.dataloader import CVNetsDataLoader
    except Exception:
        return
    if getattr(CVNetsDataLoader, "_hpat_prefetch_patched", False):
        return
    original_init = CVNetsDataLoader.__init__

    def patched_init(
        self,
        dataset,
        batch_size,
        batch_sampler,
        num_workers=1,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=None,
        prefetch_factor=2,
        *args,
        **kwargs,
    ):
        # cvnets expects prefetch_factor only when num_workers > 0.
        if num_workers == 0:
            prefetch_factor = None
        return original_init(
            self,
            dataset,
            batch_size,
            batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
            prefetch_factor=prefetch_factor,
            *args,
            **kwargs,
        )

    CVNetsDataLoader.__init__ = patched_init
    CVNetsDataLoader._hpat_prefetch_patched = True


def build_eval_args(
    config_path: str,
    results_dir: str,
    run_label: str,
    model_key: str,
    weights_path: str | None,
    imagenet_val: str,
    input_size: int,
    percentage: float,
    eval_batch_size: int | None,
    workers: int | None,
    seed: int,
    enable_mean_std: bool,
    mean_std_mean: list[float],
    mean_std_std: list[float],
    resize_size: int | None,
    center_crop_size: int | None,
) -> list[str]:
    crop_size = center_crop_size if center_crop_size is not None else input_size
    resize_dim = resize_size if resize_size is not None else input_size + 32
    overrides = [
        f"dataset.root_val={imagenet_val}",
        f"image_augmentation.center_crop.size={crop_size}",
        f"image_augmentation.resize.size={resize_dim}",
        f"sampler.vbs.crop_size_width={crop_size}",
        f"sampler.vbs.crop_size_height={crop_size}",
        f"dataset.percentage_of_samples={percentage}",
        f"dataset.sample_selection_random_seed={seed}",
    ]
    for key, value in classification_override_kwargs(model_key, weights_path=weights_path).items():
        overrides.append(f"{key}={value}")
    if enable_mean_std:
        overrides.append(
            "image_augmentation.to_tensor.mean_std_normalization.enable=True"
        )
        overrides.append(
            "image_augmentation.to_tensor.mean_std_normalization.mean="
            + ",".join(str(x) for x in mean_std_mean)
        )
        overrides.append(
            "image_augmentation.to_tensor.mean_std_normalization.std="
            + ",".join(str(x) for x in mean_std_std)
        )
    if eval_batch_size is not None:
        overrides.append(f"dataset.eval_batch_size0={eval_batch_size}")
    if workers is not None:
        overrides.append(f"dataset.workers={workers}")

    eval_args = [
        "--common.config-file",
        config_path,
        "--common.results-loc",
        results_dir,
        "--common.run-label",
        run_label,
        "--common.seed",
        str(seed),
        "--common.override-kwargs",
        *overrides,
    ]
    return eval_args


def _maybe_swap_rgb_bgr(tensor: torch.Tensor | object) -> torch.Tensor | object:
    if not torch.is_tensor(tensor):
        return tensor
    if tensor.ndim < 3:
        return tensor
    if tensor.shape[1] != 3:
        return tensor
    return tensor[:, [2, 1, 0], ...]


def _apply_input_adjustments(
    samples: object,
    *,
    data_color_order="rgb",
    model_color_order="rgb",
    input_scale=1.0,
):
    swap_channels = (
        data_color_order in {"rgb", "bgr"}
        and model_color_order in {"rgb", "bgr"}
        and data_color_order != model_color_order
    )
    if torch.is_tensor(samples):
        adjusted = samples * input_scale if input_scale != 1.0 else samples
        if swap_channels:
            adjusted = _maybe_swap_rgb_bgr(adjusted)
        return adjusted
    if isinstance(samples, dict):
        adjusted = dict(samples)
        if "image" in adjusted and torch.is_tensor(adjusted["image"]):
            img = adjusted["image"]
            if input_scale != 1.0:
                img = img * input_scale
            if swap_channels:
                img = _maybe_swap_rgb_bgr(img)
            adjusted["image"] = img
        return adjusted
    return samples


def _optimize_samples_for_device(
    samples: object,
    *,
    use_channels_last: bool,
):
    if not use_channels_last:
        return samples
    if torch.is_tensor(samples):
        if samples.ndim == 4 and samples.is_floating_point():
            if samples.is_contiguous(memory_format=torch.channels_last):
                return samples
            return samples.contiguous(memory_format=torch.channels_last)
        return samples
    if isinstance(samples, dict):
        adjusted = dict(samples)
        if "image" in adjusted and torch.is_tensor(adjusted["image"]):
            adjusted["image"] = _optimize_samples_for_device(
                adjusted["image"],
                use_channels_last=use_channels_last,
            )
        return adjusted
    return samples


def _prepare_batch_for_device_transfer(
    batch: object,
    *,
    data_color_order: str,
    model_color_order: str,
    input_scale: float,
    use_channels_last: bool,
) -> object:
    if not isinstance(batch, dict):
        return batch
    adjusted = dict(batch)
    if "samples" in adjusted:
        samples = adjusted["samples"]
        samples = _apply_input_adjustments(
            samples,
            data_color_order=data_color_order,
            model_color_order=model_color_order,
            input_scale=input_scale,
        )
        samples = _optimize_samples_for_device(
            samples,
            use_channels_last=use_channels_last,
        )
        adjusted["samples"] = samples
    return adjusted


def _shutdown_loader_workers(loader: object) -> None:
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def _release_accelerator_memory(device_name: str) -> None:
    gc.collect()
    normalized = (device_name or "").strip().lower()
    if normalized == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        return
    if normalized != "mps":
        return
    mps_backend = getattr(torch, "mps", None)
    empty_cache = getattr(mps_backend, "empty_cache", None)
    if callable(empty_cache):
        try:
            empty_cache()
        except Exception:
            pass


def _infer_batch_size(samples: object) -> int:
    if torch.is_tensor(samples):
        return int(samples.shape[0])
    if isinstance(samples, dict):
        for key in ("image", "video", "audio"):
            if key in samples:
                return _infer_batch_size(samples[key])
        raise ValueError(f"Unsupported sample dict keys: {samples.keys()}")
    if isinstance(samples, list):
        return len(samples)
    raise ValueError(f"Unsupported samples type: {type(samples)}")


def _slice_batched_value(value: object, size: int) -> object:
    if size < 0:
        raise ValueError("size must be non-negative")
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value
        return value[:size]
    if isinstance(value, dict):
        return {k: _slice_batched_value(v, size) for k, v in value.items()}
    if isinstance(value, list):
        return value[:size]
    if isinstance(value, tuple):
        return value.__class__(_slice_batched_value(v, size) for v in value[:size])
    return value


def _truncate_batch_to_size(batch: dict[str, object], size: int) -> dict[str, object]:
    if size < 0:
        raise ValueError("size must be non-negative")
    return {key: _slice_batched_value(value, size) for key, value in batch.items()}


def _resolve_total_samples(
    test_loader: torch.utils.data.DataLoader,
    *,
    max_samples: int | None,
) -> int | None:
    """Resolve total sample count for progress reporting."""
    if max_samples is not None:
        return max(1, int(max_samples))
    dataset = getattr(test_loader, "dataset", None)
    if dataset is not None:
        try:
            return int(len(dataset))
        except Exception:
            pass
    try:
        return int(len(test_loader))
    except Exception:
        return None


def _maybe_print_percent_progress(
    *,
    prefix: str,
    processed_samples: int,
    total_samples: int | None,
    next_percent_marker: int,
    progress_callback=None,
) -> int:
    """Emit compact percentage progress logs at 5% increments."""
    if total_samples is None or total_samples <= 0:
        return next_percent_marker
    percent = (100.0 * float(processed_samples)) / float(total_samples)
    while percent >= next_percent_marker:
        marker = min(next_percent_marker, 100)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{timestamp} [progress] {prefix} {marker}% ({processed_samples}/{total_samples})",
            flush=True,
        )
        if progress_callback is not None:
            progress_callback(marker)
        next_percent_marker += 5
    return next_percent_marker


def _compute_topk_correct_counts(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute top-1 and top-k correct counts without host scalar sync."""
    if predictions.ndim != 2:
        raise ValueError(
            f"Expected predictions with shape [batch, classes], got {tuple(predictions.shape)}"
        )
    if targets.ndim != 1:
        raise ValueError(f"Expected targets with shape [batch], got {tuple(targets.shape)}")
    if predictions.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Batch size mismatch between predictions={tuple(predictions.shape)} and targets={tuple(targets.shape)}"
        )

    resolved_k = max(1, min(int(max_k), int(predictions.shape[1])))
    topk_indices = predictions.topk(resolved_k, dim=1, largest=True, sorted=True).indices
    matches = topk_indices.eq(targets.view(-1, 1))
    top1_correct = matches[:, 0].sum(dtype=torch.int64)
    topk_correct = matches.any(dim=1).sum(dtype=torch.int64)
    return top1_correct, topk_correct


def evaluate_model(
    opts,
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    *,
    log_progress=True,
    max_samples=None,
    data_color_order="rgb",
    model_color_order="rgb",
    input_scale=1.0,
    progress_prefix: str | None = None,
    progress_recorder: ProgressRecorder | None = None,
    total_passes: int | None = None,
    pass_index: int | None = None,
    pass_kind: str = "eval",
    gaussian_noise_std: float | None = None,
    crosstalk_alpha: float | None = None,
    return_stats: bool = False,
):
    from common import DEFAULT_LOG_FREQ
    from utils.common_utils import move_to_device
    from utils.ddp_utils import is_master

    device = getattr(opts, "dev.device", torch.device("cpu"))
    log_freq = getattr(opts, "common.log_freq", DEFAULT_LOG_FREQ)
    use_mixed_precision = getattr(opts, "common.mixed_precision", False)
    mixed_precision_dtype = getattr(opts, "common.mixed_precision_dtype", "float16")
    use_channels_last = bool(getattr(opts, "common.channels_last", False))

    if use_mixed_precision and device.type == "cuda":
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        autocast_dtype = dtype_map.get(mixed_precision_dtype, torch.float16)
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
    else:
        autocast_ctx = nullcontext()

    # Inference-only evaluation; no gradients, optional autocast for CUDA.
    model.eval()
    start_time = time.time()
    processed_samples = 0
    total_samples = _resolve_total_samples(
        test_loader,
        max_samples=max_samples,
    )
    if progress_recorder is not None and total_passes is not None and pass_index is not None:
        progress_recorder.emit_pass_event(
            event="pass_start",
            total_passes=total_passes,
            pass_index=pass_index,
            pass_kind=pass_kind,
            gaussian_noise_std=gaussian_noise_std,
            crosstalk_alpha=crosstalk_alpha,
            processed_samples=0,
            total_samples=total_samples,
            pass_elapsed_seconds=0.0,
        )
    next_percent_marker = 5
    top1_correct_total = torch.zeros((), dtype=torch.int64, device=device)
    top5_correct_total = torch.zeros((), dtype=torch.int64, device=device)
    with torch.inference_mode():
        for batch_index, batch in enumerate(test_loader):
            _raise_if_stop_requested("batch_start")
            if max_samples is not None:
                remaining = max_samples - processed_samples
                if remaining <= 0:
                    break
                batch_samples = _infer_batch_size(batch["samples"])
                if remaining < batch_samples:
                    batch = _truncate_batch_to_size(batch, remaining)
            batch = _prepare_batch_for_device_transfer(
                batch,
                data_color_order=data_color_order,
                model_color_order=model_color_order,
                input_scale=input_scale,
                use_channels_last=use_channels_last,
            )
            batch = move_to_device(opts=opts, x=batch, device=device)
            samples, targets = batch["samples"], batch["targets"]
            batch_size = _infer_batch_size(samples)

            with autocast_ctx:
                predictions = model(samples)

            top1_correct, top5_correct = _compute_topk_correct_counts(
                predictions,
                targets,
                max_k=5,
            )
            top1_correct_total = top1_correct_total + top1_correct
            top5_correct_total = top5_correct_total + top5_correct
            del predictions

            processed_samples += batch_size
            progress_label = progress_prefix or "eval"
            next_percent_marker = _maybe_print_percent_progress(
                prefix=progress_label,
                processed_samples=processed_samples,
                total_samples=total_samples,
                next_percent_marker=next_percent_marker,
                progress_callback=(
                    None
                    if progress_recorder is None or total_passes is None or pass_index is None
                    else lambda marker: progress_recorder.emit_pass_event(
                        event="pass_progress",
                        total_passes=total_passes,
                        pass_index=pass_index,
                        pass_kind=pass_kind,
                        gaussian_noise_std=gaussian_noise_std,
                        crosstalk_alpha=crosstalk_alpha,
                        processed_samples=processed_samples,
                        total_samples=total_samples,
                        pass_elapsed_seconds=max(0.0, time.time() - start_time),
                        milestone_percent=marker,
                    )
                ),
            )
            if log_progress and batch_index % log_freq == 0 and is_master(opts):
                elapsed = time.time() - start_time
                print(
                    f"[eval-status] {progress_label} processed={processed_samples}/{total_samples if total_samples is not None else '?'} elapsed={elapsed:5.2f}s",
                    flush=True,
                )
            _raise_if_stop_requested("batch_end")
            if max_samples is not None and processed_samples >= max_samples:
                break

    top1 = None
    top5 = None
    if processed_samples > 0:
        final_accuracy = (
            torch.stack([top1_correct_total, top5_correct_total], dim=0)
            .to(dtype=torch.float32)
            .mul(100.0 / float(processed_samples))
            .cpu()
            .tolist()
        )
        top1 = float(final_accuracy[0])
        top5 = float(final_accuracy[1])
    if total_samples and processed_samples >= total_samples:
        if next_percent_marker <= 100:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"{timestamp} [progress] {(progress_prefix or 'eval')} 100% ({processed_samples}/{total_samples})",
                flush=True,
            )
    elapsed_s = max(0.0, time.time() - start_time)
    if progress_recorder is not None and total_passes is not None and pass_index is not None:
        progress_recorder.emit_pass_event(
            event="pass_complete",
            total_passes=total_passes,
            pass_index=pass_index,
            pass_kind=pass_kind,
            gaussian_noise_std=gaussian_noise_std,
            crosstalk_alpha=crosstalk_alpha,
            processed_samples=processed_samples,
            total_samples=total_samples,
            pass_elapsed_seconds=elapsed_s,
            milestone_percent=100 if total_samples is not None else None,
        )
    stats = {
        "processed_samples": processed_samples,
        "elapsed_s": elapsed_s,
    }
    if return_stats:
        return top1, top5, stats
    return top1, top5


def write_results(
    output_path: Path, rows: list[dict[str, object]], append: bool
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    mode = "a" if append else "w"
    if file_exists and not append:
        backup = backup_existing_file(output_path)
        if backup:
            print(f"Existing results moved to {backup}")
    with output_path.open(mode, newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=RESULT_FIELDNAMES)
        if not append or not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _normalize_row_identity_value(field: str, value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if field in {"baseline", "det_perturbation", "sparse_perturbation"}:
        return "1" if text.lower() == "true" else "0"
    if field in {
        "input_size",
        "quant_bits",
        "noise_sigma_lsb",
        "crosstalk_alpha",
        "drift_lsb",
        "noise_correlation",
        "burst_error_prob",
        "burst_error_scale_lsb",
        "burst_span",
        "seed",
        "det_k_global",
        "det_prefix_error_mean",
        "det_prefix_error_p95",
        "sparse_tau_global",
        "sparse_active_fraction",
    }:
        try:
            return f"{float(text):.12g}"
        except ValueError:
            return text
    return text


def result_row_identity(row: dict[str, object]) -> tuple[str, ...]:
    return tuple(
        _normalize_row_identity_value(field, row.get(field))
        for field in ROW_IDENTITY_FIELDS
    )


def load_existing_result_identities(path: Path) -> set[tuple[str, ...]]:
    if not path.is_file():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {result_row_identity(row) for row in reader}


def _request_graceful_stop(signum, _frame) -> None:
    global _STOP_REQUESTED
    if _STOP_REQUESTED:
        raise KeyboardInterrupt
    _STOP_REQUESTED = True
    signal_name = signal.Signals(signum).name
    print(
        f"[interrupt] received {signal_name}; will stop at the next safe checkpoint so resume can continue cleanly.",
        flush=True,
    )


def _install_interrupt_handlers() -> None:
    for signame in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, signame, None)
        if signum is not None:
            signal.signal(signum, _request_graceful_stop)


def _raise_if_stop_requested(where: str) -> None:
    if _STOP_REQUESTED:
        raise EvaluationInterrupted(where)


def _coerce_evaluate_result(
    result: object,
) -> tuple[float | None, float | None, dict[str, object]]:
    if not isinstance(result, tuple):
        raise TypeError(
            "evaluate_model() must return a tuple of "
            "(top1, top5) or (top1, top5, stats)."
        )
    if len(result) == 2:
        top1, top5 = result
        return top1, top5, {}
    if len(result) == 3:
        top1, top5, stats = result
        return top1, top5, dict(stats or {})
    raise ValueError(
        "evaluate_model() must return exactly 2 or 3 values for "
        "(top1, top5[, stats])."
    )


def _build_result_row(
    *,
    base_row: dict[str, object],
    baseline: bool,
    model_key: str,
    input_size: int,
    quant_bits: int | None,
    noise_sigma_lsb: float,
    crosstalk_alpha: float,
    drift_lsb: float,
    noise_correlation: float,
    burst_error_prob: float,
    burst_error_scale_lsb: float,
    burst_span: int,
    top1: float | None,
    top5: float | None,
    top1_delta: float | None,
    top5_delta: float | None,
    measured_pass_elapsed_s: float | None = None,
    measured_processed_samples: int | None = None,
    latency_ms_per_sample: float | None = None,
    measurement_window: str = "",
    seed: int,
    notes: str,
) -> dict[str, object]:
    return {
        **base_row,
        "baseline": baseline,
        "model": model_key,
        "input_size": input_size,
        "quant_bits": quant_bits,
        "noise_sigma_lsb": noise_sigma_lsb,
        "gaussian_noise_std": noise_sigma_lsb,
        "crosstalk_alpha": crosstalk_alpha,
        "drift_lsb": drift_lsb,
        "noise_correlation": noise_correlation,
        "burst_error_prob": burst_error_prob,
        "burst_error_scale_lsb": burst_error_scale_lsb,
        "burst_span": burst_span,
        "top1": top1,
        "top5": top5,
        "top1_delta": top1_delta,
        "top5_delta": top5_delta,
        "measured_pass_elapsed_s": measured_pass_elapsed_s,
        "measured_processed_samples": measured_processed_samples,
        "latency_ms_per_sample": latency_ms_per_sample,
        "measurement_window": measurement_window,
        "seed": seed,
        "notes": notes,
    }


def _config_conditioned_enabled(args: argparse.Namespace) -> bool:
    return bool(args.apply_det_perturbation or args.apply_sparse_perturbation)


def _result_note(
    *,
    sigma_value: float,
    alpha_value: float,
    drift_lsb: float,
    noise_correlation: float,
    burst_error_prob: float,
    burst_error_scale_lsb: float,
    config_conditioned: bool,
) -> str:
    if config_conditioned:
        return "config_conditioned_sim"
    if (
        sigma_value == 0.0
        and
        alpha_value == 0.0
        and drift_lsb == 0.0
        and noise_correlation == 0.0
        and burst_error_prob == 0.0
        and burst_error_scale_lsb == 0.0
    ):
        return "baseline_quant"
    return "hpat_sim"


def _parse_json_mapping(raw: str | None, *, arg_name: str) -> dict[str, float] | None:
    if raw is None or not str(raw).strip():
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{arg_name} must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"{arg_name} must decode to a JSON object.")
    resolved: dict[str, float] = {}
    for key, value in parsed.items():
        try:
            resolved[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"{arg_name} values must be numeric; got {value!r} for key {key!r}."
            ) from exc
    return resolved or None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MobileViT ImageNet accuracy with quantization, noise, and crosstalk."
    )
    parser.add_argument("--imagenet_val", required=True, help="Path to ImageNet val root.")
    parser.add_argument(
        "--imagenet_manifest",
        default=None,
        help="Optional CSV manifest (path,label). Requires --opencv_pipeline.",
    )
    parser.add_argument(
        "--results_csv",
        default=str(
            Path(__file__).resolve().parents[1] / "results" / "accuracy_noise.csv"
        ),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--models",
        default="mobilevit_xxs,mobilevit_xs,mobilevit_s",
        help="Comma-separated model keys.",
    )
    parser.add_argument("--cvnets_dir", default=None, help="Local ml-cvnets path.")
    parser.add_argument("--config", default=None, help="Override config path.")
    parser.add_argument("--weights_dir", default=None, help="Local weights directory.")
    parser.add_argument(
        "--weights_override",
        default=None,
        help="Comma-separated overrides: mobilevit_xxs=/path/to/xxs.pt",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--run_id", default=None, help="Run ID for traceability.")
    parser.add_argument(
        "--source_run_id",
        default=None,
        help="Optional source run ID for cross-tool provenance.",
    )
    parser.add_argument(
        "--experiment_id",
        default=None,
        help="Experiment ID (E0-E6) for alignment with the design doc.",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Logical noise profile label to record in results.",
    )
    parser.add_argument(
        "--sweep_resolution",
        default="unspecified",
        help="Logical sweep resolution label such as dense or sparse.",
    )
    parser.add_argument(
        "--device_label",
        default=None,
        help="Optional device label override for results.",
    )
    parser.add_argument(
        "--workload",
        default=None,
        help="Workload label to include in results.",
    )
    parser.add_argument(
        "--git_hash",
        default=None,
        help="Explicit git hash to record (overrides --record_git_hash).",
    )
    parser.add_argument(
        "--record_git_hash",
        action="store_true",
        help="Record git hash from the current repo.",
    )
    parser.add_argument(
        "--config_snapshot",
        default=None,
        help="Path to a saved config snapshot (optional).",
    )
    parser.add_argument(
        "--metadata_json",
        default=None,
        help="Optional JSON path to write run metadata.",
    )
    parser.add_argument(
        "--progress_jsonl",
        default=None,
        help="Optional JSONL path for pass-level progress and ETA events.",
    )
    parser.add_argument(
        "--progress_label",
        default=None,
        help="Optional label for console progress and progress JSONL events.",
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=100.0,
        help="Percentage of ImageNet val samples.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Override dataset.eval_batch_size0.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Limit evaluation to the first N samples for quick smoke tests.",
    )
    parser.add_argument(
        "--pass_mode",
        choices=PASS_MODE_CHOICES,
        default=PASS_MODE_PAIRED,
    )
    parser.add_argument("--baseline_reference_csv", default=None)
    parser.add_argument("--baseline_reference_run_id", default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override dataset.workers.",
    )
    parser.add_argument(
        "--opencv_pipeline",
        action="store_true",
        help="Use an OpenCV-based ImageNet loader (v0.1 style).",
    )
    parser.add_argument(
        "--resize_size",
        type=int,
        default=None,
        help="Override image_augmentation.resize.size (shorter side).",
    )
    parser.add_argument(
        "--center_crop_size",
        type=int,
        default=None,
        help="Override image_augmentation.center_crop.size.",
    )
    parser.add_argument(
        "--input_color_order",
        choices=["rgb", "bgr"],
        default="rgb",
        help="Color channel order expected by the model.",
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        default=1.0,
        help="Multiply input tensor by this factor after ToTensor.",
    )
    parser.add_argument(
        "--enable_mean_std",
        action="store_true",
        help="Enable mean/std normalization in the ToTensor transform.",
    )
    parser.add_argument(
        "--mean_std_mean",
        default="0.485,0.456,0.406",
        help="Comma-separated mean values for normalization.",
    )
    parser.add_argument(
        "--mean_std_std",
        default="0.229,0.224,0.225",
        help="Comma-separated std values for normalization.",
    )
    parser.add_argument(
        "--quant_bits",
        type=int,
        default=8,
        help="Quantization bits.",
    )
    parser.add_argument(
        "--noise_sigma_lsb",
        "--gaussian_noise_sigma_lsb",
        dest="noise_sigma_lsb",
        default=DEFAULT_NOISE_SIGMA_LSB,
        help="Comma-separated Gaussian-noise sigma list in LSB units.",
    )
    parser.add_argument(
        "--crosstalk_alpha",
        default=DEFAULT_CROSSTALK_ALPHA,
        help="Comma-separated alpha list.",
    )
    parser.add_argument(
        "--gaussian_noise_std",
        default=DEFAULT_GAUSSIAN_NOISE_STD,
        help="Comma-separated channel-shared Gaussian drift std list in LSB-equivalent units.",
    )
    parser.add_argument("--drift_lsb", type=float, default=0.0, help="Static drift in LSB units.")
    parser.add_argument(
        "--noise_correlation",
        type=float,
        default=0.0,
        help="Reserved for future correlated-noise modeling; non-zero values are rejected.",
    )
    parser.add_argument(
        "--burst_error_prob",
        type=float,
        default=0.0,
        help="Reserved for future burst-error modeling; non-zero values are rejected.",
    )
    parser.add_argument(
        "--burst_error_scale_lsb",
        type=float,
        default=0.0,
        help="Reserved for future burst-error scale modeling; non-zero values are rejected.",
    )
    parser.add_argument("--burst_span", type=int, default=1, help="Burst span in elements.")
    parser.add_argument("--apply_det_perturbation", action="store_true")
    parser.add_argument("--det_mode", default=None)
    parser.add_argument("--det_bsl_max", type=float, default=None)
    parser.add_argument("--det_policy", default=None)
    parser.add_argument("--det_k_signature", default=None)
    parser.add_argument("--det_k_global", type=float, default=None)
    parser.add_argument("--det_k_by_layer_json", default=None)
    parser.add_argument("--det_prefix_error_mean", type=float, default=0.0)
    parser.add_argument("--det_prefix_error_p95", type=float, default=None)
    parser.add_argument("--det_prefix_error_by_layer_json", default=None)
    parser.add_argument("--apply_sparse_perturbation", action="store_true")
    parser.add_argument("--sparse_tau_global", type=float, default=None)
    parser.add_argument("--sparse_active_fraction", type=float, default=None)
    parser.add_argument(
        "--collect_sparse_activity_stats",
        action="store_true",
        help="Record measured sparse activity statistics from the actual gate mask.",
    )
    parser.add_argument(
        "--sparse_activity_layers_csv",
        type=str,
        default=None,
        help="Optional CSV path for per-module sparse activity statistics.",
    )
    parser.add_argument(
        "--disable_conv",
        action="store_true",
        help="Disable ConvLayer2d perturbation.",
    )
    parser.add_argument(
        "--disable_linear",
        action="store_true",
        help="Disable LinearLayer perturbation.",
    )
    parser.add_argument(
        "--enable_torch_conv",
        action="store_true",
        help="Also perturb torch.nn.Conv2d modules.",
    )
    parser.add_argument(
        "--enable_torch_linear",
        action="store_true",
        help="Also perturb torch.nn.Linear modules.",
    )
    parser.add_argument(
        "--enable_attention",
        action="store_true",
        help="Perturb attention matmul outputs.",
    )
    parser.add_argument(
        "--disable_calibration",
        action="store_true",
        help="Disable calibrated scales and use per-batch dynamic scaling.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to CSV instead of overwriting.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing results CSV by skipping already written rows.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device selection for evaluation.",
    )
    parser.add_argument(
        "--allow_unvalidated_mps",
        action="store_true",
        help="Legacy compatibility flag; MPS is now accepted for strict accuracy evaluation after local parity validation.",
    )
    return parser


def main():
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    parser = _build_arg_parser()
    args = parser.parse_args()
    _install_interrupt_handlers()
    if args.pass_mode == PASS_MODE_QUANTIZED_ONLY:
        if not str(args.baseline_reference_csv or "").strip():
            raise SystemExit(
                "--pass_mode quantized_only requires --baseline_reference_csv."
            )
        if not str(args.baseline_reference_run_id or "").strip():
            raise SystemExit(
                "--pass_mode quantized_only requires --baseline_reference_run_id."
            )
    if args.imagenet_manifest and not args.opencv_pipeline:
        raise SystemExit("--imagenet_manifest requires --opencv_pipeline.")

    # Resolve cvnets root + config first so downstream imports work.
    cvnets_root = resolve_cvnets_root(args.cvnets_dir)
    config_path = resolve_config_path(args.config, cvnets_root)
    if not config_path:
        raise SystemExit("Could not resolve mobilevit.yaml config.")

    from cvnets import get_model
    from data import create_test_loader
    from utils.common_utils import device_setup

    results_path = Path(args.results_csv)
    existing_row_identities = (
        load_existing_result_identities(results_path)
        if args.resume
        else set()
    )
    append_results = bool(args.append or (args.resume and results_path.exists()))
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    git_hash = args.git_hash or (_get_git_hash(ROOT_DIR) if args.record_git_hash else None)
    det_prefix_error_by_layer = _parse_json_mapping(
        args.det_prefix_error_by_layer_json,
        arg_name="--det_prefix_error_by_layer_json",
    )
    det_k_by_layer = _parse_json_mapping(
        args.det_k_by_layer_json,
        arg_name="--det_k_by_layer_json",
    )

    model_keys = parse_model_keys(args.models)
    weights_override = parse_weights_override(args.weights_override)
    noise_sigmas = parse_list(args.gaussian_noise_std, float)
    if not noise_sigmas:
        noise_sigmas = [0.0]
    legacy_noise_sigmas = parse_list(args.noise_sigma_lsb, float)
    if not legacy_noise_sigmas:
        legacy_noise_sigmas = [0.0]
    if legacy_noise_sigmas != noise_sigmas:
        default_noise_sigmas = parse_list(DEFAULT_NOISE_SIGMA_LSB, float) or [0.0]
        default_gaussian_noise_stds = parse_list(DEFAULT_GAUSSIAN_NOISE_STD, float) or [0.0]
        legacy_overridden = legacy_noise_sigmas != default_noise_sigmas
        gaussian_overridden = noise_sigmas != default_gaussian_noise_stds
        if legacy_overridden and gaussian_overridden:
            raise SystemExit(
                "--noise_sigma_lsb and --gaussian_noise_std encode the same active "
                "noise axis in this bounded harness and must match when both are set."
            )
        if legacy_overridden:
            noise_sigmas = legacy_noise_sigmas
    if float(args.noise_correlation) != 0.0:
        raise SystemExit(
            "--noise_correlation is reserved for a future correlated-noise proxy and "
            "must remain 0.0 in the current bounded evaluation harness."
        )
    if float(args.burst_error_prob) != 0.0 or float(args.burst_error_scale_lsb) != 0.0:
        raise SystemExit(
            "--burst_error_* is reserved for future modeling and must remain 0.0 in "
            "the current bounded evaluation harness."
        )
    crosstalk_alphas = parse_list(args.crosstalk_alpha, float)
    config_conditioned = _config_conditioned_enabled(args)
    max_eval_samples = None
    if args.max_eval_samples is not None:
        max_eval_samples = max(1, int(args.max_eval_samples))
        print(f"Smoke test: limiting eval samples to {max_eval_samples}.")
    resolved_device_name, device_note = resolve_device_preference(
        args.device,
        cuda_probe_fn=safe_cuda_available,
    )
    ensure_validated_accuracy_backend(
        resolved_device_name,
        allow_unvalidated_mps=bool(args.allow_unvalidated_mps),
    )
    if args.device == "auto":
        if device_note:
            print(
                f"Auto device selection chose {resolved_device_name}. Details: {device_note}"
            )
        else:
            print(f"Auto device selection chose {resolved_device_name}.")

    resolved_worker_count = resolve_data_workers(args.workers, resolved_device_name)
    resolved_prefetch_factor = resolve_dataloader_prefetch_factor(
        resolved_device_name,
        resolved_worker_count,
    )
    if resolved_worker_count == 0:
        patch_dataloader_prefetch_for_single_worker()

    mean_std_mean = parse_float_list(args.mean_std_mean)
    mean_std_std = parse_float_list(args.mean_std_std)
    runtime_tuning: dict[str, dict[str, object]] = {}
    planned_passes_by_model: dict[str, int] = {}

    for model_key in model_keys:
        if model_key not in MODEL_SPECS:
            raise SystemExit(f"Unsupported model: {model_key}")
        spec = MODEL_SPECS[model_key]
        input_size = spec["input_size"]
        weights_path = resolve_weights_path(model_key, args.weights_dir, weights_override)
        resolved_eval_batch_size = resolve_eval_batch_size(
            args.eval_batch_size,
            resolved_device_name,
            model_key,
            max_eval_samples=max_eval_samples,
        )

        # Build cvnets options for this model.
        eval_args = build_eval_args(
            config_path=config_path,
            results_dir=str(results_path.parent),
            run_label=f"{model_key}_accuracy",
            model_key=model_key,
            weights_path=weights_path,
            imagenet_val=args.imagenet_val,
            input_size=input_size,
            percentage=args.percentage,
            eval_batch_size=resolved_eval_batch_size,
            workers=resolved_worker_count,
            seed=args.seed,
            enable_mean_std=args.enable_mean_std,
            mean_std_mean=mean_std_mean,
            mean_std_std=mean_std_std,
            resize_size=args.resize_size,
            center_crop_size=args.center_crop_size,
        )
        from options.opts import get_eval_arguments

        opts = get_eval_arguments(args=eval_args)
        # Let cvnets initialize CUDA-specific settings when CUDA is selected.
        if resolved_device_name == "cuda":
            opts = device_setup(opts)
        apply_torch_device(opts, resolved_device_name)
        setattr(opts, "dataset.workers", resolved_worker_count)
        setattr(opts, "dataset.persistent_workers", resolved_worker_count > 0)
        if resolved_prefetch_factor is not None:
            setattr(opts, "dataset.prefetch_factor", resolved_prefetch_factor)
        else:
            setattr(opts, "dataset.prefetch_factor", None)
        use_channels_last = prefer_vision_channels_last(
            resolved_device_name,
            bool(getattr(opts, "common.channels_last", False)),
        )
        setattr(opts, "common.channels_last", use_channels_last)
        resolved_device = args.device_label or str(getattr(opts, "dev.device"))
        base_row = {
            "run_id": run_id,
            "source_run_id": args.source_run_id or run_id,
            "experiment_id": args.experiment_id,
            "baseline": None,
            "device": resolved_device,
            "workload": args.workload,
            "profile": args.profile,
            "sweep_resolution": args.sweep_resolution,
            "git_hash": git_hash,
            "imagenet_val": args.imagenet_val,
            "imagenet_manifest": args.imagenet_manifest,
            "config_snapshot": args.config_snapshot,
            "det_policy": args.det_policy,
            "det_k_signature": args.det_k_signature,
            "det_k_global": args.det_k_global,
            "det_prefix_error_mean": args.det_prefix_error_mean if args.apply_det_perturbation else None,
            "det_prefix_error_p95": args.det_prefix_error_p95 if args.apply_det_perturbation else None,
            "det_perturbation": bool(args.apply_det_perturbation),
            "sparse_tau_global": args.sparse_tau_global if args.apply_sparse_perturbation else None,
            "sparse_active_fraction": args.sparse_active_fraction if args.apply_sparse_perturbation else None,
            "sparse_perturbation": bool(args.apply_sparse_perturbation),
            "sparse_gate_mode": None,
            "sparse_measured_activity_fraction": None,
            "sparse_measured_zero_fraction": None,
            "sparse_stats_total_elements": None,
            "sparse_stats_active_elements": None,
            "sparse_stats_call_count": None,
            "sparse_stats_module_count": None,
        }
        baseline_row_template = _build_result_row(
            base_row=base_row,
            baseline=True,
            model_key=model_key,
            input_size=input_size,
            quant_bits=None,
            noise_sigma_lsb=0.0,
            crosstalk_alpha=0.0,
            drift_lsb=0.0,
            noise_correlation=0.0,
            burst_error_prob=0.0,
            burst_error_scale_lsb=0.0,
            burst_span=max(1, int(args.burst_span)),
            top1=None,
            top5=None,
            top1_delta=None,
            top5_delta=None,
            measured_pass_elapsed_s=None,
            measured_processed_samples=None,
            latency_ms_per_sample=None,
            measurement_window="baseline_eval_pass",
            seed=args.seed,
            notes="baseline_fp32",
        )
        sweep_row_templates = [
            _build_result_row(
                base_row=base_row,
                baseline=False,
                model_key=model_key,
                input_size=input_size,
                quant_bits=args.quant_bits,
                noise_sigma_lsb=float(sigma_value),
                crosstalk_alpha=float(alpha_value),
                drift_lsb=float(args.drift_lsb),
                noise_correlation=float(args.noise_correlation),
                burst_error_prob=float(args.burst_error_prob),
                burst_error_scale_lsb=float(args.burst_error_scale_lsb),
                burst_span=max(1, int(args.burst_span)),
                top1=None,
                top5=None,
                top1_delta=None,
                top5_delta=None,
                measured_pass_elapsed_s=None,
                measured_processed_samples=None,
                latency_ms_per_sample=None,
                measurement_window="quantized_eval_pass",
                seed=args.seed,
                notes=_result_note(
                    sigma_value=float(sigma_value),
                    alpha_value=float(alpha_value),
                    drift_lsb=float(args.drift_lsb),
                    noise_correlation=float(args.noise_correlation),
                    burst_error_prob=float(args.burst_error_prob),
                    burst_error_scale_lsb=float(args.burst_error_scale_lsb),
                    config_conditioned=config_conditioned,
                ),
            )
            for sigma_value in noise_sigmas
            for alpha_value in crosstalk_alphas
        ]
        planned_passes_by_model[model_key] = _planned_pass_count(
            noise_sigmas=noise_sigmas,
            crosstalk_alphas=crosstalk_alphas,
            pass_mode=str(args.pass_mode),
        )
        expected_row_identities = set()
        if args.pass_mode in {PASS_MODE_PAIRED, PASS_MODE_BASELINE_ONLY}:
            expected_row_identities.add(result_row_identity(baseline_row_template))
        if args.pass_mode in {PASS_MODE_PAIRED, PASS_MODE_QUANTIZED_ONLY}:
            expected_row_identities.update(
                result_row_identity(row) for row in sweep_row_templates
            )
        if args.resume and all(
            row_identity in existing_row_identities
            for row_identity in expected_row_identities
        ):
            print(
                f"[resume] skip completed model={model_key} run_id={run_id}",
                flush=True,
            )
            continue

        # Build model and move to device.
        model = get_model(opts)
        memory_format = (
            torch.channels_last
            if use_channels_last
            else torch.contiguous_format
        )
        model = model.to(device=getattr(opts, "dev.device"), memory_format=memory_format)
        model.eval()
        runtime_tuning[model_key] = {
            "device": resolved_device_name,
            "workers": resolved_worker_count,
            "prefetch_factor": resolved_prefetch_factor,
            "eval_batch_size": resolved_eval_batch_size,
            "channels_last": use_channels_last,
        }
        print(
            "[acc-noise] "
            f"model={model_key} device={resolved_device_name} "
            f"workers={resolved_worker_count} "
            f"prefetch_factor={resolved_prefetch_factor if resolved_prefetch_factor is not None else 'n/a'} "
            f"eval_batch_size={resolved_eval_batch_size if resolved_eval_batch_size is not None else getattr(opts, 'dataset.eval_batch_size0', 'n/a')} "
            f"channels_last={use_channels_last}",
            flush=True,
        )

        test_loader = None
        perturb_config = None
        hook_handles = []
        try:
            # Build the evaluation dataloader (cvnets loader by default).
            if args.opencv_pipeline:
                eval_batch_size = resolved_eval_batch_size
                if eval_batch_size is None:
                    eval_batch_size = getattr(opts, "dataset.eval_batch_size0", 100)
                test_loader = build_opencv_loader(
                    args.imagenet_val,
                    args.imagenet_manifest,
                    input_size=input_size,
                    resize_size=args.resize_size,
                    center_crop_size=args.center_crop_size,
                    percentage=args.percentage,
                    seed=args.seed,
                    batch_size=eval_batch_size,
                    workers=resolved_worker_count,
                    enable_mean_std=args.enable_mean_std,
                    mean_std_mean=mean_std_mean,
                    mean_std_std=mean_std_std,
                    device=getattr(opts, "dev.device"),
                    prefetch_factor=resolved_prefetch_factor,
                    use_channels_last=use_channels_last,
                )
            else:
                test_loader = create_test_loader(opts)

            # Attach perturbation hooks (Conv/Linear/Attention).
            perturb_config = PerturbationConfig(
                bits=args.quant_bits,
                sigma_lsb=0.0,
                crosstalk_alpha=0.0,
                drift_lsb=0.0,
                noise_correlation=0.0,
                burst_error_prob=0.0,
                burst_error_scale_lsb=0.0,
                burst_span=max(1, int(args.burst_span)),
                det_enabled=bool(args.apply_det_perturbation),
                det_prefix_error_mean=float(args.det_prefix_error_mean or 0.0),
                sparse_enabled=bool(args.apply_sparse_perturbation),
                sparse_active_fraction=args.sparse_active_fraction,
                sparse_tau_global=(
                    args.sparse_tau_global if args.apply_sparse_perturbation else None
                ),
                enabled=False,
            )

            hook_handles = attach_hooks(
                model,
                perturb_config,
                enable_conv=not args.disable_conv,
                enable_linear=not args.disable_linear,
                enable_torch_conv=args.enable_torch_conv,
                enable_torch_linear=args.enable_torch_linear,
            )

            if args.enable_attention:
                patch_multi_head_attention(model, perturb_config)

            baseline_top1: float | None = None
            baseline_top5: float | None = None
            data_color_order = "bgr" if args.opencv_pipeline else "rgb"
            if args.pass_mode == PASS_MODE_QUANTIZED_ONLY:
                baseline_reference_row = _load_baseline_reference_row(
                    baseline_reference_csv=str(args.baseline_reference_csv),
                    baseline_reference_run_id=str(args.baseline_reference_run_id),
                    model_key=model_key,
                )
                baseline_top1, baseline_top5 = _baseline_metrics_from_reference(
                    baseline_reference_row
                )
            else:
                # Baseline evaluation (no noise). Optional calibration caches scales.
                if not args.disable_calibration:
                    perturb_config.calibration_mode = True
                _raise_if_stop_requested("before_baseline_eval")
                set_seeds(args.seed)
                baseline_top1, baseline_top5, baseline_stats = _coerce_evaluate_result(
                    evaluate_model(
                        opts,
                        model,
                        test_loader,
                        max_samples=max_eval_samples,
                        data_color_order=data_color_order,
                        model_color_order=args.input_color_order,
                        input_scale=args.input_scale,
                        return_stats=True,
                    )
                )
                baseline_processed_samples = int(
                    baseline_stats.get("processed_samples") or 0
                )
                baseline_elapsed_s = float(baseline_stats.get("elapsed_s") or 0.0)
                baseline_latency_ms_per_sample = (
                    (baseline_elapsed_s / float(baseline_processed_samples)) * 1000.0
                    if baseline_processed_samples > 0
                    else None
                )
                if not args.disable_calibration:
                    perturb_config.calibration_mode = False
                    perturb_config.use_calibrated_scale = True

                baseline_row = _build_result_row(
                    base_row=base_row,
                    baseline=True,
                    model_key=model_key,
                    input_size=input_size,
                    quant_bits=None,
                    noise_sigma_lsb=0.0,
                    crosstalk_alpha=0.0,
                    drift_lsb=0.0,
                    noise_correlation=0.0,
                    burst_error_prob=0.0,
                    burst_error_scale_lsb=0.0,
                    burst_span=max(1, int(args.burst_span)),
                    top1=baseline_top1,
                    top5=baseline_top5,
                    top1_delta=None,
                    top5_delta=None,
                    measured_pass_elapsed_s=baseline_elapsed_s,
                    measured_processed_samples=baseline_processed_samples,
                    latency_ms_per_sample=baseline_latency_ms_per_sample,
                    measurement_window="baseline_eval_pass",
                    seed=args.seed,
                    notes="baseline_fp32",
                )
                baseline_identity = result_row_identity(baseline_row)
                if baseline_identity not in existing_row_identities:
                    write_results(results_path, [baseline_row], append=append_results)
                    append_results = True
                    existing_row_identities.add(baseline_identity)

            if args.pass_mode == PASS_MODE_BASELINE_ONLY:
                continue

            # Noise + crosstalk sweep across the requested grid.
            perturb_config.enabled = True
            perturb_config.bits = args.quant_bits
            perturb_config.drift_lsb = float(args.drift_lsb)
            perturb_config.noise_correlation = float(args.noise_correlation)
            perturb_config.burst_error_prob = float(args.burst_error_prob)
            perturb_config.burst_error_scale_lsb = float(args.burst_error_scale_lsb)
            perturb_config.burst_span = max(1, int(args.burst_span))

            for sigma_value in noise_sigmas:
                for alpha_value in crosstalk_alphas:
                    result_note = _result_note(
                        sigma_value=float(sigma_value),
                        alpha_value=float(alpha_value),
                        drift_lsb=float(args.drift_lsb),
                        noise_correlation=float(args.noise_correlation),
                        burst_error_prob=float(args.burst_error_prob),
                        burst_error_scale_lsb=float(args.burst_error_scale_lsb),
                        config_conditioned=config_conditioned,
                    )
                    pending_row = _build_result_row(
                        base_row=base_row,
                        baseline=False,
                        model_key=model_key,
                        input_size=input_size,
                        quant_bits=args.quant_bits,
                        noise_sigma_lsb=float(sigma_value),
                        crosstalk_alpha=float(alpha_value),
                        drift_lsb=float(args.drift_lsb),
                        noise_correlation=float(args.noise_correlation),
                        burst_error_prob=float(args.burst_error_prob),
                        burst_error_scale_lsb=float(args.burst_error_scale_lsb),
                        burst_span=max(1, int(args.burst_span)),
                        top1=None,
                        top5=None,
                        top1_delta=None,
                        top5_delta=None,
                        measured_pass_elapsed_s=None,
                        measured_processed_samples=None,
                        latency_ms_per_sample=None,
                        measurement_window="quantized_eval_pass",
                        seed=args.seed,
                        notes=result_note,
                    )
                    pending_identity = result_row_identity(pending_row)
                    if args.resume and pending_identity in existing_row_identities:
                        print(
                            f"[resume] skip completed row model={model_key} sigma={float(sigma_value):.12g} alpha={float(alpha_value):.12g} run_id={run_id}",
                            flush=True,
                        )
                        continue

                    perturb_config.sigma_lsb = sigma_value
                    perturb_config.crosstalk_alpha = alpha_value
                    _raise_if_stop_requested("before_sweep_eval")
                    set_seeds(args.seed)
                    top1, top5, sweep_stats = _coerce_evaluate_result(
                        evaluate_model(
                            opts,
                            model,
                            test_loader,
                            max_samples=max_eval_samples,
                            data_color_order=data_color_order,
                            model_color_order=args.input_color_order,
                            input_scale=args.input_scale,
                            return_stats=True,
                        )
                    )
                    sweep_processed_samples = int(
                        sweep_stats.get("processed_samples") or 0
                    )
                    sweep_elapsed_s = float(sweep_stats.get("elapsed_s") or 0.0)
                    sweep_latency_ms_per_sample = (
                        (sweep_elapsed_s / float(sweep_processed_samples)) * 1000.0
                        if sweep_processed_samples > 0
                        else None
                    )
                    top1_delta = (
                        None if baseline_top1 is None else top1 - baseline_top1
                    )
                    top5_delta = (
                        None if baseline_top5 is None else top5 - baseline_top5
                    )
                    completed_row = _build_result_row(
                        base_row=base_row,
                        baseline=False,
                        model_key=model_key,
                        input_size=input_size,
                        quant_bits=args.quant_bits,
                        noise_sigma_lsb=float(sigma_value),
                        crosstalk_alpha=float(alpha_value),
                        drift_lsb=float(args.drift_lsb),
                        noise_correlation=float(args.noise_correlation),
                        burst_error_prob=float(args.burst_error_prob),
                        burst_error_scale_lsb=float(args.burst_error_scale_lsb),
                        burst_span=max(1, int(args.burst_span)),
                        top1=top1,
                        top5=top5,
                        top1_delta=top1_delta,
                        top5_delta=top5_delta,
                        measured_pass_elapsed_s=sweep_elapsed_s,
                        measured_processed_samples=sweep_processed_samples,
                        latency_ms_per_sample=sweep_latency_ms_per_sample,
                        measurement_window="quantized_eval_pass",
                        seed=args.seed,
                        notes=result_note,
                    )
                    write_results(results_path, [completed_row], append=append_results)
                    append_results = True
                    existing_row_identities.add(pending_identity)
        except EvaluationInterrupted as exc:
            print(
                f"[resume] graceful stop recorded at {exc}; rerun the same command with --resume to continue from the existing CSV.",
                flush=True,
            )
            raise SystemExit(130)
        finally:
            for handle in hook_handles:
                handle.remove()
            if test_loader is not None:
                _shutdown_loader_workers(test_loader)
                del test_loader
            del model
            if perturb_config is not None:
                del perturb_config
            _release_accelerator_memory(resolved_device_name)

    if args.metadata_json:
        meta_path = Path(args.metadata_json)
        meta = {
            "run_id": run_id,
            "source_run_id": args.source_run_id or run_id,
            "experiment_id": args.experiment_id,
            "git_hash": git_hash,
            "results_csv": str(results_path),
            "imagenet_val": args.imagenet_val,
            "imagenet_manifest": args.imagenet_manifest,
            "config_snapshot": args.config_snapshot,
            "det_policy": args.det_policy,
            "det_k_signature": args.det_k_signature,
            "seed": args.seed,
            "device": resolved_device_name,
            "workload": args.workload,
            "profile": args.profile,
            "sweep_resolution": args.sweep_resolution,
            "runtime_tuning": runtime_tuning,
            "progress_jsonl": args.progress_jsonl,
            "planned_passes_by_model": planned_passes_by_model,
            "pass_mode": args.pass_mode,
            "baseline_reference_csv": args.baseline_reference_csv,
            "baseline_reference_run_id": args.baseline_reference_run_id,
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
