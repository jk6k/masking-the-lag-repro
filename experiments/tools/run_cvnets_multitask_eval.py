#!/usr/bin/env python3
"""Audit and stage official CVNets multi-task evaluation chains.

This tool does two things:
1. Audits whether local CVNets evaluation chains are runnable for a small set of
   classification / detection / segmentation workloads.
2. Builds the exact command that would be used to execute each workload, without
   fabricating any broader-task results when local datasets or weights are absent.

Actual execution is optional. The audit mode is the default and is the primary
artifact used for AICAS reviewer-facing readiness disclosure.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from exp_common.runtime import resolve_device_preference
from experiments.tools.path_policy import MAIN_PROJECT_REPORT_DATA_DIR


DEFAULT_CVNETS_ROOT = ROOT / "experiments" / "ml-cvnets"
DEFAULT_IMAGENET_ROOT = ROOT / "experiments" / "datasets" / "imagenet"
DEFAULT_RESULTS_DIR = ROOT / "experiments" / "results" / "multitask_eval"
DEFAULT_AUDIT_CSV = MAIN_PROJECT_REPORT_DATA_DIR / "cvnets_multitask_chain_audit_20260307.csv"


def _display_path(path_like: str | Path) -> str:
    return Path(path_like).as_posix()


def _display_pure_path(path_like: str | Path) -> PurePosixPath:
    return PurePosixPath(_display_path(path_like))


LOCAL_DATASET_ROOT_OVERRIDES: dict[str, PurePosixPath] = {
    "mobilevitv2_1p0_ade20k_seg": _display_pure_path(
        ROOT / "experiments" / "datasets" / "ade20k_hf_val2000" / "ADEChallengeData2016"
    ),
    "mobilevitv2_1p0_coco_det": _display_pure_path(ROOT / "experiments" / "datasets" / "coco_hf_val5000"),
    "mobilevit_s_pascalvoc_seg": _display_pure_path(ROOT / "experiments" / "datasets" / "pascal_voc_hf"),
}

CLASSIFICATION_TUTORIAL_URL = "https://apple.github.io/ml-cvnets/en/models/classification/README-classification-tutorial.html"
DETECTION_TUTORIAL_URL = "https://apple.github.io/ml-cvnets/en/models/detection/README-detection-SSD-tutorial.html"
SEGMENTATION_TUTORIAL_URL = "https://apple.github.io/ml-cvnets/en/models/segmentation/README-segmentation-deeplabv3-tutorial.html"
MODEL_ZOO_URL = "https://apple.github.io/ml-cvnets/en/general/README-model-zoo.html"
REPO_URL = "https://github.com/apple/ml-cvnets"


@dataclass(frozen=True)
class EvalSpec:
    workload_id: str
    task_id: str
    task_label: str
    model: str
    model_family: str
    dataset_name: str
    primary_metric_name: str
    eval_kind: str
    local_config_rel: str
    recommended_local_weights_rel: str
    official_weights_url: str
    official_config_url: str
    tutorial_url: str
    model_zoo_url: str
    entrypoint_label: str
    n_classes: int | None = None
    extra_override_kwargs: tuple[str, ...] = ()


WORKLOAD_SPECS: tuple[EvalSpec, ...] = (
    EvalSpec(
        workload_id="mobilevit_xxs_imagenet_cls",
        task_id="imagenet_cls",
        task_label="ImageNet classification",
        model="mobilevit_xxs",
        model_family="mobilevit",
        dataset_name="ImageNet-1k",
        primary_metric_name="Top1",
        eval_kind="classification",
        local_config_rel="experiments/ml-cvnets/config/classification/imagenet/mobilevit.yaml",
        recommended_local_weights_rel="weights/mobilevit_xxs.pt",
        official_weights_url="https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt",
        official_config_url="https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.yaml",
        tutorial_url=CLASSIFICATION_TUTORIAL_URL,
        model_zoo_url=MODEL_ZOO_URL,
        entrypoint_label="eval_cvnets_imagenet_noise.py",
    ),
    EvalSpec(
        workload_id="mobilevit_xs_imagenet_cls",
        task_id="imagenet_cls",
        task_label="ImageNet classification",
        model="mobilevit_xs",
        model_family="mobilevit",
        dataset_name="ImageNet-1k",
        primary_metric_name="Top1",
        eval_kind="classification",
        local_config_rel="experiments/ml-cvnets/config/classification/imagenet/mobilevit.yaml",
        recommended_local_weights_rel="weights/mobilevit_xs.pt",
        official_weights_url="https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.pt",
        official_config_url="https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.yaml",
        tutorial_url=CLASSIFICATION_TUTORIAL_URL,
        model_zoo_url=MODEL_ZOO_URL,
        entrypoint_label="eval_cvnets_imagenet_noise.py",
    ),
    EvalSpec(
        workload_id="mobilevit_s_imagenet_cls",
        task_id="imagenet_cls",
        task_label="ImageNet classification",
        model="mobilevit_s",
        model_family="mobilevit",
        dataset_name="ImageNet-1k",
        primary_metric_name="Top1",
        eval_kind="classification",
        local_config_rel="experiments/ml-cvnets/config/classification/imagenet/mobilevit.yaml",
        recommended_local_weights_rel="weights/mobilevit_s.pt",
        official_weights_url="https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt",
        official_config_url="https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.yaml",
        tutorial_url=CLASSIFICATION_TUTORIAL_URL,
        model_zoo_url=MODEL_ZOO_URL,
        entrypoint_label="eval_cvnets_imagenet_noise.py",
    ),
    EvalSpec(
        workload_id="mobilevitv2_1p0_coco_det",
        task_id="coco_det",
        task_label="MS-COCO detection",
        model="mobilevitv2_1.0",
        model_family="mobilevitv2",
        dataset_name="MS-COCO",
        primary_metric_name="mAP@[.5:.95]",
        eval_kind="detection",
        local_config_rel="experiments/ml-cvnets/config/detection/ssd_coco/mobilevit_v2.yaml",
        recommended_local_weights_rel="weights/coco-ssd-mobilevitv2-1.0.pt",
        official_weights_url="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-1.0.pt",
        official_config_url="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-1.0.yaml",
        tutorial_url=DETECTION_TUTORIAL_URL,
        model_zoo_url=MODEL_ZOO_URL,
        entrypoint_label="main_eval.main_worker_detection",
        n_classes=81,
        extra_override_kwargs=("model.classification.mitv2.width_multiplier=1.0",),
    ),
    EvalSpec(
        workload_id="mobilevitv2_1p0_ade20k_seg",
        task_id="ade20k_seg",
        task_label="ADE20K segmentation",
        model="mobilevitv2_1.0",
        model_family="mobilevitv2",
        dataset_name="ADE20K",
        primary_metric_name="mIoU",
        eval_kind="segmentation",
        local_config_rel="experiments/ml-cvnets/config/segmentation/ade20k/deeplabv3_mobilevitv2.yaml",
        recommended_local_weights_rel="weights/deeplabv3-mobilevitv2-1.0.pt",
        official_weights_url="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-1.0.pt",
        official_config_url="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-1.0.yaml",
        tutorial_url=SEGMENTATION_TUTORIAL_URL,
        model_zoo_url=MODEL_ZOO_URL,
        entrypoint_label="main_eval.main_worker_segmentation",
        n_classes=150,
    ),
    EvalSpec(
        workload_id="mobilevit_s_pascalvoc_seg",
        task_id="voc_seg",
        task_label="PASCAL VOC segmentation",
        model="mobilevit_s",
        model_family="mobilevit",
        dataset_name="PASCAL VOC 2012",
        primary_metric_name="mIoU",
        eval_kind="segmentation",
        local_config_rel="experiments/ml-cvnets/config/segmentation/pascal_voc/deeplabv3_mobilevit.yaml",
        recommended_local_weights_rel="weights/deeplabv3-mobilevitv1.pt",
        official_weights_url="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/deeplabv3-mobilevitv1.pt",
        official_config_url="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/deeplabv3-mobilevitv1.yaml",
        tutorial_url=SEGMENTATION_TUTORIAL_URL,
        model_zoo_url=MODEL_ZOO_URL,
        entrypoint_label="main_eval.main_worker_segmentation",
        n_classes=21,
    ),
)


def _config_path(spec: EvalSpec) -> Path:
    return ROOT / spec.local_config_rel


def _weights_path(spec: EvalSpec) -> Path:
    return ROOT / spec.recommended_local_weights_rel


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _get_nested(data: dict, *keys: str) -> object | None:
    cur: object = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _dataset_root_hint(spec: EvalSpec) -> Path:
    override = LOCAL_DATASET_ROOT_OVERRIDES.get(spec.workload_id)
    if override is not None:
        return Path(override)
    if spec.eval_kind == "classification":
        return DEFAULT_IMAGENET_ROOT
    cfg = _load_yaml(_config_path(spec))
    root_val = _get_nested(cfg, "dataset", "root_val")
    if root_val:
        return Path(str(root_val))
    return Path("/missing_dataset_root")


def _secondary_dataset_root_hint(spec: EvalSpec) -> Path | None:
    if spec.eval_kind != "segmentation":
        return None
    cfg = _load_yaml(_config_path(spec))
    extra = _get_nested(cfg, "dataset", "pascal", "coco_root_dir")
    if extra:
        return Path(str(extra))
    return None


def _command_to_text(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run_label(spec: EvalSpec) -> str:
    return f"audit_{spec.workload_id}"


def _is_pascal_voc_segmentation(spec: EvalSpec) -> bool:
    return spec.task_id == "voc_seg"


def _resolve_pascal_voc_root(dataset_root: Path) -> Path:
    if (dataset_root / "VOC2012").is_dir():
        return dataset_root
    candidate = dataset_root / "VOCdevkit"
    if (candidate / "VOC2012").is_dir():
        return candidate
    return dataset_root


def _link_dataset_path(source: Path, target: Path) -> None:
    """Create a lightweight dataset view without copying image payloads."""
    if target.exists() or target.is_symlink():
        return
    target.symlink_to(source, target_is_directory=source.is_dir())


def prepare_pascal_voc_eval_files(
    dataset_root: Path,
    *,
    max_eval_samples: int | None = None,
    subset_root: Path | None = None,
) -> Path:
    root = _resolve_pascal_voc_root(dataset_root)
    voc_root = root / "VOC2012"
    split_path = voc_root / "ImageSets" / "Segmentation" / "val.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing Pascal VOC val split file: {split_path}")

    lines: list[str] = []
    for raw_line in split_path.read_text(encoding="utf-8").splitlines():
        image_id = raw_line.strip()
        if not image_id:
            continue
        image_rel = Path("JPEGImages") / f"{image_id}.jpg"
        mask_rel = Path("SegmentationClass") / f"{image_id}.png"
        image_path = voc_root / image_rel
        mask_path = voc_root / mask_rel
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing Pascal VOC image file: {image_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing Pascal VOC mask file: {mask_path}")
        lines.append(f"{image_rel.as_posix()} {mask_rel.as_posix()}")

    if max_eval_samples is not None:
        if max_eval_samples < 1:
            raise ValueError("max_eval_samples must be positive when provided.")
        if subset_root is None:
            raise ValueError("subset_root is required when max_eval_samples is provided for Pascal VOC.")
        lines = lines[:max_eval_samples]
        subset_voc_root = subset_root / "VOC2012"
        output_dir = subset_voc_root / "list"
        output_dir.mkdir(parents=True, exist_ok=True)
        for dirname in ("JPEGImages", "SegmentationClass"):
            _link_dataset_path(voc_root / dirname, subset_voc_root / dirname)
        output_path = output_dir / "val.txt"
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return subset_root

    output_dir = voc_root / "list"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "val.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def _wrapper_command(
    spec: EvalSpec,
    *,
    dataset_root: Path,
    weights_path: Path,
    results_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "experiments" / "tools" / "run_cvnets_multitask_eval.py"),
        "run",
        "--workload",
        spec.workload_id,
        "--dataset_root",
        str(dataset_root),
        "--weights",
        str(weights_path),
        "--results_dir",
        str(results_dir),
        "--device",
        "mps",
        "--dry_run",
    ]


def build_run_command(
    spec: EvalSpec,
    *,
    dataset_root: Path | None = None,
    weights_path: Path | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    python_bin: str = sys.executable,
    max_eval_samples: int | None = None,
    device: str = "auto",
) -> list[str]:
    dataset_root = Path(dataset_root) if dataset_root is not None else _dataset_root_hint(spec)
    weights_path = Path(weights_path) if weights_path is not None else _weights_path(spec)
    results_dir = Path(results_dir)
    if _is_pascal_voc_segmentation(spec):
        dataset_root = _resolve_pascal_voc_root(dataset_root)

    if spec.eval_kind == "classification":
        output_csv = results_dir / f"{spec.workload_id}.csv"
        cmd = [
            python_bin,
            _display_path(ROOT / "experiments" / "accuracy" / "eval_cvnets_imagenet_noise.py"),
            "--imagenet_val",
            _display_path(dataset_root),
            "--results_csv",
            _display_path(output_csv),
            "--models",
            spec.model,
            "--cvnets_dir",
            _display_path(DEFAULT_CVNETS_ROOT),
            "--weights_override",
            f"{spec.model}={_display_path(weights_path)}",
            "--run_id",
            _run_label(spec),
            "--experiment_id",
            "EXT_TASK",
            "--workload",
            spec.workload_id,
            "--device",
            device,
            "--eval_batch_size",
            "0",
            "--workers",
            "-1",
            "--opencv_pipeline",
        ]
        if max_eval_samples is not None:
            cmd.extend(["--max_eval_samples", str(max_eval_samples)])
        return cmd

    if device == "auto":
        resolved_device, _device_note = resolve_device_preference(device)
    else:
        resolved_device = device
    bootstrap_lines = [
        "import os, sys",
        "os.environ.setdefault('CVNETS_TOLERATE_IMPORT_ERRORS', '1')",
        "os.environ.setdefault('CVNETS_SUPPRESS_OPTIONAL_IMPORT_WARNINGS', '1')",
        f"os.environ['CVNETS_DEVICE_BACKEND'] = {resolved_device!r}",
    ]
    if resolved_device != "cuda":
        bootstrap_lines.append("os.environ['CUDA_VISIBLE_DEVICES'] = ''")
    bootstrap_lines.append(f"sys.path.insert(0, {_display_path(DEFAULT_CVNETS_ROOT)!r})")
    if resolved_device == "mps":
        bootstrap_lines.extend(
            [
                f"sys.path.insert(0, {_display_path(EXPERIMENTS_ROOT)!r})",
                "import random",
                "import numpy as np",
                "import torch",
                "from exp_common.runtime import apply_torch_device",
                "from utils import logger",
                "from utils.ddp_utils import is_master",
                "import utils.common_utils as cvnets_common",
                "def _codex_mps_device_setup(opts):",
                "    random_seed = getattr(opts, 'common.seed', 0)",
                "    random.seed(random_seed)",
                "    torch.manual_seed(random_seed)",
                "    np.random.seed(random_seed)",
                "    if is_master(opts):",
                "        logger.log('Random seeds are set to {}'.format(random_seed))",
                "        logger.log('Using PyTorch version {}'.format(torch.__version__))",
                "        logger.log('Using MPS device via CVNETS_DEVICE_BACKEND=mps')",
                "    apply_torch_device(opts, 'mps')",
                "    return opts",
                "cvnets_common.device_setup = _codex_mps_device_setup",
            ]
        )
    bootstrap_lines.extend(
        [
            "import main_eval",
            (
                "main_eval.main_worker_detection(args=sys.argv[1:])"
                if spec.eval_kind == "detection"
                else "main_eval.main_worker_segmentation(args=sys.argv[1:])"
            ),
        ]
    )
    bootstrap = "\n".join(bootstrap_lines)
    override_kwargs = [
        f"common.results_loc={results_dir}",
        f"common.run_label={_run_label(spec)}",
        f"dataset.root_train={_display_path(dataset_root)}",
        f"dataset.root_val={_display_path(dataset_root)}",
        "dataset.workers=1",
    ]
    if resolved_device != "cuda":
        override_kwargs.append("common.mixed_precision=False")
    override_kwargs.extend(spec.extra_override_kwargs)
    cmd = [
        python_bin,
        "-c",
        bootstrap,
        "--common.config-file",
        _display_path(_config_path(spec)),
    ]
    secondary_root = None if _is_pascal_voc_segmentation(spec) else _secondary_dataset_root_hint(spec)
    if secondary_root is not None:
        override_kwargs.append(f"dataset.pascal.coco_root_dir={_display_path(secondary_root)}")
    if spec.eval_kind == "detection":
        override_kwargs.extend(
            [
                f"model.detection.pretrained={_display_path(weights_path)}",
                f"model.detection.n_classes={spec.n_classes or 81}",
            ]
        )
        cmd.extend(
            [
                "--evaluation.detection.resize-input-images",
                "--evaluation.detection.mode",
                "validation_set",
            ]
        )
    else:
        override_kwargs.extend(
            [
                f"model.segmentation.pretrained={_display_path(weights_path)}",
                f"model.segmentation.n_classes={spec.n_classes or 21}",
            ]
        )
        cmd.extend(
            [
                "--evaluation.segmentation.resize-input-images",
                "--evaluation.segmentation.mode",
                "validation_set",
            ]
        )
        if _is_pascal_voc_segmentation(spec):
            override_kwargs.append("dataset.pascal.use_coco_data=False")
    cmd.extend(["--common.override-kwargs", *override_kwargs])
    return cmd


def _readiness_level(
    *,
    runner_ready: bool,
    config_exists: bool,
    weights_exists: bool,
    dataset_exists: bool,
    extra_dataset_exists: bool,
) -> str:
    if runner_ready and config_exists and weights_exists and dataset_exists and extra_dataset_exists:
        return "local_ready"
    if runner_ready and config_exists and not dataset_exists:
        if weights_exists:
            return "chain_ready_missing_dataset"
        return "chain_ready_missing_dataset_and_weights"
    if runner_ready and config_exists and dataset_exists and not weights_exists:
        return "chain_ready_missing_weights"
    if runner_ready and not config_exists:
        return "missing_local_config"
    return "runner_not_ready"


def build_audit_row(spec: EvalSpec) -> dict[str, object]:
    cfg_path = _config_path(spec)
    weights_path = _weights_path(spec)
    dataset_root = _dataset_root_hint(spec)
    secondary_dataset_root = None if _is_pascal_voc_segmentation(spec) else _secondary_dataset_root_hint(spec)
    cvnets_ready = DEFAULT_CVNETS_ROOT.is_dir() and (DEFAULT_CVNETS_ROOT / "main_eval.py").is_file()
    runner_ready = cvnets_ready and (
        (ROOT / "experiments" / "accuracy" / "eval_cvnets_imagenet_noise.py").is_file()
        if spec.eval_kind == "classification"
        else True
    )
    config_exists = cfg_path.is_file()
    weights_exists = weights_path.is_file()
    dataset_exists = Path(dataset_root).exists()
    extra_dataset_exists = True if secondary_dataset_root is None else Path(secondary_dataset_root).exists()
    blocking_assets: list[str] = []
    if not config_exists:
        blocking_assets.append("local_config")
    if not weights_exists:
        blocking_assets.append("local_weights")
    if not dataset_exists:
        blocking_assets.append("dataset_root")
    if secondary_dataset_root is not None and not extra_dataset_exists:
        blocking_assets.append("secondary_dataset_root")
    if not runner_ready:
        blocking_assets.append("runner")
    readiness_level = _readiness_level(
        runner_ready=runner_ready,
        config_exists=config_exists,
        weights_exists=weights_exists,
        dataset_exists=dataset_exists,
        extra_dataset_exists=extra_dataset_exists,
    )
    command = build_run_command(spec)
    wrapper_cmd = _wrapper_command(
        spec,
        dataset_root=dataset_root,
        weights_path=weights_path,
        results_dir=DEFAULT_RESULTS_DIR,
    )
    return {
        "workload_id": spec.workload_id,
        "task_id": spec.task_id,
        "task_label": spec.task_label,
        "model": spec.model,
        "model_family": spec.model_family,
        "dataset_name": spec.dataset_name,
        "primary_metric_name": spec.primary_metric_name,
        "eval_kind": spec.eval_kind,
        "entrypoint_label": spec.entrypoint_label,
        "local_config_path": str(cfg_path),
        "local_config_exists": int(config_exists),
        "recommended_local_weights_path": str(weights_path),
        "local_weights_exists": int(weights_exists),
        "dataset_root_hint": str(dataset_root),
        "dataset_root_exists": int(dataset_exists),
        "secondary_dataset_root_hint": str(secondary_dataset_root or ""),
        "secondary_dataset_root_exists": int(extra_dataset_exists),
        "cvnets_root": str(DEFAULT_CVNETS_ROOT),
        "cvnets_root_exists": int(DEFAULT_CVNETS_ROOT.is_dir()),
        "runner_ready": int(runner_ready),
        "run_ready": int(
            runner_ready and config_exists and weights_exists and dataset_exists and extra_dataset_exists
        ),
        "readiness_level": readiness_level,
        "blocking_assets": ";".join(blocking_assets),
        "recommended_invocation": _command_to_text(wrapper_cmd),
        "resolved_backend_command": _command_to_text(command),
        "official_weights_url": spec.official_weights_url,
        "official_config_url": spec.official_config_url,
        "tutorial_url": spec.tutorial_url,
        "model_zoo_url": spec.model_zoo_url,
        "repo_url": REPO_URL,
    }


def build_audit_rows() -> list[dict[str, object]]:
    rows = [build_audit_row(spec) for spec in WORKLOAD_SPECS]
    rows.sort(key=lambda row: (str(row["task_id"]), str(row["workload_id"])))
    return rows


def write_audit_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No audit rows to write.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _spec_by_workload(workload_id: str) -> EvalSpec:
    for spec in WORKLOAD_SPECS:
        if spec.workload_id == workload_id:
            return spec
    raise SystemExit(f"Unknown workload_id: {workload_id}")


def run_workload(
    spec: EvalSpec,
    *,
    dataset_root: Path | None = None,
    weights_path: Path | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    python_bin: str = sys.executable,
    max_eval_samples: int | None = None,
    device: str = "auto",
    dry_run: bool = False,
) -> None:
    dataset_root = Path(dataset_root) if dataset_root is not None else _dataset_root_hint(spec)
    if not dry_run and _is_pascal_voc_segmentation(spec):
        subset_root = None
        if max_eval_samples is not None:
            subset_root = results_dir / "prepared_datasets" / f"{spec.workload_id}_max{max_eval_samples}"
        dataset_root = prepare_pascal_voc_eval_files(
            dataset_root,
            max_eval_samples=max_eval_samples,
            subset_root=subset_root,
        )
    cmd = build_run_command(
        spec,
        dataset_root=dataset_root,
        weights_path=weights_path,
        results_dir=results_dir,
        python_bin=python_bin,
        max_eval_samples=max_eval_samples,
        device=device,
    )
    print(_command_to_text(cmd))
    if dry_run:
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("CVNETS_TOLERATE_IMPORT_ERRORS", "1")
    resolved_device = resolve_device_preference(device)[0] if device == "auto" else device
    if resolved_device != "cuda":
        env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and optionally execute CVNets multi-task evaluation chains.")
    subparsers = parser.add_subparsers(dest="command")

    audit_parser = subparsers.add_parser("audit", help="Write a local readiness audit CSV.")
    audit_parser.add_argument("--out_csv", type=Path, default=DEFAULT_AUDIT_CSV)

    run_parser = subparsers.add_parser("run", help="Print or execute a workload command.")
    run_parser.add_argument("--workload", required=True)
    run_parser.add_argument("--dataset_root", type=Path, default=None)
    run_parser.add_argument("--weights", type=Path, default=None)
    run_parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    run_parser.add_argument("--python_bin", default=sys.executable)
    run_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    run_parser.add_argument("--max_eval_samples", type=int, default=None)
    run_parser.add_argument("--dry_run", action="store_true")

    list_parser = subparsers.add_parser("list", help="List known workload IDs.")
    list_parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    command = args.command or "audit"

    if command == "audit":
        rows = build_audit_rows()
        write_audit_csv(args.out_csv, rows)
        print(f"[cvnets-multitask] wrote audit csv: {args.out_csv}")
        return

    if command == "list":
        for spec in WORKLOAD_SPECS:
            if args.verbose:
                print(f"{spec.workload_id}: {spec.task_label} | {spec.model} | {spec.dataset_name}")
            else:
                print(spec.workload_id)
        return

    if command == "run":
        spec = _spec_by_workload(args.workload)
        run_workload(
            spec,
            dataset_root=args.dataset_root,
            weights_path=args.weights,
            results_dir=args.results_dir,
            python_bin=args.python_bin,
            max_eval_samples=args.max_eval_samples,
            device=args.device,
            dry_run=args.dry_run,
        )
        return

    raise SystemExit(f"Unsupported command: {command}")


if __name__ == "__main__":
    main()
