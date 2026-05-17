"""Asset registry and lightweight status helpers for local development."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from exp_common.model_specs import MODEL_SPECS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_DIR = ROOT / "weights"


@dataclass(frozen=True)
class WeightSpec:
    key: str
    filename: str
    url: str
    group: str
    description: str


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    path: Path
    description: str
    meta_path: Path | None = None


WEIGHT_SPECS: tuple[WeightSpec, ...] = (
    WeightSpec(
        key="mobilevit_xxs",
        filename="mobilevit_xxs.pt",
        url=str(MODEL_SPECS["mobilevit_xxs"]["weights"]),
        group="classification",
        description="MobileViT-XXS ImageNet classification checkpoint",
    ),
    WeightSpec(
        key="mobilevit_xs",
        filename="mobilevit_xs.pt",
        url=str(MODEL_SPECS["mobilevit_xs"]["weights"]),
        group="classification",
        description="MobileViT-XS ImageNet classification checkpoint",
    ),
    WeightSpec(
        key="mobilevit_s",
        filename="mobilevit_s.pt",
        url=str(MODEL_SPECS["mobilevit_s"]["weights"]),
        group="classification",
        description="MobileViT-S ImageNet classification checkpoint",
    ),
    WeightSpec(
        key="resnet_50",
        filename="resnet-50.pt",
        url=str(MODEL_SPECS["resnet_50"]["weights"]),
        group="classification",
        description="ResNet-50 ImageNet classification checkpoint",
    ),
    WeightSpec(
        key="mobilevitv2_1p0_coco_det",
        filename="coco-ssd-mobilevitv2-1.0.pt",
        url="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-1.0.pt",
        group="multitask",
        description="MobileViTv2-1.0 COCO detection checkpoint",
    ),
    WeightSpec(
        key="mobilevitv2_1p0_ade20k_seg",
        filename="deeplabv3-mobilevitv2-1.0.pt",
        url="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-1.0.pt",
        group="multitask",
        description="MobileViTv2-1.0 ADE20K segmentation checkpoint",
    ),
    WeightSpec(
        key="mobilevit_s_pascalvoc_seg",
        filename="deeplabv3-mobilevitv1.pt",
        url="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/deeplabv3-mobilevitv1.pt",
        group="multitask",
        description="MobileViT-S PASCAL VOC segmentation checkpoint",
    ),
)


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="imagenet_val",
        path=ROOT / "experiments" / "datasets" / "imagenet" / "val",
        description="ImageNet-1k validation tree for accuracy evaluation",
        meta_path=ROOT / "experiments" / "datasets" / "imagenet" / "val" / "_val_manifest.csv",
    ),
    DatasetSpec(
        key="coco_hf_val5000",
        path=ROOT / "experiments" / "datasets" / "coco_hf_val5000",
        description="COCO validation mirror for multitask detection evaluation",
        meta_path=ROOT / "experiments" / "datasets" / "coco_hf_val5000" / "coco_hf_val5000_meta.json",
    ),
    DatasetSpec(
        key="ade20k_hf_val2000",
        path=ROOT / "experiments" / "datasets" / "ade20k_hf_val2000" / "ADEChallengeData2016",
        description="ADE20K validation mirror for multitask segmentation evaluation",
        meta_path=ROOT / "experiments" / "datasets" / "ade20k_hf_val2000" / "ade20k_hf_val2000_meta.json",
    ),
    DatasetSpec(
        key="pascal_voc_hf",
        path=ROOT / "experiments" / "datasets" / "pascal_voc_hf" / "VOCdevkit",
        description="PASCAL VOC validation mirror for multitask segmentation evaluation",
        meta_path=ROOT / "experiments" / "datasets" / "pascal_voc_hf" / "pascal_voc_hf_meta.json",
    ),
    DatasetSpec(
        key="imagenet_r_even256",
        path=ROOT / "experiments" / "datasets" / "imagenet_r_even256",
        description="ImageNet-R robustness subset for broader-task validation",
        meta_path=ROOT / "experiments" / "datasets" / "imagenet_r_even256" / "imagenet_r_meta.json",
    ),
)


def iter_weight_specs(
    *,
    keys: Iterable[str] | None = None,
    groups: Iterable[str] | None = None,
) -> list[WeightSpec]:
    wanted_keys = {item.strip() for item in keys or [] if str(item).strip()}
    wanted_groups = {item.strip() for item in groups or [] if str(item).strip()}
    specs = list(WEIGHT_SPECS)
    if wanted_keys:
        specs = [spec for spec in specs if spec.key in wanted_keys or spec.filename in wanted_keys]
    if wanted_groups:
        specs = [spec for spec in specs if spec.group in wanted_groups]
    return specs


def resolve_weight_path(spec: WeightSpec, weights_dir: str | Path | None = None) -> Path:
    if weights_dir is not None and str(weights_dir).startswith("/"):
        return PurePosixPath(str(weights_dir)) / spec.filename
    base = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
    return base / spec.filename


def summarize_meta_payload(payload: dict[str, object]) -> str:
    chunks: list[str] = []
    image_count = payload.get("image_count") or payload.get("sample_count_written")
    if image_count is not None:
        chunks.append(f"{image_count} images")
    annotation_count = payload.get("annotation_count_written")
    if annotation_count is not None:
        chunks.append(f"{annotation_count} annotations")
    class_count = payload.get("class_count_written") or payload.get("category_count_present")
    if class_count is not None:
        chunks.append(f"{class_count} classes")
    failure_count = payload.get("failure_count")
    if failure_count not in (None, 0, "0"):
        chunks.append(f"{failure_count} failures")
    split_name = payload.get("split_name")
    if split_name:
        chunks.append(f"split={split_name}")
    return ", ".join(str(chunk) for chunk in chunks if str(chunk).strip())


def _summarize_imagenet_tree(root: Path, manifest_path: Path | None = None) -> str:
    class_count = len([path for path in root.iterdir() if path.is_dir()]) if root.is_dir() else 0
    image_count = None
    if manifest_path and manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            image_count = max(0, sum(1 for _ in handle) - 1)
    chunks = []
    if class_count:
        chunks.append(f"{class_count} classes")
    if image_count is not None:
        chunks.append(f"{image_count} images")
    return ", ".join(chunks) if chunks else "present"


def collect_weight_status(weights_dir: str | Path | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in WEIGHT_SPECS:
        path = resolve_weight_path(spec, weights_dir)
        local_path = Path(str(path))
        rows.append(
            {
                "key": spec.key,
                "group": spec.group,
                "description": spec.description,
                "path": path,
                "exists": local_path.is_file(),
                "url": spec.url,
            }
        )
    return rows


def collect_dataset_status() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in DATASET_SPECS:
        exists = spec.path.exists()
        summary = "missing"
        if exists:
            if spec.key == "imagenet_val":
                summary = _summarize_imagenet_tree(spec.path, spec.meta_path)
            elif spec.meta_path and spec.meta_path.is_file():
                try:
                    payload = json.loads(spec.meta_path.read_text(encoding="utf-8"))
                    summary = summarize_meta_payload(payload) or "present"
                except Exception:
                    summary = "present"
            else:
                summary = "present"
        rows.append(
            {
                "key": spec.key,
                "description": spec.description,
                "path": spec.path,
                "exists": exists,
                "summary": summary,
            }
        )
    return rows


__all__ = [
    "DATASET_SPECS",
    "DEFAULT_WEIGHTS_DIR",
    "WEIGHT_SPECS",
    "DatasetSpec",
    "WeightSpec",
    "collect_dataset_status",
    "collect_weight_status",
    "iter_weight_specs",
    "resolve_weight_path",
    "summarize_meta_payload",
]
