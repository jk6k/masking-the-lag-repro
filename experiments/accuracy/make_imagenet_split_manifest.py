#!/usr/bin/env python3
"""Create deterministic ImageNet validation split manifests.

The evaluators accept CSV manifests with at least ``path,label`` columns. This
helper keeps the label assignment identical to the built-in ImageNet directory
scanner: synset directories are sorted lexicographically and assigned labels
0..999 in that order.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def _resolve_val_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    nested = root / "val"
    if nested.is_dir() and not any(child.name.startswith("n") for child in root.iterdir() if child.is_dir()):
        root = nested
    if not root.is_dir():
        raise SystemExit(f"ImageNet val directory not found: {root}")
    return root


def _class_dirs(root: Path) -> list[Path]:
    dirs = sorted([child for child in root.iterdir() if child.is_dir()], key=lambda p: p.name)
    synsets = [child for child in dirs if len(child.name) == 9 and child.name[0] == "n" and child.name[1:].isdigit()]
    if len(synsets) != len(dirs):
        raise SystemExit(
            "ImageNet val root must contain synset-named directories like n01440764."
        )
    if not synsets:
        raise SystemExit(f"No synset directories found under {root}")
    return synsets


def _manifest_path(path: Path, *, root: Path, mode: str) -> str:
    if mode == "absolute":
        return str(path.resolve())
    if mode == "relative-to-val":
        return path.resolve().relative_to(root).as_posix()
    raise SystemExit(f"Unsupported path mode: {mode}")


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "class_name", "split"])
        writer.writeheader()
        writer.writerows(rows)


def build_manifests(
    imagenet_val: Path,
    out_dir: Path,
    *,
    calib_fraction: float,
    seed: int,
    path_mode: str,
) -> dict[str, object]:
    if not (0.0 < calib_fraction < 1.0):
        raise SystemExit("--calib_fraction must be between 0 and 1")

    root = _resolve_val_root(imagenet_val)
    rng = random.Random(seed)
    calib_rows: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []
    all_rows: list[dict[str, str]] = []

    for label, class_dir in enumerate(_class_dirs(root)):
        samples = sorted([path for path in class_dir.rglob("*") if path.is_file() and _is_image(path)])
        if not samples:
            raise SystemExit(f"No images found for class {class_dir.name}: {class_dir}")
        shuffled = list(samples)
        rng.shuffle(shuffled)
        calib_count = max(1, int(round(len(shuffled) * calib_fraction)))
        calib_set = {path.resolve() for path in shuffled[:calib_count]}
        for sample in samples:
            split = "calib" if sample.resolve() in calib_set else "eval"
            row = {
                "path": _manifest_path(sample, root=root, mode=path_mode),
                "label": str(label),
                "class_name": class_dir.name,
                "split": split,
            }
            all_rows.append(row)
            if split == "calib":
                calib_rows.append(row)
            else:
                eval_rows.append(row)

    if not calib_rows or not eval_rows:
        raise SystemExit("Split produced an empty calib or eval manifest")

    out_dir.mkdir(parents=True, exist_ok=True)
    calib_csv = out_dir / "imagenet_val_calib.csv"
    eval_csv = out_dir / "imagenet_val_eval.csv"
    all_csv = out_dir / "imagenet_val_all.csv"
    _write_rows(calib_csv, calib_rows)
    _write_rows(eval_csv, eval_rows)
    _write_rows(all_csv, all_rows)

    return {
        "imagenet_val": str(root),
        "seed": seed,
        "calib_fraction": calib_fraction,
        "path_mode": path_mode,
        "class_count": len(_class_dirs(root)),
        "total_samples": len(all_rows),
        "calib_samples": len(calib_rows),
        "eval_samples": len(eval_rows),
        "calib_csv": str(calib_csv),
        "eval_csv": str(eval_csv),
        "all_csv": str(all_csv),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenet_val", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--calib_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--path_mode",
        choices=("absolute", "relative-to-val"),
        default="absolute",
        help="Use absolute image paths, or paths relative to --imagenet_val.",
    )
    args = parser.parse_args()

    payload = build_manifests(
        args.imagenet_val,
        args.out_dir,
        calib_fraction=args.calib_fraction,
        seed=args.seed,
        path_mode=args.path_mode,
    )
    for key, value in payload.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
