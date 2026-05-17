#!/usr/bin/env python3
"""Optimize Fig1 Module-5 local alignment by deterministic wedge transfer.

This script removes residual seam artifacts from the local AI patch area and
re-injects the missing branch wedge using a row-to-row difference mask from
the original image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


# Region that may contain visible patch seams from previous local edits.
RESET_BOX = (2720, 1860, 3040, 2360)

# Source wedge in original fig (an existing branch), copied down by DY to fill
# the missing branch while preserving exact style/anti-aliasing.
SRC_BOX = (2768, 2022, 2940, 2148)
DY = 154
DIFF_THRESHOLD = 8


def _largest_component(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask

    visited = np.zeros(mask.shape, dtype=bool)
    best_component: list[tuple[int, int]] = []
    height, width = mask.shape

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            component: list[tuple[int, int]] = []

            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            if len(component) > len(best_component):
                best_component = component

    out = np.zeros(mask.shape, dtype=bool)
    for y, x in best_component:
        out[y, x] = True
    return out


def optimize(current_path: Path, orig_path: Path, out_path: Path, *, alpha_blur: float = 0.0) -> None:
    cur = Image.open(current_path).convert("RGB")
    orig = Image.open(orig_path).convert("RGB")
    orig_np = np.asarray(orig)

    out = cur.copy()
    # 1) Restore local area from original to remove rectangular seam residue.
    out.paste(orig.crop(RESET_BOX), RESET_BOX)

    # 2) Extract only the wedge shape by comparing source row and target row in original.
    x0, y0, x1, y1 = SRC_BOX
    src = orig_np[y0:y1, x0:x1]
    dst = orig_np[y0 + DY : y1 + DY, x0:x1]
    diff = np.max(np.abs(src.astype(np.int16) - dst.astype(np.int16)), axis=2)
    mask = diff > DIFF_THRESHOLD
    mask = _largest_component(mask)

    patch = Image.fromarray(src)
    alpha = Image.fromarray(mask.astype(np.uint8) * 255)
    if alpha_blur > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(alpha_blur))
    out.paste(patch, (x0, y0 + DY, x1, y1 + DY), alpha)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Optimize Fig1 Module-5 wedge alignment and remove seam artifacts.")
    ap.add_argument(
        "--current",
        default="figures/paper_figures_20260403_det_sparse_mlx_final_cpu_context_repair_xs_xxs_dense_successor_freeze/Fig1_SystemArchitecture.png",
        help="Current Fig1 image path.",
    )
    ap.add_argument(
        "--orig",
        default="figures/paper_figures_20260403_det_sparse_mlx_final_cpu_context_repair_xs_xxs_dense_successor_freeze/Fig1_SystemArchitecture.png",
        help="Original baseline Fig1 image path.",
    )
    ap.add_argument(
        "--out",
        default="tmp/fig1_local_mask_repair/Fig1_SystemArchitecture.fixed.png",
        help="Output Fig1 image path.",
    )
    ap.add_argument(
        "--alpha-blur",
        type=float,
        default=0.0,
        help="Optional Gaussian blur radius on alpha mask. Default 0.0 for strict template alignment.",
    )
    args = ap.parse_args()

    optimize(Path(args.current), Path(args.orig), Path(args.out), alpha_blur=args.alpha_blur)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
