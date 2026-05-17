#!/usr/bin/env python3
"""Targeted local-AI repair for Fig.1 (Module 1 + Module 5 only).

Fix goals:
1) Module 1: redraw the second-half fully connected icon to remove noisy artifacts.
2) Module 5: add the missing input wedge on one branch.

All other regions are preserved by difference-mask compositing on the original image.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


M1_BOX = (720, 520, 1180, 1000)
M5_BOX = (2750, 2040, 3020, 2320)

M1_PROMPT = (
    "Redraw this neural fully-connected icon in clean IEEE technical vector style. "
    "Keep the same layout: 4 left nodes, 5 middle nodes, 3 right nodes, "
    "directional arrows left-to-right. Remove smudges/noise and keep stroke "
    "consistent. Preserve the same light-blue module background tone. No text."
)

M5_PROMPT = (
    "Keep this photonic-compute snippet unchanged except one fix: add the missing "
    "rainbow input wedge on the lower horizontal waveguide so it matches the upper "
    "row style and geometry. Preserve existing colors, line thickness, spacing, and "
    "right-side ring segment. No extra symbols or text."
)


def _run_nanobanana(ref_image: Path, out_image: Path, prompt: str, timeout_s: int = 180) -> None:
    cmd = [
        "python3",
        "scripts/nanobanana_pro_gen.py",
        "--provider",
        "gemini",
        "--model",
        "nanobanana-2",
        "--ref-image",
        str(ref_image),
        "--prompt",
        prompt,
        "--aspect-ratio",
        "1:1",
        "--resolution",
        "1K",
        "--n",
        "1",
        "--out",
        str(out_image),
        "--timeout",
        str(timeout_s),
        "--max-wait",
        "240",
    ]
    subprocess.run(cmd, check=True)


def _paste_diff_patch(
    canvas: Image.Image,
    ai_patch: Image.Image,
    box: tuple[int, int, int, int],
    threshold: int,
    border_clip: int = 0,
) -> None:
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0

    patch = ai_patch.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    base_crop = canvas.crop(box).convert("RGB")

    base_np = np.asarray(base_crop, dtype=np.int16)
    patch_np = np.asarray(patch, dtype=np.int16)
    diff = np.max(np.abs(patch_np - base_np), axis=2)

    mask = np.where(diff >= threshold, 255, 0).astype(np.uint8)
    if border_clip > 0:
        # Keep patch boundary fully untouched to avoid accidental seam edits.
        mask[:border_clip, :] = 0
        mask[-border_clip:, :] = 0
        mask[:, :border_clip] = 0
        mask[:, -border_clip:] = 0
    mask_img = Image.fromarray(mask)

    canvas.paste(patch, box, mask_img)


def main() -> int:
    ap = argparse.ArgumentParser(description="Repair Fig.1 Module1/Module5 local defects with local AI patches.")
    ap.add_argument(
        "--input",
        default="figures/fig1_ai_text_readability_final.png",
        help="Input Fig.1 PNG path.",
    )
    ap.add_argument(
        "--output",
        default="figures/fig1_ai_text_readability_final_fixed_m1m5.png",
        help="Output PNG path.",
    )
    ap.add_argument(
        "--work-dir",
        default="tmp/fig1_local_ai_fix_m1m5",
        help="Directory for intermediate crops/patches.",
    )
    ap.add_argument(
        "--m1-ai",
        default=None,
        help="Optional pre-generated Module 1 patch image; skip generating M1 when provided.",
    )
    ap.add_argument(
        "--m5-ai",
        default=None,
        help="Optional pre-generated Module 5 patch image; skip generating M5 when provided.",
    )
    ap.add_argument(
        "--m1-threshold",
        type=int,
        default=10,
        help="Diff threshold for Module 1 compositing mask.",
    )
    ap.add_argument(
        "--m5-threshold",
        type=int,
        default=8,
        help="Diff threshold for Module 5 compositing mask.",
    )
    ap.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep temporary files for inspection.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    work_dir = Path(args.work_dir)

    if not in_path.is_file():
        raise SystemExit(f"Input image not found: {in_path}")

    work_dir.mkdir(parents=True, exist_ok=True)
    m1_ref = work_dir / "m1_fc_ref.png"
    m5_ref = work_dir / "m5_missing_ref.png"
    m1_ai = Path(args.m1_ai) if args.m1_ai else work_dir / "m1_fc_ai.png"
    m5_ai = Path(args.m5_ai) if args.m5_ai else work_dir / "m5_missing_ai.png"

    base = Image.open(in_path).convert("RGB")
    base.crop(M1_BOX).save(m1_ref)
    base.crop(M5_BOX).save(m5_ref)

    if not m1_ai.is_file():
        _run_nanobanana(m1_ref, m1_ai, M1_PROMPT)
    if not m5_ai.is_file():
        _run_nanobanana(m5_ref, m5_ai, M5_PROMPT)

    m1_patch = Image.open(m1_ai)
    m5_patch = Image.open(m5_ai)

    # Module 1 redraw needs broader replacement; Module 5 needs tighter local edits.
    _paste_diff_patch(base, m1_patch, M1_BOX, threshold=args.m1_threshold, border_clip=18)
    _paste_diff_patch(base, m5_patch, M5_BOX, threshold=args.m5_threshold, border_clip=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.save(out_path)
    print(out_path)

    if not args.keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
