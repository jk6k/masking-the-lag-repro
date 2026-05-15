#!/usr/bin/env python3
"""Render the data figures for the 20260516 TETC submission figure candidate."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from types import SimpleNamespace

import render_suds_tetc_figures as base
import render_suds_tetc_scheduler_evidence as scheduler


OUT_TAG = "20260516_submission_figure_pack"
OUT_DIR = base.REPO_ROOT / f"figures/suds_tetc_{OUT_TAG}"

STEM_MAP = {
    "Fig2_AccuracyEDPPareto": "Fig4_AccuracyEDPPareto",
    "Fig3_EnergyBreakdown": "Fig5_EnergyBreakdown",
    "Fig4_ConservativeParetoAccuracy": "Fig6_ConservativeParetoAccuracy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=base.SUMMARY_CSV)
    parser.add_argument("--conservative-json", type=Path, default=base.CONSERVATIVE_JSON)
    parser.add_argument("--scheduler-ablation-csv", type=Path, default=scheduler.DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def copy_rendered_stems(scratch_dir: Path, output_dir: Path) -> list[Path]:
    written: list[Path] = []
    for old_stem, new_stem in STEM_MAP.items():
        for ext in ("pdf", "svg", "png"):
            src = scratch_dir / f"{old_stem}.{ext}"
            dst = output_dir / f"{new_stem}.{ext}"
            if not src.is_file():
                raise SystemExit(f"missing rendered artifact: {src}")
            shutil.copy2(src, dst)
            written.append(dst)
    return written


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = args.output_dir / ".data_rerender"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True)

    render_args = SimpleNamespace(
        summary_csv=args.summary_csv,
        conservative_json=args.conservative_json,
        output_dir=scratch_dir,
    )
    rows = base.load_csv(args.summary_csv)
    base.render_fig2(render_args, rows)
    base.render_fig3(render_args, rows)
    base.render_fig4(render_args)
    written = copy_rendered_stems(scratch_dir, args.output_dir)
    shutil.rmtree(scratch_dir)
    scheduler_rows = scheduler.load_rows(args.scheduler_ablation_csv)
    scheduler_by_key = scheduler.selected_rows(scheduler_rows)
    source = scheduler.write_source_summary(scheduler_by_key, args.output_dir)
    scheduler_outputs = scheduler.render_figure(scheduler_by_key, args.output_dir)

    for path in written:
        print(f"wrote {base.rel(path)}")
    print(f"source={scheduler.rel(source)}")
    for ext, path in scheduler_outputs.items():
        print(f"{ext}={path}")


if __name__ == "__main__":
    main()
