#!/usr/bin/env python3
"""Build an ImageNet baseline-closure note from the CVNets reproduction audit and official-eval probes."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_SUMMARY_CSV = ROOT / "AICAS" / "assets" / "candidate_data" / "cvnets_reproduction_audit_summary_20260308_cvnetsrepro.csv"
DEFAULT_OUT_DIR = ROOT / "AICAS" / "assets" / "candidate_data"
DEFAULT_TAG = "20260310"


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def _extract_probe_metadata(log_path: Path) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    resize_match = re.search(r"Resize\(size=(\d+),.*?CenterCrop\(size=\(h=(\d+), w=(\d+)\)\)", text, re.DOTALL)
    sample_match = re.search(r"Total number of samples:\s*(\d+)", text)
    ckpt_match = re.search(r"--model\.classification\.pretrained\s+(\S+)", text)
    entrypoint = "main_eval.main_worker" if "main_eval.main_worker" in text else "unknown"
    mixed_precision = "False" if "common.mixed_precision': False" in text else "unknown"
    return {
        "probe_log": str(log_path),
        "entrypoint": entrypoint,
        "resize_size": int(resize_match.group(1)) if resize_match else None,
        "crop_h": int(resize_match.group(2)) if resize_match else None,
        "crop_w": int(resize_match.group(3)) if resize_match else None,
        "num_samples": int(sample_match.group(1)) if sample_match else None,
        "checkpoint_path": ckpt_match.group(1) if ckpt_match else "",
        "mixed_precision": mixed_precision,
    }


def _build_rows(audit_summary_csv: Path) -> list[dict[str, object]]:
    summary = pd.read_csv(audit_summary_csv)
    cls = summary.loc[summary["task_group"] == "classification"].copy()
    rows: list[dict[str, object]] = []
    for item in cls.itertuples(index=False):
        probe_meta = _extract_probe_metadata(Path(item.probe_source))
        rows.append(
            {
                "model": item.model,
                "official_model_zoo_top1_pct": float(item.official_metric_pct),
                "paper_local_top1_pct": float(item.local_metric_pct),
                "official_probe_top1_pct": float(item.official_eval_probe_top1_pct),
                "official_probe_top5_pct": float(item.official_eval_probe_top5_pct),
                "paper_gap_vs_official_pp": float(item.delta_local_minus_official_pp),
                "probe_gap_vs_official_pp": float(item.probe_delta_vs_official_pp),
                "probe_minus_paper_local_pp": float(item.probe_delta_vs_paper_local_pp),
                "official_source": str(item.official_source),
                "paper_local_source": str(item.local_source),
                "probe_log": probe_meta["probe_log"],
                "entrypoint": probe_meta["entrypoint"],
                "checkpoint_path": probe_meta["checkpoint_path"],
                "resize_size": probe_meta["resize_size"],
                "crop_h": probe_meta["crop_h"],
                "crop_w": probe_meta["crop_w"],
                "num_samples": probe_meta["num_samples"],
                "mixed_precision": probe_meta["mixed_precision"],
                "alignment_band": str(item.alignment_band),
            }
        )
    return rows


def _write_report(out_path: Path, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows).sort_values("model")
    max_paper_gap = float(df["paper_gap_vs_official_pp"].abs().max())
    min_paper_gap = float(df["paper_gap_vs_official_pp"].abs().min())
    max_probe_gap = float(df["probe_gap_vs_official_pp"].abs().max())

    lines = [
        "# ImageNet Baseline Closure Report (20260310)",
        "",
        "Scope",
        "- This note covers only the MobileViTv1 ImageNet-1k classification baseline used by the paper's local E0/E6 chain.",
        "- Detection and segmentation baseline closure is already near-official; the unresolved baseline problem is classification only.",
        "",
        "What was checked",
        "- checkpoint path for each official-entrypoint rerun",
        "- official CVNets evaluation entrypoint and preprocessing trace",
        "- eval split size and dataset routing",
        "- mixed-precision status in the official-entrypoint probe",
        "",
        "Closure findings",
    ]
    for row in rows:
        lines.extend(
            [
                (
                    f"- `{row['model']}`: paper local `{_fmt(row['paper_local_top1_pct'], 3)}%`, "
                    f"official-entrypoint probe `{_fmt(row['official_probe_top1_pct'], 3)}%`, "
                    f"official model-zoo `{_fmt(row['official_model_zoo_top1_pct'], 3)}%`."
                ),
                (
                    f"  probe path: checkpoint=`{row['checkpoint_path']}`, entrypoint=`{row['entrypoint']}`, "
                    f"transform=`Resize({row['resize_size']}) + CenterCrop({row['crop_h']})`, "
                    f"samples=`{row['num_samples']}`, mixed_precision=`{row['mixed_precision']}`."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "Interpretation",
            (
                f"- Normalizing to the official CVNets evaluation entrypoint narrows part of the gap "
                f"(paper-local to probe gain ranges from `{_fmt(df['probe_minus_paper_local_pp'].min(), 2)}` to "
                f"`{_fmt(df['probe_minus_paper_local_pp'].max(), 2)} pp`), but the remaining probe-to-official gap is still "
                f"`{_fmt(df['probe_gap_vs_official_pp'].abs().min(), 2)}-{_fmt(max_probe_gap, 2)} pp`."
            ),
            (
                f"- The original paper-local baseline is still `{_fmt(min_paper_gap, 2)}-{_fmt(max_paper_gap, 2)} pp` below the "
                "official legacy model-zoo, so the mismatch is not explained by custom runner differences alone."
            ),
            "- The probe logs show standard official preprocessing on the full 50k ImageNet val set, which rules out a simple resize/crop or split-routing bug as the sole cause.",
            "- What remains unresolved is checkpoint provenance / legacy-weight equivalence for the MobileViTv1 ImageNet chain, so this baseline is not closed to official parity.",
            "",
            "Manuscript consequence",
            "- The paper should describe ImageNet classification as a degraded local reference-chain case study, not as an externally matched CVNets reproduction.",
            "- External-benchmark language should stay out of the headline, abstract, and discussion for the classification path.",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ImageNet baseline-closure report.")
    parser.add_argument("--audit_summary_csv", type=Path, default=DEFAULT_AUDIT_SUMMARY_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    args = parser.parse_args()

    rows = _build_rows(args.audit_summary_csv)
    csv_path = args.out_dir / f"imagenet_baseline_closure_summary_{args.tag}.csv"
    md_path = args.out_dir / f"imagenet_baseline_closure_report_{args.tag}.md"
    _write_csv(
        csv_path,
        rows,
        [
            "model",
            "official_model_zoo_top1_pct",
            "paper_local_top1_pct",
            "official_probe_top1_pct",
            "official_probe_top5_pct",
            "paper_gap_vs_official_pp",
            "probe_gap_vs_official_pp",
            "probe_minus_paper_local_pp",
            "official_source",
            "paper_local_source",
            "probe_log",
            "entrypoint",
            "checkpoint_path",
            "resize_size",
            "crop_h",
            "crop_w",
            "num_samples",
            "mixed_precision",
            "alignment_band",
        ],
    )
    _write_report(md_path, rows)


if __name__ == "__main__":
    main()
