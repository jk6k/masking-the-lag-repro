#!/usr/bin/env python3
"""Build a compact per-model headline appendix summary.

The historical report chain expected this module to exist as a small companion
to ``build_headline_statistics_report.py``.  Keep the implementation narrow:
read the per-model headline CSV, write a normalized appendix CSV, and optionally
write a short Markdown note.  No new experiment evidence is generated here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from experiments.tools.path_policy import (
        MAIN_PROJECT_REPORT_DATA_DIR,
        MAIN_PROJECT_REPORT_FIG_DIR,
        assert_main_project_path,
    )
except ModuleNotFoundError:  # direct script execution
    from path_policy import (
        MAIN_PROJECT_REPORT_DATA_DIR,
        MAIN_PROJECT_REPORT_FIG_DIR,
        assert_main_project_path,
    )


DEFAULT_TAG = "20260310"
DEFAULT_OUT_DATA_DIR = MAIN_PROJECT_REPORT_DATA_DIR
DEFAULT_OUT_FIG_DIR = MAIN_PROJECT_REPORT_FIG_DIR
DEFAULT_ACCURACY_CSV = DEFAULT_OUT_DATA_DIR / f"headline_statistics_per_model_{DEFAULT_TAG}.csv"


def _load_per_model_rows(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing per-model headline CSV: {path}")
    df = pd.read_csv(path)
    required = {"experiment_id", "model"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    return df.sort_values(["experiment_id", "model"]).reset_index(drop=True)


def _write_note(path: Path, *, source_csv: Path, appendix_csv: Path, row_count: int) -> None:
    lines = [
        "# Per-Model Headline Appendix",
        "",
        f"- Source CSV: `{source_csv}`",
        f"- Appendix CSV: `{appendix_csv}`",
        f"- Row count: `{row_count}`",
        "",
        "This appendix table is a reader-facing summary of existing per-model headline rows. "
        "It does not create new experiment evidence or change claim boundaries.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_appendix(*, accuracy_csv: Path, out_data_dir: Path, tag: str) -> dict[str, Path]:
    accuracy_csv = assert_main_project_path(accuracy_csv, arg_name="--accuracy_csv")
    out_data_dir = assert_main_project_path(out_data_dir, arg_name="--out_data_dir")
    out_data_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_per_model_rows(accuracy_csv)
    appendix_csv = out_data_dir / f"per_model_headline_appendix_{tag}.csv"
    note_md = out_data_dir / f"per_model_headline_appendix_{tag}.md"
    rows.to_csv(appendix_csv, index=False)
    _write_note(note_md, source_csv=accuracy_csv, appendix_csv=appendix_csv, row_count=len(rows))
    return {"appendix_csv": appendix_csv, "note_md": note_md}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a per-model headline appendix summary.")
    parser.add_argument("--accuracy_csv", type=Path, default=DEFAULT_ACCURACY_CSV)
    parser.add_argument("--out_data_dir", type=Path, default=DEFAULT_OUT_DATA_DIR)
    parser.add_argument("--out_fig_dir", type=Path, default=DEFAULT_OUT_FIG_DIR)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    args = parser.parse_args()

    assert_main_project_path(args.out_fig_dir, arg_name="--out_fig_dir").mkdir(parents=True, exist_ok=True)
    outputs = build_appendix(accuracy_csv=args.accuracy_csv, out_data_dir=args.out_data_dir, tag=args.tag)
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
