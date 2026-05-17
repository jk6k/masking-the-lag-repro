#!/usr/bin/env python3
"""Compatibility defaults for DET main-text support assets."""

from __future__ import annotations

try:
    from experiments.tools.path_policy import MAIN_PROJECT_REPORT_DATA_DIR, MAIN_PROJECT_REPORT_TABLE_DIR
except ModuleNotFoundError:  # direct script execution
    from path_policy import MAIN_PROJECT_REPORT_DATA_DIR, MAIN_PROJECT_REPORT_TABLE_DIR


DEFAULT_OUT_SUMMARY = MAIN_PROJECT_REPORT_DATA_DIR / "det_main_text_summary.md"
DEFAULT_OUT_NOTE = MAIN_PROJECT_REPORT_DATA_DIR / "det_main_text_note.md"
DEFAULT_TABLE_CSV = MAIN_PROJECT_REPORT_TABLE_DIR / "det_main_text_table.csv"
DEFAULT_TABLE_TEX = MAIN_PROJECT_REPORT_TABLE_DIR / "det_main_text_table.tex"
