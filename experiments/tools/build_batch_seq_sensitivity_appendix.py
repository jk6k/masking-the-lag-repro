#!/usr/bin/env python3
"""Compatibility defaults for batch/sequence sensitivity appendix assets."""

from __future__ import annotations

try:
    from experiments.tools.path_policy import MAIN_PROJECT_REPORT_DATA_DIR
except ModuleNotFoundError:  # direct script execution
    from path_policy import MAIN_PROJECT_REPORT_DATA_DIR


DEFAULT_SUMMARY = MAIN_PROJECT_REPORT_DATA_DIR / "batch_seq_sensitivity_summary.csv"
DEFAULT_NOTE = MAIN_PROJECT_REPORT_DATA_DIR / "batch_seq_sensitivity_note.md"
