#!/usr/bin/env python3
"""Compatibility defaults for CVNets multi-task readiness appendix assets."""

from __future__ import annotations

try:
    from experiments.tools.path_policy import MAIN_PROJECT_REPORT_DATA_DIR, MAIN_PROJECT_REPORT_FIG_DIR
except ModuleNotFoundError:  # direct script execution
    from path_policy import MAIN_PROJECT_REPORT_DATA_DIR, MAIN_PROJECT_REPORT_FIG_DIR


DEFAULT_OUT_DATA_DIR = MAIN_PROJECT_REPORT_DATA_DIR
DEFAULT_OUT_FIG_DIR = MAIN_PROJECT_REPORT_FIG_DIR
