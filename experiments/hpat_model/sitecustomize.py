"""Auto-select the repo virtualenv for HPAT scripts when `yaml` is unavailable."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repo_python_bootstrap import maybe_reexec_for_module


maybe_reexec_for_module("yaml", anchor=Path(__file__))
