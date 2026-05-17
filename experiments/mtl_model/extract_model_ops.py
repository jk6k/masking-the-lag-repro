"""MTL wrapper for op-shape extraction."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hpat_model.extract_model_ops import main


if __name__ == "__main__":
    out_dir_flag = "--out_dir"
    if out_dir_flag not in sys.argv:
        default_out = Path(__file__).resolve().parent / "ops"
        sys.argv.extend([out_dir_flag, str(default_out)])
    main()
