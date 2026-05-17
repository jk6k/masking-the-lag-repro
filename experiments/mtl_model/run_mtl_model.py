"""MTL wrapper for single-config parameterized estimation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hpat_model.run_hpat_model import main


if __name__ == "__main__":
    mtl_root = Path(__file__).resolve().parent
    defaults = {
        "--ops_dir": str(mtl_root / "ops"),
        "--config": str(mtl_root / "mtl_config.yaml"),
        "--out": str(ROOT_DIR / "results" / "mtl_estimates.csv"),
        "--out_ops_dir": str(ROOT_DIR / "results" / "mtl_ops"),
    }
    for flag, value in defaults.items():
        if flag not in sys.argv:
            sys.argv.extend([flag, value])
    main()
