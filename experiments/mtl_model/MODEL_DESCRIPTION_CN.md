# MTL Parameterized Model Description

This document defines the paper-facing MTL estimation scope. The implementation currently reuses the stable computational core from `hpat_model/`, while aligning input and output fields with `phase1_runner.py`.

Recommended entry points:

- `mtl_model/extract_model_ops.py`
- `mtl_model/run_mtl_model.py`
- `mtl_model/run_mtl_sweep.py`

Configuration files:

- `mtl_model/mtl_config.yaml`
- `mtl_model/mtl_config_asic.yaml`
