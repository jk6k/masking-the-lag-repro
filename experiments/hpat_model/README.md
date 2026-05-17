# `hpat_model` Module

This module provides an end-to-end HPAT-style parameterized estimator. It is a modeling tool, not a hardware measurement pipeline, and it produces paper-facing latency and energy estimates for photonic accelerators.

## Entry Scripts

1. operator extraction: `extract_model_ops.py`
2. estimation: `run_hpat_model.py` and `run_hpat_sweep.py`

## Basic Flow

Run from the `experiments/` root:

```bash
python3 hpat_model/extract_model_ops.py --out_dir hpat_model/ops
python3 hpat_model/run_hpat_model.py --ops_dir hpat_model/ops --out results/hpat_estimates.csv
```

Outputs:

- `results/hpat_estimates.csv`
- `results/hpat_ops/*_ops.csv`

The summary CSV includes both mJ-scale breakdown fields and paper-facing Joule fields.

## ASIC Single-Configuration Sweep

```bash
python3 hpat_model/run_hpat_sweep.py \
  --ops_dir hpat_model/ops \
  --config hpat_model/hpat_config_asic.yaml \
  --out results/hpat_estimates_range.csv \
  --save_per_config
```

Outputs:

- `results/hpat_estimates_range.csv`
- `results/hpat_estimates_range_per_config.csv`

## Current Coverage

Included:

- GEMM for Conv and Linear layers
- attention `QK` and `AV`
- softmax, normalization, and activation estimated in the electronic domain

Not yet included:

- on-chip and off-chip communication costs
- additional controller and digital-logic overhead

See `MODEL_DESCRIPTION_CN.md` for the detailed model description.
