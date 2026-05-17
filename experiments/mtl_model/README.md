# `mtl_model` Module

This module is the paper-facing Masking-the-Lag parameterized estimation chain. It reuses the stable implementation in `hpat_model/` while standardizing naming around `MTL`.

## Quick Start

```bash
python3 mtl_model/extract_model_ops.py --out_dir mtl_model/ops
python3 mtl_model/run_mtl_model.py --ops_dir mtl_model/ops --config mtl_model/mtl_config.yaml --out results/mtl_estimates.csv
```

For multi-configuration sweeps:

```bash
python3 mtl_model/run_mtl_sweep.py \
  --ops_dir mtl_model/ops \
  --config mtl_model/mtl_config_asic.yaml \
  --out results/mtl_estimates_range.csv \
  --save_per_config
```

## Recommended Outputs

- `results/mtl_estimates.csv`
- `results/mtl_estimates_range.csv`
- `results/mtl_estimates_range_per_config.csv`
- `results/mtl_ops/*_ops.csv`
