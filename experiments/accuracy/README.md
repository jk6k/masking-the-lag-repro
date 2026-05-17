# Accuracy Module Guide

This module handles **accuracy evaluation** by injecting
"quantization + Gaussian noise + crosstalk" into MobileViT Conv / Linear /
Attention paths to simulate photonic device errors.

---

## Entry Script

`eval_cvnets_imagenet_noise.py`

Core functions:
- build MobileViT
- inject noise / crosstalk into Conv / Linear / Attention
- evaluate Top-1 / Top-5 on ImageNet val

---

## Recommended Run Mode (Project Mac / Apple Silicon)

The commands below assume you run from the `experiments` root.

```bash
caffeinate -dimsu python3 accuracy/eval_cvnets_imagenet_noise.py \
  --imagenet_val /path/to/imagenet/val \
  --models mobilevit_xxs,mobilevit_xs,mobilevit_s \
  --quant_bits 8 \
  --gaussian_noise_sigma_lsb 0,0.25,0.5,1.0,2.0 \
  --crosstalk_alpha 0,0.01,0.02,0.05 \
  --enable_attention \
  --device mps \
  --results_csv results/accuracy_noise_mps.csv \
  --run_id 20260204_001 \
  --experiment_id E0 \
  --record_git_hash \
  --metadata_json results/accuracy_noise_mps_meta.json
```

Outputs:

```text
results/accuracy_noise_mps.csv
results/accuracy_noise_mps_meta.json
```

Local accelerator-backed reruns for this project use Apple Silicon `mps`.
CPU fallback is not a governed substitute for an accelerator-backed run. CUDA
is relevant only when the same code is intentionally moved to a separate
CUDA-capable host.

Optional traceability fields (recommended):

- `--run_id` / `--experiment_id` / `--workload`
- `--record_git_hash` or `--git_hash`
- `--config_snapshot` (aligned with the Phase-1 configuration)
- `--metadata_json` (writes run metadata as JSON)

If you use the ImageNet val layout prepared by this repository:

```text
--imagenet_val experiments/datasets/imagenet/val
```

---

## Reproducible CalibSet / EvalSet Split (Recommended)

To avoid tuning on the same validation set that you later report, first create
a **fixed-seed split** of ImageNet val. Use `CalibSet` to choose hyperparameters
such as `tau / k / sigma_ref`, and use `EvalSet` for the final accuracy report.

Generate the split manifests (CSV):

```bash
python3 accuracy/make_imagenet_split_manifest.py \
  --imagenet_val /path/to/imagenet/val \
  --out_dir results/splits \
  --calib_fraction 0.1 \
  --seed 0
```

Then evaluate with the OpenCV pipeline + manifest on only that subset:

```bash
python3 accuracy/eval_cvnets_imagenet_noise.py \
  --imagenet_val /path/to/imagenet/val \
  --opencv_pipeline \
  --imagenet_manifest results/splits/imagenet_val_calib.csv \
  --models mobilevit_xxs \
  --quant_bits 8 \
  --gaussian_noise_sigma_lsb 0,0.25,0.5,1.0,2.0 \
  --crosstalk_alpha 0,0.01,0.02,0.05 \
  --enable_attention \
  --device mps \
  --results_csv results/accuracy_noise_calib.csv
```

Apply the same pattern to `results/splits/imagenet_val_eval.csv` to get the
final report curves.

---

## Key Parameter Notes

- `--quant_bits`: quantization bit width
- `--noise_sigma_lsb` / `--gaussian_noise_sigma_lsb`: Gaussian noise magnitude introduced by photonic-device mapping, in LSB
- `--crosstalk_alpha`: crosstalk strength
- `--enable_attention`: inject noise into attention outputs
- `--device`: `auto` / `cpu` / `cuda` / `mps`
- `--percentage` or `--max_eval_samples`: quick-test controls

---

## Noise Model

The current implementation is equivalent to:

```text
symmetric quantization -> add Gaussian noise (from photonic-device mapping, sigma_lsb) -> add crosstalk (alpha)
```

This is an algorithm-level equivalent simulation of photonic noise / crosstalk
and does not depend on a specific CPU / GPU.

---

## FAQ

- **Slow runtime**: on Mac, prefer `--device mps`; or add `--percentage 1`
- **Abnormal accuracy**: check the ImageNet directory structure and avoid mixing RGB and BGR
- **Multiprocessing stalls**: try `--workers 0`

## Bitstream Measured-Accuracy Contract

If a measured accuracy CSV should satisfy the phase1 bitstream accuracy gate,
annotate the target rows with explicit bitstream semantics first:

```bash
python3 accuracy/annotate_bitstream_accuracy_csv.py \
  --input_csv results/accuracy_noise_eval.csv \
  --output_csv results/accuracy_noise_eval_bitstream.csv \
  --generator low_discrepancy \
  --stream_length 64 \
  --encoding_mode bipolar \
  --multiplier_mode xnor \
  --accumulator_mode bitcount \
  --calibration_source results/bitstream/mobilevit_s_summary.json \
  --contract_note measured_bitstream_dark_launch_candidate
```

Default annotation scope is `baseline=false`, so baseline reference rows are
left untouched unless you provide explicit `--match` filters.

---

## MLX Migration Landing Points

- `photonic_perturb.py` remains the current `torch` mainline implementation.
- `mlx_photonic_perturb.py` provides an MLX perturbation kernel with the same scope, making further Apple Silicon migration easier.
- `../tools/export_mobilevit_weights_npz.py` exports existing `mobilevit_*.pt` checkpoints to `.npz` files as weight inputs for later MLX MobileViT adapters.
- `mlx_mobilevit.py` provides MLX inference definitions and weight loading for MobileViT_xxs/xs/s.
- `eval_mlx_imagenet_noise.py` is the MLX-based entry point for ImageNet noise / crosstalk evaluation.
- `../tools/audit_mlx_mobilevit_parity.py` directly emits stage/block-level MLX vs PyTorch difference JSON so numerical parity can keep converging.
- The current recommendation is still to keep `eval_cvnets_imagenet_noise.py` as the regression-reference baseline for checking MLX / PyTorch numerical consistency.
