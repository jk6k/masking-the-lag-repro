# Full Original Experiment Rerun Guide

This package is designed so an external reader can rerun the original
experiment code after preparing the non-redistributed inputs locally. The
repository intentionally does not include ImageNet, MobileViT checkpoints,
generated MLX weight bundles, private literature mirrors, historical run
payloads, or pre-rendered figures.

The active freeze is `20260430_full_figure_strict_remediated`. Lightweight figure regeneration uses the
checked CSV/JSON evidence. Full reruns regenerate the raw accuracy/modeling
outputs first, then rebuild the quick-report CSVs and paper figures.

## 1. Host And Python Environment

Use Apple Silicon with the MPS backend for local accelerator-backed reruns.
CPU fallback is not a substitute for project-governed full reruns.

```bash
python3 -m venv .venv311-mps
. .venv311-mps/bin/activate
pip install --upgrade pip
pip install -r requirements-full.txt
```

If you use the PyTorch/CVNets reference backend, also make `ml-cvnets`
importable by either installing it or setting:

```bash
export CVNETS_ROOT=/absolute/path/to/ml-cvnets
```

## 2. External Data And Weights

Prepare ImageNet-1k validation images at the synset directory layer:

```text
/data/imagenet/val/
  n01440764/
  n01443537/
  ...
```

Prepare MobileViT checkpoints for all three model keys:

```text
/data/mobilevit_weights/
  mobilevit_xxs.pt
  mobilevit_xs.pt
  mobilevit_s.pt
```

The default model-spec URLs in `experiments/exp_common/model_specs.py` point to
the public Apple MobileViT checkpoints, but for reproducible full reruns you
should keep explicit local copies and pass `WEIGHTS_DIR`.

## 3. Preflight

Run the setup validator before launching long jobs:

```bash
make full-rerun-preflight \
  IMAGENET_VAL=/data/imagenet/val \
  WEIGHTS_DIR=/data/mobilevit_weights
```

For MLX runs, export `.npz` weights once:

```bash
make export-mlx-weights \
  WEIGHTS_DIR=/data/mobilevit_weights \
  MLX_WEIGHTS_DIR=build/mlx_weights \
  WEIGHTS_NPZ_MANIFEST=build/mlx_weights/manifest.json
```

Then rerun the stricter preflight:

```bash
make full-rerun-preflight \
  IMAGENET_VAL=/data/imagenet/val \
  WEIGHTS_NPZ_MANIFEST=build/mlx_weights/manifest.json
```

## 4. Deterministic ImageNet Split Manifests

Generate the calibration/evaluation split manifests used by the accuracy
entry points:

```bash
make imagenet-splits IMAGENET_VAL=/data/imagenet/val
```

This writes:

```text
build/imagenet_splits/imagenet_val_calib.csv
build/imagenet_splits/imagenet_val_eval.csv
build/imagenet_splits/imagenet_val_all.csv
```

The CSV schema is `path,label,class_name,split`; the evaluators consume the
`path,label` columns.

## 5. Accuracy Reruns

The MLX evaluator is the Apple Silicon mainline. A bounded smoke run:

```bash
caffeinate -dimsu .venv311-mps/bin/python experiments/accuracy/eval_mlx_imagenet_noise.py \
  --imagenet_val /data/imagenet/val \
  --imagenet_manifest build/imagenet_splits/imagenet_val_eval.csv \
  --models mobilevit_xxs,mobilevit_xs,mobilevit_s \
  --mlx_weights_dir build/mlx_weights \
  --device mps \
  --opencv_pipeline \
  --max_eval_samples 512 \
  --eval_batch_size 32 \
  --noise_sigma_lsb 0,0.25,0.5,1.0,2.0 \
  --crosstalk_alpha 0,0.01,0.02,0.05 \
  --results_csv build/full_rerun/accuracy_mlx_smoke.csv \
  --metadata_json build/full_rerun/accuracy_mlx_smoke_meta.json
```

For a full evaluation, remove `--max_eval_samples`. Keep `--device mps` and
the `caffeinate -dimsu` wrapper.

The PyTorch/CVNets reference evaluator is available when `ml-cvnets` and
PyTorch weights are prepared:

```bash
caffeinate -dimsu .venv311-mps/bin/python experiments/accuracy/eval_cvnets_imagenet_noise.py \
  --imagenet_val /data/imagenet/val \
  --imagenet_manifest build/imagenet_splits/imagenet_val_eval.csv \
  --opencv_pipeline \
  --models mobilevit_xxs,mobilevit_xs,mobilevit_s \
  --weights_dir /data/mobilevit_weights \
  --device mps \
  --workers 0 \
  --eval_batch_size 32 \
  --gaussian_noise_sigma_lsb 0,0.25,0.5,1.0,2.0 \
  --crosstalk_alpha 0,0.01,0.02,0.05 \
  --enable_attention \
  --results_csv build/full_rerun/accuracy_torch_reference.csv \
  --metadata_json build/full_rerun/accuracy_torch_reference_meta.json
```

## 6. HOPS/FULLER Modeling Reruns

The phase-1 runner is the modeling entry point:

```bash
caffeinate -dimsu .venv311-mps/bin/python experiments/tools/phase1_runner.py \
  --config configs/fuller_det_sparse_reentry_slice_template_20260331.yaml \
  --device mps
```

To materialize the canonical FULLER phase-1 configuration family:

```bash
.venv311-mps/bin/python experiments/tools/prepare_fuller_phase1_bundle.py \
  --bundle configs/fuller_phase1_canonical_bundle_20260421.yaml
```

Then run the generated configs under:

```text
experiments/results/generated_configs/20260421_fuller_phase1_canonical/
```

with `phase1_runner.py --device mps`, using `caffeinate -dimsu` for long
runs.

## 7. Noise And Scaling Families

The final-unreserved noise/scaling support bundles can materialize commands
from the packaged configs. First inspect the commands:

```bash
make full-rerun-noise-dry-run \
  IMAGENET_VAL=/data/imagenet/val \
  WEIGHTS_NPZ_MANIFEST=build/mlx_weights/manifest.json
```

The dry run should print the generated `caffeinate -dimsu ... --device mps`
phase-1 and MLX accuracy commands. It also exercises the packaged rerun
contracts before any long evaluation starts.

To execute the materialized noise family:

```bash
make full-rerun-noise-execute \
  IMAGENET_VAL=/data/imagenet/val \
  WEIGHTS_NPZ_MANIFEST=build/mlx_weights/manifest.json
```

This uses `configs/fuller_final_unreserved_noise_20260427.yaml`, the MLX
accuracy backend, MPS, and the launch wrappers recorded by the generated
manifest.

## 8. Rebuild Paper Evidence After Reruns

After regenerated accuracy/modeling summaries are available, rebuild the
public quick-report layer and figures:

```bash
.venv311-mps/bin/python experiments/tools/build_fuller_final_unreserved_quick_reports.py \
  --run_tag 20260430_full_figure_strict_remediated \
  --quick_dir experiments/results/quick_reports/20260430_full_figure_strict_remediated \
  --review_dir experiments/results/review/20260430_full_figure_strict_remediated

make render-paper-figures
make repro-check
```

`make render-paper-figures` regenerates the data-driven paper figures into
`build/rendered_figures/`. Mechanism schematics (`Fig1` and `Fig9-Fig12`) are
source-only explanatory assets in the thesis; they do not add numeric results
and are governed by the claim-contract metadata.

## 9. Expected Boundaries

- ImageNet and MobileViT checkpoints are external inputs.
- Local accelerator-backed runs must use MPS.
- Long runs should use `caffeinate -dimsu`.
- CPU fallback is a blocker, not an acceptable substitute.
- Regenerated rows must keep the claim boundaries in
  `experiments/results/review/20260430_full_figure_strict_remediated/claim_contract_final_unreserved_20260430.csv`.
