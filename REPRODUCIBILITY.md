# Reproducibility Guide

The active public freeze is `20260430_full_figure_strict_remediated`.

## Lightweight Verification

Run the public validation gate:

```bash
make repro-check
```

This checks:

- freeze pointer consistency
- figure numbering and traceability
- data-figure source evidence and claim boundaries
- presence of the core experiment code surface
- absence of private datasets, weights, draft artifacts, and old freeze tokens
- absence of tracked pre-rendered image artifacts
- clean tracked public surface when the directory is a Git checkout

Render the data figures into ignored build outputs:

```bash
make render-paper-figures
```

Rendered files under `build/` are for local inspection and CI smoke only. The
tracked repository keeps source CSV/JSON and metadata rather than pre-rendered
PNG/PDF/SVG assets.

For full original experiment reruns with external ImageNet data and MobileViT
weights, follow `FULL_RERUN.md`.

## Artifact Map

- Freeze pointer: `experiments/results/paper_sync/current_freeze.json`
- Quick reports: `experiments/results/quick_reports/20260430_full_figure_strict_remediated/`
- Experiment summaries: `experiments/results/report_data/`
- Figure metadata: `figures/paper_figures_20260430_full_figure_strict_remediated/`
- Review pack: `experiments/results/review/20260430_full_figure_strict_remediated/`
- Figure registry: `figures/paper_figures_20260430_full_figure_strict_remediated/figure_numbering_registry.csv`
- Traceability: `figures/paper_figures_20260430_full_figure_strict_remediated/figure_traceability.csv`
- Claim contract: `experiments/results/review/20260430_full_figure_strict_remediated/claim_contract_final_unreserved_20260430.csv`
- HOPS/FULLER runner: `experiments/tools/phase1_runner.py`
- Accuracy rerun entry points: `experiments/accuracy/eval_mlx_imagenet_noise.py`,
  `experiments/accuracy/eval_mlx_imagenet_bitstream_slice.py`, and
  `experiments/accuracy/eval_cvnets_imagenet_noise.py`
- Full-rerun setup validator: `scripts/validate_full_rerun_setup.py`
- ImageNet split manifest helper: `experiments/accuracy/make_imagenet_split_manifest.py`
- Modelling modules: `experiments/exp_common/`, `experiments/hpat_model/`,
  `experiments/mtl_model/`, and `experiments/sc_bitstream/`

## Claim Boundary

The public freeze supports runtime/materialization inspection, bounded
sensitivity and scaling support, device-context inspection, source-driven
data-figure regeneration, and traceability review. It does not promote accuracy preservation, broad
robustness, universal scaling, silicon measurement, hardware validation, device
superiority, benchmark equivalence, deployment readiness, or broad workload
generalization.

## Advanced Full Rerun Boundary

Full reruns require local datasets, local weights, the Mac/MPS development
environment, optional dependencies from `requirements-full.txt`, and long-run
handling. On the project Mac, accelerator-backed runs must use `mps` and long
runs must be wrapped with `caffeinate -dimsu`.

Use this public package for reader inspection, source-driven data-figure
regeneration, and full-rerun preparation/execution after external data and
weights are supplied. Use local ImageNet/weights on a governed Apple Silicon
host for full MPS reruns and promotion decisions.
