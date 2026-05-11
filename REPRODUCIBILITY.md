# Reproducibility Guide

The active anonymous public freeze is `20260511_suds_maxq`.

## Lightweight Verification

Run the public validation gate:

```bash
make repro-check
```

This checks:

- freeze pointer consistency
- SUDS figure numbering and traceability
- phase-summary, slack-manifest, MAX-Q report-data, and AI schematic source inputs
- checksum validation for tracked static artifacts
- absence of datasets, weights, private KB material, archives, trial drafts, real author identity, personal remotes, and absolute local paths
- clean tracked public surface when the directory is a Git checkout

Render the paper figures into ignored build outputs:

```bash
make render-paper-figures
```

Rendered files under `build/rendered_figures/` are for local inspection and CI
smoke only. The tracked repository keeps source JSON/CSV, accepted schematic
source masters, and metadata rather than final figure-pack images.

## Artifact Map

- Freeze pointer: `experiments/results/paper_sync/current_freeze.json`
- Phase summaries and slack manifest: `experiments/results/runs/`
- MAX-Q report data: `experiments/results/report_data/`
- Figure metadata: `figures/paper_figures_20260511_suds_maxq/`
- Figure registry: `figures/paper_figures_20260511_suds_maxq/figure_numbering_registry.csv`
- Traceability: `figures/paper_figures_20260511_suds_maxq/figure_traceability.csv`
- Review metadata: `experiments/results/review/20260511_suds_maxq_public/`
- Checksums: `checksums_manifest.json`

## Claim Boundary

The public freeze supports the SUDS paper's scoped evidence surface: accepted
AI schematics, modeled analytical figures, measured accuracy summaries for
MobileViT-S and minimum GLUE/BERT tasks, calibrated energy-sensitivity tables,
interface-overhead accounting, boundary/counterexample suite,
synthetic-supporting stress checks, parametric PHY checks, and figure
traceability. It does not promote universal scaling, silicon measurement,
SPICE closure, measured hardware energy, broad workload generalization,
deployment readiness, or hardware superiority.

## Full Rerun Boundary

Full reruns require local datasets, local weights, the Mac/MPS development
environment, and long-run handling. On the project Mac, accelerator-backed runs
must use `mps` and long runs must be wrapped with `caffeinate -dimsu`.

Use this package for anonymous reader inspection and source-driven
figure-regeneration. Use the full project workspace only for governed
maintainer MPS reruns and evidence-promotion decisions.
