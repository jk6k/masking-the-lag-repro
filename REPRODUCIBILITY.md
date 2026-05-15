# Reproducibility Guide

The active anonymous public freeze is `20260513_tetc_pivot`.

## Lightweight Verification

Run the public validation gate:

```bash
make repro-check
```

This checks:

- freeze pointer consistency
- SUDS figure numbering and traceability
- phase-summary, slack-manifest, TETC report-data, and source figure inputs
- checksum validation for tracked static artifacts
- absence of datasets, weights, private KB material, archives, trial drafts, real author identity, personal remotes, absolute local paths, and active legacy-route entry points
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
- TETC report data: `experiments/results/report_data/`
- Figure metadata: `figures/suds_tetc_20260513_tetc_pivot/`
- Figure registry: `figures/suds_tetc_20260513_tetc_pivot/figure_numbering_registry.csv`
- Traceability: `figures/suds_tetc_20260513_tetc_pivot/figure_traceability.csv`
- Review metadata: `experiments/results/review/20260513_tetc_pivot_public/`
- Checksums: `checksums_manifest.json`

## Claim Boundary

The public freeze supports the SUDS IEEE TETC scoped evidence surface:
architecture-modeled PPA, governed measured accuracy summaries for MobileViT-S
and GLUE/BERT tasks, calibrated ADC/RTL/PHY parameter inputs, design-space
sweeps, public-repro alignment, local red-team review, and figure traceability.
It also carries the R11 external-abandonment closure record and
major-revision response scaffold. R12 post-R11 reinforcement is included as
compact support and boundary evidence: RTL functional simulation, GLUE task
coverage, MobileViT resolution checks, BERT perturbation-mechanism rows,
DeiT-Tiny and cross-workload boundary records, ADC corner rows, adversarial
review, and the R12 acceptance gate. It does not promote universal scaling,
silicon measurement, SPICE closure, measured hardware energy, broad workload
generalization, universal vision applicability, deployment readiness, or
hardware superiority.

## Full Rerun Boundary

Full reruns require local datasets, local weights, the Mac/MPS development
environment, and long-run handling. On the project Mac, accelerator-backed runs
must use `mps` and long runs must be wrapped with `caffeinate -dimsu`.

Use this package for anonymous reader inspection and source-driven
figure-regeneration. Use the full project workspace only for governed
maintainer MPS reruns and evidence-promotion decisions.
