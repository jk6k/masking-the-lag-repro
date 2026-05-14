# SUDS Anonymous Reproduction Package

This repository is the double-blind supplementary reproduction package for the
SUDS IEEE TETC evidence freeze `20260513_tetc_pivot`.

It is intentionally smaller than the full development workspace. It contains
the compact JSON/CSV evidence, accepted schematic source masters, metadata,
checksums, and scripts needed to validate the public evidence surface and
regenerate the paper figures. Final pre-rendered figure-pack images are
deliberately excluded; local rendered outputs are produced under `build/`.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
make repro-check
make render-paper-figures
```

`make repro-check` validates the freeze pointer, SUDS figure registry,
traceability, artifact paths, SHA-256 checksum manifest, anonymous package
boundary, and exclusion of private materials. `make render-paper-figures`
rerenders Fig1-Fig4 into `build/rendered_figures/`.

## Public Evidence Surface

- Freeze pointer: `experiments/results/paper_sync/current_freeze.json`
- Phase summaries and slack manifest: `experiments/results/runs/`
- TETC report data: `experiments/results/report_data/`
- Figure metadata: `figures/suds_tetc_20260513_tetc_pivot/`
- Review metadata: `experiments/results/review/20260513_tetc_pivot_public/`
- Static-file checksums: `checksums_manifest.json`

The public package keeps compact source artifacts and render scripts rather
than final submitted images. Figure roles for the architecture-first TETC route
are fixed in the TETC figure traceability metadata.

- `Fig1`: TETC architecture and evidence flow
- `Fig2`: accuracy/EDP Pareto surface
- `Fig3`: modeled energy-component breakdown
- `Fig4`: conservative measured accuracy boundary

## Boundary

The package supports paper inspection, traceability review, and local
figure-regeneration from compact source artifacts. It includes architecture
simulation, calibration, public-repro alignment, and local red-team compact
data, R11 external-review accepted-risk records, and the major-revision
response scaffold. It does not include private datasets, model weights,
private literature mirrors, trial drafts, candidate histories, historical
FULLER freeze payloads, personal remotes, or absolute local paths.

Full accelerator-backed reruns are outside this anonymous compact package. On
the project Mac, those governed maintainer reruns require Apple Silicon `mps`
and long runs are launched with `caffeinate -dimsu`. The package carries
sanitized compact summaries, not datasets, model weights, or private literature
mirrors.
