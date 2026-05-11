# SUDS Anonymous Reproduction Package

This repository is the double-blind supplementary reproduction package for the
SUDS evidence freeze `20260511_suds_maxq`.

It is intentionally smaller than the full development workspace. It contains
the compact JSON/CSV evidence, accepted schematic source masters, metadata, and
scripts needed to validate the public evidence surface and regenerate the paper
figures. Final pre-rendered figure-pack images are deliberately excluded; local
rendered outputs are produced under `build/`.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
make repro-check
make render-paper-figures
```

`make repro-check` validates the freeze pointer, SUDS figure registry,
traceability, artifact paths, anonymous package boundary, and exclusion of
private materials. `make render-paper-figures` rerenders Fig1-Fig6 and
Fig.A1-Fig.A4 into `build/rendered_figures/`.

## Public Evidence Surface

- Freeze pointer: `experiments/results/paper_sync/current_freeze.json`
- Phase summaries and slack manifest: `experiments/results/runs/`
- MAX-Q report data: `experiments/results/report_data/`
- Figure metadata: `figures/paper_figures_20260511_suds_maxq/`
- Review metadata: `experiments/results/review/20260511_suds_maxq_public/`

Figure roles are fixed for this freeze:

- `Fig1`: SAIG/SUDS interface schematic
- `Fig2`: slack signal availability and independence
- `Fig3`: SUDS ternary quality-budget policy schematic
- `Fig4`: modeled accuracy-energy trade-off
- `Fig5`: tier distribution and ADC-energy waterfall
- `Fig6`: full MobileViT-S measured ablation and modeled ADC ratio
- `Fig.A1-Fig.A4`: threshold scan, SUDS+L1 overlay, synthetic profile stress, and parametric PHY check

## Boundary

The package supports paper inspection, traceability review, and local
figure-regeneration from compact source artifacts. It does not include private
datasets, model weights, private literature mirrors, trial drafts, candidate
histories, historical FULLER freeze payloads, personal remotes, or absolute
local paths.

Full accelerator-backed reruns are outside this anonymous compact package. On
the project Mac, those governed maintainer reruns require Apple Silicon `mps`
and long runs are launched with `caffeinate -dimsu`. The package carries
sanitized compact summaries, not datasets, model weights, or private literature
mirrors.
