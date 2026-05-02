# Masking the Lag Reproduction

This repository is the reader-facing reproduction package for the promoted
paper evidence freeze `20260430_full_figure_strict_remediated`.

Project positioning is fixed as an accelerator-design study with evidence-gated
operating-point promotion. The public package lets readers inspect why the
current HOPS/FULLER operating points are not promoted; it should not be read as
a standalone negative-result audit detached from the accelerator design.

It is intentionally smaller than the full development workspace. It contains
the compact CSV/JSON evidence, metadata, and scripts needed to validate the
evidence surface and regenerate the public data figures. Pre-rendered image
assets are deliberately not tracked.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
make repro-check
make render-paper-figures
```

`make repro-check` validates the freeze pointer, figure registry, claim
boundaries, artifact paths, and public repository surface. `make
render-paper-figures` rerenders the Matplotlib data figures into `build/`
from the checked CSV inputs.

## Public Evidence Surface

- Freeze pointer: `experiments/results/paper_sync/current_freeze.json`
- Quick reports: `experiments/results/quick_reports/20260430_full_figure_strict_remediated/`
- Figure metadata: `figures/paper_figures_20260430_full_figure_strict_remediated/`
- Review metadata: `experiments/results/review/20260430_full_figure_strict_remediated/`

Figure roles are fixed for this freeze:

- `Fig1`: system stack schematic
- `Fig2`: HOPS timeline
- `Fig3-Fig8`: main-text data figures
- `Fig9-Fig12`: explanation schematics
- `AppF1-AppF6`: appendix figures

## Advanced Full Rerun

The lightweight public package does not include ImageNet, model checkpoints, or
the full local experiment workspace. Full accelerator-backed reruns on the
project Mac require Apple Silicon `mps` and should be launched with
`caffeinate -dimsu`; CPU fallback is not a substitute for an MPS-backed rerun.

Example shape:

```bash
caffeinate -dimsu .venv311-mps/bin/python experiments/tools/phase1_runner.py \
  --config <config.yaml> \
  --device mps
```

The public repository is therefore a two-layer artifact: fast inspection and
data-figure regeneration by default, with full local reruns documented as a
governed maintainer workflow outside this compact package.

Datasets, model weights, pre-rendered images, private literature mirrors, draft
candidates, project management files, and historical run payloads are
deliberately excluded.
