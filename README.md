# Masking the Lag Reproduction

This repository is the reader-facing reproduction package for the promoted
paper evidence freeze `20260430_full_figure_strict_remediated`.

It contains two reproduction layers: compact CSV/JSON evidence for fast audit
and paper-figure regeneration, plus the project experiment code needed to
rerun the HOPS/FULLER modelling and MobileViT accuracy lanes from local
datasets and weights. Pre-rendered image assets, ImageNet data, model
checkpoints, private literature mirrors, draft work products, and historical
heavyweight run payloads are deliberately not tracked.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
make repro-check
make render-paper-figures
```

`make repro-check` validates the freeze pointer, figure registry, claim
boundaries, artifact paths, public repository surface, and presence of the
core experiment code. `make
render-paper-figures` rerenders the Matplotlib data figures into `build/`
from the checked CSV inputs.

## Public Evidence Surface

- Freeze pointer: `experiments/results/paper_sync/current_freeze.json`
- Quick reports: `experiments/results/quick_reports/20260430_full_figure_strict_remediated/`
- Experiment summaries: `experiments/results/report_data/`
- Figure metadata: `figures/paper_figures_20260430_full_figure_strict_remediated/`
- Review metadata: `experiments/results/review/20260430_full_figure_strict_remediated/`
- Core experiment code: `experiments/accuracy/`, `experiments/exp_common/`,
  `experiments/hpat_model/`, `experiments/mtl_model/`,
  `experiments/sc_bitstream/`, and `experiments/tools/`

Figure roles are fixed for this freeze:

- `Fig1`: system stack schematic
- `Fig2`: HOPS timeline
- `Fig3-Fig8`: main-text data figures
- `Fig9-Fig12`: explanation schematics
- `AppF1-AppF6`: appendix figures

## Advanced Full Rerun

The package includes the experiment code and governed configs/results needed to
inspect or rerun the HOPS/FULLER evidence lanes. It does not include ImageNet
or model checkpoints. Full accelerator-backed reruns on the project Mac require
Apple Silicon `mps` and should be launched with `caffeinate -dimsu`; CPU
fallback is not a substitute for an MPS-backed rerun.

Example shape:

```bash
caffeinate -dimsu .venv311-mps/bin/python experiments/tools/phase1_runner.py \
  --config <config.yaml> \
  --device mps
```

Install optional full-run dependencies with:

```bash
pip install -r requirements-full.txt
```

Datasets, model weights, pre-rendered images, private literature mirrors, draft
candidates, project management files, and historical heavyweight run payloads are
deliberately excluded.
