# Masking the Lag Reproduction

This repository is the reader-facing reproduction package for the promoted paper
evidence freeze `20260430_full_figure_strict_remediated`.

It is intentionally not a mirror of the full development workspace. It contains
only the code, compact CSV/JSON evidence, current figure pack, and validation
scripts needed to inspect and reproduce the paper-facing results.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
make repro-check
```

To rerender the data figures into an ignored build directory:

```bash
make render-paper-figures
```

The checked-in figure pack remains the canonical promoted pack. The render
target writes to `build/` so validation does not mutate the frozen evidence.

## Public Evidence Surface

- Freeze pointer: `experiments/results/paper_sync/current_freeze.json`
- Quick reports: `experiments/results/quick_reports/20260430_full_figure_strict_remediated/`
- Figure pack: `figures/paper_figures_20260430_full_figure_strict_remediated/`
- Review metadata: `experiments/results/review/20260430_full_figure_strict_remediated/`

Datasets, model weights, private literature mirrors, draft candidates, project
management files, and historical run payloads are deliberately excluded.
