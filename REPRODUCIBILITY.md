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

## Artifact Map

- Freeze pointer: `experiments/results/paper_sync/current_freeze.json`
- Quick reports: `experiments/results/quick_reports/20260430_full_figure_strict_remediated/`
- Figure metadata: `figures/paper_figures_20260430_full_figure_strict_remediated/`
- Review pack: `experiments/results/review/20260430_full_figure_strict_remediated/`
- Figure registry: `figures/paper_figures_20260430_full_figure_strict_remediated/figure_numbering_registry.csv`
- Traceability: `figures/paper_figures_20260430_full_figure_strict_remediated/figure_traceability.csv`
- Claim contract: `experiments/results/review/20260430_full_figure_strict_remediated/claim_contract_final_unreserved_20260430.csv`
- Gate E blocker bundle: `experiments/results/review/20260430_full_figure_strict_remediated/gate_e_blockers/`

## Claim Boundary

The public freeze supports runtime/materialization inspection, bounded
sensitivity and scaling support, device-context inspection, source-driven
data-figure regeneration, and traceability review. It does not promote accuracy preservation, broad
robustness, universal scaling, silicon measurement, fabricated-device evidence, device
superiority, benchmark equivalence, field-use readiness, or broad workload
generalization.

## Advanced Full Rerun Boundary

Full reruns require local datasets, local weights, the Mac/MPS development
environment, and long-run handling. On the project Mac, accelerator-backed runs
must use `mps` and long runs must be wrapped with `caffeinate -dimsu`.

Use this public package for reader inspection and source-driven data-figure
regeneration. Use the full project workspace for governed MPS reruns and
promotion decisions.

The Gate E bundle is intentionally included as public blocker evidence. It
records why the current Energy/PHY material remains context-only until
component-source and per-layer link-budget closure are supplied.
