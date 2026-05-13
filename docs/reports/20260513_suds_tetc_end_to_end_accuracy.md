# SUDS TETC End-To-End Perturbation Accuracy

Tag: `20260513_tetc_pivot`
Roadmap item: `R3_end_to_end_perturbation_accuracy`
Evidence label: `measured_mps_end_to_end_perturbation_accuracy`
Acceptance state: `pass`
Stop-condition state: `no R3 hard stop`

## Scope

This artifact joins governed MPS model-forward accuracy evidence to the
TETC architecture PPA rows and R2 SUDS-aware scheduler traces. It checks
that each promoted `suds_pareto` accuracy delta uses the same
KEEP/DEGRADE/PRUNE tier ratios as the architecture row being promoted.

The perturbation surface is a tiered column perturbation proxy: KEEP uses
the unmodified 8-bit-equivalent path, DEGRADE uses a 4-bit-equivalent
small-noise path, and PRUNE uses a stronger 2-bit/removal proxy. Optical
noise is disabled for the promoted accuracy rows and remains a later
sensitivity dimension; this report does not claim bit-exact ADC, optical
device, or silicon behavior.

## Promoted Rows

| Workload | Accuracy delta | Device | Source condition | Trace ID | Policy match |
|---|---:|---|---|---|---|
| `bert_base_glue_seq128` | 0.000 pp | `mps` | `e2_l1` | `r2_7b7427345a30be24` | `pass` |
| `mobilevit_s_transformer_blocks_256` | -0.015 pp | `mps` | `e9_suds_conservative` | `r2_e1d13a0a1855de8b` | `pass` |

## Measured Baseline And Ablation Rows

| Workload | Condition | Delta | EDP ratio | Source rows | Policy match |
|---|---|---:|---:|---:|---|
| `bert_base_glue_seq128` | `lightening_dptc` | 0.000 pp | 1.000 | 18 | `pass` |
| `bert_base_glue_seq128` | `uniform_8bit` | 0.000 pp | 1.128 | 18 | `pass` |
| `bert_base_glue_seq128` | `l1` | 0.000 pp | 0.665 | 18 | `pass` |
| `bert_base_glue_seq128` | `slack_only` | 0.000 pp | 0.666 | 18 | `pass` |
| `bert_base_glue_seq128` | `suds_only` | 0.000 pp | 0.754 | 18 | `pass` |
| `bert_base_glue_seq128` | `suds_l1` | 0.000 pp | 0.754 | 18 | `pass` |
| `bert_base_glue_seq128` | `suds_signal` | 0.000 pp | 0.754 | 18 | `pass` |
| `mobilevit_s_transformer_blocks_256` | `lightening_dptc` | 0.000 pp | 1.000 | 8 | `pass` |
| `mobilevit_s_transformer_blocks_256` | `uniform_8bit` | 0.000 pp | 1.522 | 8 | `pass` |
| `mobilevit_s_transformer_blocks_256` | `random` | -3.599 pp | 0.666 | 8 | `pass` |
| `mobilevit_s_transformer_blocks_256` | `l1` | -3.470 pp | 0.666 | 8 | `pass` |
| `mobilevit_s_transformer_blocks_256` | `slack_only` | -2.656 pp | 0.671 | 8 | `pass` |
| `mobilevit_s_transformer_blocks_256` | `suds_only` | -2.127 pp | 0.746 | 8 | `pass` |
| `mobilevit_s_transformer_blocks_256` | `signal_only` | -1.338 pp | 0.744 | 8 | `pass` |
| `mobilevit_s_transformer_blocks_256` | `suds_l1` | -1.775 pp | 0.746 | 8 | `pass` |
| `mobilevit_s_transformer_blocks_256` | `suds_signal` | -1.522 pp | 0.746 | 8 | `pass` |

## Decision

- Worst promoted accuracy delta: `-0.015 pp`
- Target: `-1.000 pp`
- Promoted policies matched: `True`
- Promoted rows on MPS: `True`
- Promoted rows linked to R2 traces: `True`
- Blockers: `none`

## Artifacts

- CSV: `experiments/results/report_data/suds_tetc_end_to_end_accuracy_20260513_tetc_pivot.csv`
- JSON: `experiments/results/report_data/suds_tetc_end_to_end_accuracy_20260513_tetc_pivot.json`
- Report: `docs/reports/20260513_suds_tetc_end_to_end_accuracy.md`

## Regeneration

```bash
make suds-tetc-end-to-end-accuracy
```
