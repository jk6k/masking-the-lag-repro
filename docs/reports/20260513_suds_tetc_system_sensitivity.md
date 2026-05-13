# SUDS TETC System Sensitivity

Tag: `20260513_tetc_pivot`
Roadmap item: `R6_memory_bandwidth_conversion_link_sensitivity`
Evidence label: `system_memory_conversion_link_sensitivity`
Acceptance state: `pass`
Stop-condition state: `no R6 hard stop; nominal and pessimistic named regimes preserve the bounded promoted EDP claim`

## Scope

This artifact holds the promoted `suds_pareto` point fixed and sweeps
memory bandwidth, activation reuse, batch size, sequence length, ADC
energy, DAC/MZM energy, laser multiplier, optical-link loss, and
sideband-control overhead against the same-scope Lightening-style DPTC
reference. The rows are parametric architecture sensitivity rows, not
new hardware measurements or model-accuracy runs.

## Decision

- R6 acceptance: `pass`
- Blockers: `none`
- Nominal and pessimistic named regimes preserve benefit: `True`
- Claim narrowing required for extreme sweeps: `True`
- Minimum pessimistic EDP improvement: `12.10%`

## Named Regimes

| Regime | Workload | Energy improvement | EDP improvement | EDP ratio | Class |
|---|---|---:|---:|---:|---|
| `optimistic` | `bert_base_glue_seq128` | 29.55% | 33.74% | 0.663 | `beneficial` |
| `optimistic` | `mobilevit_s_transformer_blocks_256` | 7.63% | 7.46% | 0.925 | `beneficial` |
| `nominal` | `bert_base_glue_seq128` | 29.23% | 33.44% | 0.666 | `beneficial` |
| `nominal` | `mobilevit_s_transformer_blocks_256` | 8.27% | 8.10% | 0.919 | `beneficial` |
| `pessimistic` | `bert_base_glue_seq128` | 28.99% | 33.21% | 0.668 | `beneficial` |
| `pessimistic` | `mobilevit_s_transformer_blocks_256` | 12.26% | 12.10% | 0.879 | `beneficial` |

## Sweep Boundaries

| Axis | Workload | Last beneficial value | First not-beneficial value | Min EDP improvement | Max EDP ratio |
|---|---|---:|---:|---:|---:|
| `activation_reuse_scale` | `bert_base_glue_seq128` | 2.000 | n/a | 32.27% | 0.677 |
| `activation_reuse_scale` | `mobilevit_s_transformer_blocks_256` | 2.000 | n/a | 7.52% | 0.925 |
| `adc_energy_scale` | `bert_base_glue_seq128` | 8.000 | n/a | 33.43% | 0.666 |
| `adc_energy_scale` | `mobilevit_s_transformer_blocks_256` | 8.000 | n/a | 4.18% | 0.958 |
| `batch_size_scale` | `bert_base_glue_seq128` | 16.000 | n/a | 33.44% | 0.666 |
| `batch_size_scale` | `mobilevit_s_transformer_blocks_256` | 16.000 | n/a | 8.10% | 0.919 |
| `dac_energy_scale` | `bert_base_glue_seq128` | 128.000 | n/a | 32.85% | 0.672 |
| `dac_energy_scale` | `mobilevit_s_transformer_blocks_256` | 32.000 | 64.000 | -0.09% | 1.001 |
| `laser_multiplier` | `bert_base_glue_seq128` | 4.000 | n/a | 33.44% | 0.666 |
| `laser_multiplier` | `mobilevit_s_transformer_blocks_256` | 4.000 | n/a | 8.09% | 0.919 |
| `memory_bandwidth_scale` | `bert_base_glue_seq128` | 4.000 | n/a | 33.24% | 0.668 |
| `memory_bandwidth_scale` | `mobilevit_s_transformer_blocks_256` | 4.000 | n/a | 8.00% | 0.920 |
| `optical_link_loss_scale` | `bert_base_glue_seq128` | 4.000 | n/a | 33.43% | 0.666 |
| `optical_link_loss_scale` | `mobilevit_s_transformer_blocks_256` | 4.000 | n/a | 6.13% | 0.939 |
| `sequence_length_scale` | `bert_base_glue_seq128` | 4.000 | n/a | 33.44% | 0.666 |
| `sequence_length_scale` | `mobilevit_s_transformer_blocks_256` | 4.000 | n/a | 8.10% | 0.919 |
| `sideband_control_overhead_scale` | `bert_base_glue_seq128` | 100.000 | n/a | 32.80% | 0.672 |
| `sideband_control_overhead_scale` | `mobilevit_s_transformer_blocks_256` | 100.000 | n/a | 5.51% | 0.945 |

## Boundary Examples

Rows below are one-at-a-time extreme sweeps where the promoted point is
not beneficial. They bound the claim and are not used as promoted
realistic regimes.

| Workload | Axis | Value | EDP ratio | EDP improvement |
|---|---|---:|---:|---:|
| `mobilevit_s_transformer_blocks_256` | `dac_energy_scale` | 128.000 | 1.001 | -0.09% |
| `mobilevit_s_transformer_blocks_256` | `dac_energy_scale` | 64.000 | 1.000 | -0.01% |

## Artifacts

- System sensitivity CSV: `experiments/results/report_data/suds_tetc_system_sensitivity_20260513_tetc_pivot.csv`
- System sensitivity JSON: `experiments/results/report_data/suds_tetc_system_sensitivity_20260513_tetc_pivot.json`
- Report: `docs/reports/20260513_suds_tetc_system_sensitivity.md`
- Architecture summary input: `experiments/results/report_data/suds_transformer_architecture_sim_20260513_tetc_pivot_summary.csv`
- R3 accuracy input: `experiments/results/report_data/suds_tetc_end_to_end_accuracy_20260513_tetc_pivot.json`
- R5 Pareto input: `experiments/results/report_data/suds_tetc_pareto_design_space_20260513_tetc_pivot.json`

## Regeneration

```bash
make suds-tetc-system-sensitivity
```
