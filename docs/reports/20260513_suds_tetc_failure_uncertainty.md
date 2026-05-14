# SUDS TETC Failure Cases And Uncertainty

Tag: `20260513_tetc_pivot`
Roadmap item: `R10_failure_cases_and_uncertainty`
Evidence label: `failure_cases_and_uncertainty`
Acceptance state: `pass`
Stop-condition state: `no R10 hard stop; target-regime uncertainty preserves positive EDP advantage`

## Scope

This artifact records counterexamples and bounded uncertainty for the
promoted `suds_pareto` point. It reuses R3 measured MPS accuracy, R5
Pareto linkage, R6 system-sensitivity logic, and R9 workload-boundary
rows. It is not a new hardware measurement or a new model evaluation.

## Decision

- R10 acceptance: `pass`
- Blockers: `none`
- Target-regime uncertainty crosses zero advantage: `False`
- Boundary-regime claim narrowing required: `True`
- Required failure families present: `True`
- Accuracy seed bootstrap within budget: `True`

## Target-Regime Monte Carlo

| Workload | Mean EDP improvement | 95% CI | P(positive) | P(material >=5%) |
|---|---:|---:|---:|---:|
| `bert_base_glue_seq128` | 33.42% | [32.70, 34.10]% | 1.000 | 1.000 |
| `mobilevit_s_transformer_blocks_256` | 10.36% | [5.84, 15.13]% | 1.000 | 0.998 |

## Boundary-Regime Monte Carlo

- Any boundary workload crosses zero advantage: `True`
- Minimum boundary 95% lower EDP-improvement bound: `-0.27%`

Boundary draws intentionally include extreme conversion, link, sideband,
and memory-pressure settings. They are used to narrow the claim, not to
replace the target regime.

## Accuracy Bootstrap

| Workload | Seed values | Mean delta | 95% CI | Worst observed | Within 1 pp budget |
|---|---:|---:|---:|---:|---:|
| `bert_base_glue_seq128` | 3 | 0.000 pp | [0.000, 0.000] pp | 0.000 pp | `True` |
| `mobilevit_s_transformer_blocks_256` | 8 | -0.015 pp | [-0.049, 0.025] pp | -0.086 pp | `True` |

## Failure Families

| Family | Rows | Should-not-use rows | Min EDP improvement | Min accuracy delta |
|---|---:|---:|---:|---:|
| `activation_sensitive_layers` | 3 | 3 | 25.38% | -3.470 pp |
| `conversion_dominated_boundary` | 18 | 18 | -0.12% | n/a pp |
| `high_memory_pressure` | 2 | 1 | 0.54% | -0.015 pp |
| `long_sequence` | 1 | 1 | 33.30% | n/a pp |
| `low_slack` | 2 | 2 | -0.07% | 0.000 pp |
| `signal_only_dominant` | 2 | 2 | 18.79% | -1.522 pp |
| `small_batch` | 1 | 1 | 6.90% | n/a pp |

## Counterexample Rows

| Case | Family | Workload | Condition | Should not use | Failure mode |
|---|---|---|---|---:|---|
| `r10_activation_sensitive_l1_mobilevit_s_transformer_blocks_256` | `activation_sensitive_layers` | `mobilevit_s_transformer_blocks_256` | `l1` | `True` | activation-sensitive MobileViT-S rows lose more than 1 pp top-1 |
| `r10_activation_sensitive_slack_only_mobilevit_s_transformer_blocks_256` | `activation_sensitive_layers` | `mobilevit_s_transformer_blocks_256` | `slack_only` | `True` | activation-sensitive MobileViT-S rows lose more than 1 pp top-1 |
| `r10_activation_sensitive_suds_only_mobilevit_s_transformer_blocks_256` | `activation_sensitive_layers` | `mobilevit_s_transformer_blocks_256` | `suds_only` | `True` | activation-sensitive MobileViT-S rows lose more than 1 pp top-1 |
| `r10_conversion_boundary_r6_combined_dac_energy_scale_x_optical_link_loss_scale_128_1_mobilevit_s_transformer_blocks_256` | `conversion_dominated_boundary` | `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `True` | DAC/MZM or combined conversion stress crosses the zero-advantage boundary |
| `r10_conversion_boundary_r6_combined_dac_energy_scale_x_optical_link_loss_scale_128_1p5_mobilevit_s_transformer_blocks_256` | `conversion_dominated_boundary` | `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `True` | DAC/MZM or combined conversion stress crosses the zero-advantage boundary |
| `r10_conversion_boundary_r6_combined_dac_energy_scale_x_optical_link_loss_scale_128_2_mobilevit_s_transformer_blocks_256` | `conversion_dominated_boundary` | `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `True` | DAC/MZM or combined conversion stress crosses the zero-advantage boundary |
| `r10_conversion_boundary_r6_combined_dac_energy_scale_x_optical_link_loss_scale_128_4_mobilevit_s_transformer_blocks_256` | `conversion_dominated_boundary` | `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `True` | DAC/MZM or combined conversion stress crosses the zero-advantage boundary |
| `r10_conversion_boundary_r6_combined_dac_energy_scale_x_optical_link_loss_scale_64_1_mobilevit_s_transformer_blocks_256` | `conversion_dominated_boundary` | `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `True` | DAC/MZM or combined conversion stress crosses the zero-advantage boundary |
| `r10_conversion_boundary_r6_combined_dac_energy_scale_x_optical_link_loss_scale_64_1p5_mobilevit_s_transformer_blocks_256` | `conversion_dominated_boundary` | `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `True` | DAC/MZM or combined conversion stress crosses the zero-advantage boundary |
| `r10_conversion_boundary_r6_combined_dac_energy_scale_x_optical_link_loss_scale_64_2_mobilevit_s_transformer_blocks_256` | `conversion_dominated_boundary` | `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `True` | DAC/MZM or combined conversion stress crosses the zero-advantage boundary |
| `r10_conversion_boundary_r6_combined_dac_energy_scale_x_optical_link_loss_scale_64_4_mobilevit_s_transformer_blocks_256` | `conversion_dominated_boundary` | `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `True` | DAC/MZM or combined conversion stress crosses the zero-advantage boundary |
| `r10_conversion_boundary_r6_combined_dac_energy_scale_x_sideband_control_overhead_scale_128_100_mobilevit_s_transformer_blocks_256` | `conversion_dominated_boundary` | `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `True` | DAC/MZM or combined conversion stress crosses the zero-advantage boundary |
| `r10_conversion_boundary_r6_combined_dac_energy_scale_x_sideband_control_overhead_scale_128_10_mobilevit_s_transformer_blocks_256` | `conversion_dominated_boundary` | `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `True` | DAC/MZM or combined conversion stress crosses the zero-advantage boundary |
| `r10_conversion_boundary_r6_combined_dac_energy_scale_x_sideband_control_overhead_scale_128_1_mobilevit_s_transformer_blocks_256` | `conversion_dominated_boundary` | `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `True` | DAC/MZM or combined conversion stress crosses the zero-advantage boundary |
| `r10_conversion_boundary_r6_combined_dac_energy_scale_x_sideband_control_overhead_scale_128_25_mobilevit_s_transformer_blocks_256` | `conversion_dominated_boundary` | `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `True` | DAC/MZM or combined conversion stress crosses the zero-advantage boundary |
| `r10_conversion_boundary_r6_combined_dac_energy_scale_x_sideband_control_overhead_scale_64_100_mobilevit_s_transformer_blocks_256` | `conversion_dominated_boundary` | `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `True` | DAC/MZM or combined conversion stress crosses the zero-advantage boundary |
| `r10_conversion_boundary_r6_combined_dac_energy_scale_x_sideband_control_overhead_scale_64_10_mobilevit_s_transformer_blocks_256` | `conversion_dominated_boundary` | `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `True` | DAC/MZM or combined conversion stress crosses the zero-advantage boundary |
| `r10_conversion_boundary_r6_combined_dac_energy_scale_x_sideband_control_overhead_scale_64_1_mobilevit_s_transformer_blocks_256` | `conversion_dominated_boundary` | `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `True` | DAC/MZM or combined conversion stress crosses the zero-advantage boundary |
| `r10_conversion_boundary_r6_combined_dac_energy_scale_x_sideband_control_overhead_scale_64_25_mobilevit_s_transformer_blocks_256` | `conversion_dominated_boundary` | `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `True` | DAC/MZM or combined conversion stress crosses the zero-advantage boundary |
| `r10_conversion_boundary_r6_sweep_dac_energy_scale_128_mobilevit_s_transformer_blocks_256` | `conversion_dominated_boundary` | `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `True` | DAC/MZM or combined conversion stress crosses the zero-advantage boundary |
| `r10_conversion_boundary_r6_sweep_dac_energy_scale_64_mobilevit_s_transformer_blocks_256` | `conversion_dominated_boundary` | `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `True` | DAC/MZM or combined conversion stress crosses the zero-advantage boundary |
| `r10_high_memory_pressure_bert_base_glue_seq128` | `high_memory_pressure` | `bert_base_glue_seq128` | `suds_pareto` | `False` | memory movement dominates the conversion savings |
| `r10_high_memory_pressure_mobilevit_s_transformer_blocks_256` | `high_memory_pressure` | `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `True` | memory movement dominates the conversion savings |
| `r10_long_sequence_bert512_accuracy_boundary` | `long_sequence` | `bert_base_seq512_batch1_r9` | `suds_pareto` | `True` | no governed measured accuracy row for this exact sequence-length/batch setting |
| `r10_low_slack_forced_keep_bert_base_glue_seq128` | `low_slack` | `bert_base_glue_seq128` | `suds_forced_keep_all` | `True` | low slack leaves no safe budget to degrade or prune columns |
| `r10_low_slack_forced_keep_mobilevit_s_transformer_blocks_256` | `low_slack` | `mobilevit_s_transformer_blocks_256` | `suds_forced_keep_all` | `True` | low slack leaves no safe budget to degrade or prune columns |
| `r10_signal_only_dominant_signal_only_mobilevit_s_transformer_blocks_256` | `signal_only_dominant` | `mobilevit_s_transformer_blocks_256` | `signal_only` | `True` | signal-only selector dominates raw PPA but not the promoted accuracy/fairness contract |
| `r10_signal_only_dominant_suds_signal_mobilevit_s_transformer_blocks_256` | `signal_only_dominant` | `mobilevit_s_transformer_blocks_256` | `suds_signal` | `True` | signal-only selector dominates raw PPA but not the promoted accuracy/fairness contract |
| `r10_small_batch_deit_tiny_batch1_accuracy_blocker` | `small_batch` | `deit_tiny_patch16_224_batch1_r9` | `suds_pareto` | `True` | no local governed DeiT-Tiny weights/dataset accuracy run found; R9 emits simulator-only traces instead of hiding the setup blocker |

## Artifacts

- Failure-suite CSV: `experiments/results/report_data/suds_tetc_failure_suite_20260513_tetc_pivot.csv`
- Uncertainty JSON: `experiments/results/report_data/suds_tetc_uncertainty_20260513_tetc_pivot.json`
- Report: `docs/reports/20260513_suds_tetc_failure_uncertainty.md`

## Regeneration

```bash
make suds-tetc-failure-uncertainty
```
