# SUDS Transformer Architecture Simulator

Tag: `20260513_tetc_pivot`
Evidence label: `modeled_system_ppa`
Promotion decision: `architecture_evidence_ready`

## Scope

This artifact maps BERT-base GLUE encoder kernels and MobileViT-S Transformer
blocks onto a Lightening-style DPTC optical tile array, then compares uniform,
selector, SUDS, Lightening-style, and HyAtten-style policies. It reports
architecture-modeled system PPA terms: conversion, DAC/MZM, detector/TIA,
laser, memory movement, optical link, sideband control, and digital fallback.

It is modeled architecture evidence. It does not claim fabrication,
physical-design, device-solver signoff, bench-energy, or deployment evidence.

## Readiness

- Architecture simulator status: `pass`
- Blockers: `none`
- Workloads: `bert_base_glue_seq128,mobilevit_s_transformer_blocks_256`
- Conditions: `astra_boundary,hyatten_style,l1,lightening_dptc,random,signal_only,slack_only,suds_l1,suds_only,suds_signal,tempo_time_multiplexed,uniform_4bit,uniform_8bit`
- Design-space rows: `3888`; Pareto rows: `511`

## Architecture Design Space and Selected Operating Point

Sweep dimensions: `tile_dim={16,32,64}`, `tiles={2,4,8}`,
`cores_per_tile={1,2,4}`, `sideband_group_cols={16,32,64,128}`,
and `adc_sharing={per_array,per_tile,temporal_accum}`.

| Dimension | Selected value | Rationale |
|---|---:|---|
| Tile dimension | 32 | Matches the Lightening/HyAtten DPTC comparison surface and avoids inventing a different array fabric for the main claim. |
| Tiles | 4 | Matches the LT-B-style reference and keeps inter-tile broadcast accounting comparable. |
| Cores per tile | 2 | Matches the LT-B-style reference while preserving a meaningful per-tile accumulation point. |
| Sideband group columns | 32 | Uses the local RTL sideband calibration anchor instead of extrapolating control cost from a different group size. |
| ADC sharing | temporal_accum | Preserves output-stationary DPTC temporal accumulation as the selected conversion-fabric point. |

Selected-point nominal rows:

| Workload | Condition | Energy ratio vs Lightening | EDP ratio vs Lightening | Area mm2 | Memory pJ | Optical-link pJ | Control pJ |
|---|---|---:|---:|---:|---:|---:|---:|
| `bert_base_glue_seq128` | `lightening_dptc` | 1.000 | 1.000 | 3.795 | 3332505.6 | 672399.4 | 2040.4 |
| `bert_base_glue_seq128` | `suds_l1` | 0.785 | 0.754 | 10.388 | 3144746.8 | 537099.1 | 8161.7 |
| `bert_base_glue_seq128` | `suds_signal` | 0.785 | 0.754 | 10.388 | 3144746.8 | 537099.1 | 8161.7 |
| `mobilevit_s_transformer_blocks_256` | `lightening_dptc` | 1.000 | 1.000 | 2.719 | 510889.0 | 350402.6 | 1063.3 |
| `mobilevit_s_transformer_blocks_256` | `suds_l1` | 0.775 | 0.746 | 6.159 | 482952.7 | 281971.9 | 4253.2 |
| `mobilevit_s_transformer_blocks_256` | `suds_signal` | 0.775 | 0.746 | 6.159 | 482952.7 | 281971.9 | 4253.2 |

Boundary rows are retained only as matched architecture context. TeMPO-style
time multiplexing and ASTRA-style stochastic optical rows can define alternate
conversion/readout fabrics, but they are not treated as the selected SUDS DPTC
fabric. Likewise, signal-only/L1/HyAtten wins are boundary evidence for a local
selector beating a scheduler-budgeted composition, not a reason to relabel
SUDS-only as the main method.

## Nominal PPA Summary

| Workload | Condition | Energy ratio vs Lightening | EDP ratio vs Lightening | Latency ns | Energy pJ | Accuracy evidence | Delta |
|---|---|---:|---:|---:|---:|---|---:|
| `bert_base_glue_seq128` | ASTRA-style stochastic optical boundary | 0.311 | 0.408 | 11198.49 | 30268559.6 | `literature_architecture_boundary_unmeasured_locally` | n/a |
| `bert_base_glue_seq128` | HyAtten-style low-resolution signal selector | 0.842 | 0.817 | 8274.59 | 81979445.8 | `literature_baseline_unmeasured_locally` | n/a |
| `bert_base_glue_seq128` | L1 selector | 0.708 | 0.665 | 8015.11 | 68899417.7 | `measured_mps_glue` | 0.00 |
| `bert_base_glue_seq128` | Lightening-style DPTC reference, 8-bit ADC | 1.000 | 1.000 | 8526.22 | 97367145.1 | `measured_mps_glue` | 0.00 |
| `bert_base_glue_seq128` | Random same-sparsity selector | 0.708 | 0.665 | 8015.11 | 68899417.7 | `unmeasured_random_architecture_boundary` | n/a |
| `bert_base_glue_seq128` | Signal/L1 tier selector | 0.785 | 0.754 | 8183.11 | 76456121.0 | `unmeasured_signal_architecture_boundary` | n/a |
| `bert_base_glue_seq128` | Slack-only selector | 0.708 | 0.666 | 8019.26 | 68905539.0 | `measured_mps_glue` | 0.00 |
| `bert_base_glue_seq128` | SUDS budget + L1 selector | 0.785 | 0.754 | 8187.26 | 76462242.3 | `measured_mps_glue` | 0.00 |
| `bert_base_glue_seq128` | SUDS budget only | 0.785 | 0.754 | 8187.26 | 76462242.3 | `measured_mps_glue` | 0.00 |
| `bert_base_glue_seq128` | SUDS budget + signal/overflow selector | 0.785 | 0.754 | 8187.26 | 76462242.3 | `measured_mps_glue` | 0.00 |
| `bert_base_glue_seq128` | TeMPO-style time-multiplexed boundary | 0.699 | 0.817 | 9961.96 | 68067275.9 | `literature_architecture_boundary_unmeasured_locally` | n/a |
| `bert_base_glue_seq128` | Uniform 4-bit DPTC mapping | 0.976 | 0.976 | 8526.18 | 95032425.1 | `unmeasured_accuracy_boundary` | n/a |
| `bert_base_glue_seq128` | Uniform 8-bit DPTC mapping | 1.128 | 1.128 | 8526.38 | 109818985.1 | `measured_mps_glue` | 0.00 |
| `mobilevit_s_transformer_blocks_256` | ASTRA-style stochastic optical boundary | 0.372 | 0.488 | 1568.86 | 4625375.8 | `literature_architecture_boundary_unmeasured_locally` | n/a |
| `mobilevit_s_transformer_blocks_256` | HyAtten-style low-resolution signal selector | 0.800 | 0.778 | 1160.29 | 9957790.3 | `literature_baseline_unmeasured_locally` | n/a |
| `mobilevit_s_transformer_blocks_256` | L1 selector | 0.709 | 0.666 | 1122.33 | 8819769.4 | `measured_mps_imagenet` | -3.47 |
| `mobilevit_s_transformer_blocks_256` | Lightening-style DPTC reference, 8-bit ADC | 1.000 | 1.000 | 1193.95 | 12443405.2 | `measured_mps_imagenet` | 0.00 |
| `mobilevit_s_transformer_blocks_256` | Random same-sparsity selector | 0.709 | 0.666 | 1122.33 | 8819769.4 | `measured_mps_imagenet` | -3.60 |
| `mobilevit_s_transformer_blocks_256` | Signal/L1 tier selector | 0.774 | 0.744 | 1147.18 | 9635114.0 | `measured_mps_imagenet` | -1.34 |
| `mobilevit_s_transformer_blocks_256` | Slack-only selector | 0.712 | 0.671 | 1125.17 | 8857736.6 | `measured_mps_imagenet` | -2.66 |
| `mobilevit_s_transformer_blocks_256` | SUDS budget + L1 selector | 0.775 | 0.746 | 1149.49 | 9645178.6 | `measured_mps_imagenet` | -1.78 |
| `mobilevit_s_transformer_blocks_256` | SUDS budget only | 0.775 | 0.746 | 1149.48 | 9644422.6 | `measured_mps_imagenet` | -2.13 |
| `mobilevit_s_transformer_blocks_256` | SUDS budget + signal/overflow selector | 0.775 | 0.746 | 1149.49 | 9645178.6 | `measured_mps_imagenet` | -1.52 |
| `mobilevit_s_transformer_blocks_256` | TeMPO-style time-multiplexed boundary | 0.720 | 0.842 | 1395.63 | 8963342.0 | `literature_architecture_boundary_unmeasured_locally` | n/a |
| `mobilevit_s_transformer_blocks_256` | Uniform 4-bit DPTC mapping | 0.902 | 0.902 | 1193.92 | 11226729.6 | `unmeasured_accuracy_boundary` | n/a |
| `mobilevit_s_transformer_blocks_256` | Uniform 8-bit DPTC mapping | 1.521 | 1.522 | 1194.12 | 18932341.5 | `measured_mps_imagenet` | 0.00 |

## Pessimistic Gate

| Workload | Best SUDS | Reference baseline | Energy improvement | EDP improvement | Preserved | Boundary stronger conditions |
|---|---|---|---:|---:|---|
| `bert_base_glue_seq128` | `suds_l1` | `lightening_dptc` | 18.58% | 20.99% | `True` | `l1,slack_only,signal_only` |
| `mobilevit_s_transformer_blocks_256` | `suds_l1` | `lightening_dptc` | 21.84% | 23.99% | `True` | `l1,slack_only,signal_only,hyatten_style` |

## Parameter Traceability

| Parameter | Value | Unit | Evidence | Source |
|---|---:|---|---|---|
| `tile_dim` | 32.0000 | DPTC rows/columns | `literature_anchored_assumption` | `kb_markdown/01_transformer_attention_photonic/Lightening_Transformer_HPCA2024.md` |
| `tiles` | 4.0000 | tiles | `literature_anchored_assumption` | `kb_markdown/01_transformer_attention_photonic/Lightening_Transformer_HPCA2024.md` |
| `cores_per_tile` | 2.0000 | DPTC/tile | `literature_anchored_assumption` | `kb_markdown/01_transformer_attention_photonic/Lightening_Transformer_HPCA2024.md` |
| `frequency_ghz` | 5.0000 | GHz | `literature_anchored_assumption` | `kb_markdown/01_transformer_attention_photonic/Lightening_Transformer_HPCA2024.md` |
| `sram_global_kib` | 2048.0000 | KiB | `literature_anchored_assumption` | `kb_markdown/01_transformer_attention_photonic/Lightening_Transformer_HPCA2024.md` |
| `sram_subarray_kib` | 32.0000 | KiB | `literature_anchored_assumption` | `kb_markdown/01_transformer_attention_photonic/Lightening_Transformer_HPCA2024.md` |
| `adc8_pj` | 1.0556 | pJ/conversion | `ngspice_or_fallback_macro_calibration` | `experiments/results/report_data/suds_adc_macro_sanity_20260512_j1_quality_boost.json` |
| `adc6_pj` | 0.2639 | pJ/conversion | `ngspice_or_fallback_macro_calibration` | `experiments/results/report_data/suds_adc_macro_sanity_20260512_j1_quality_boost.json` |
| `adc4_pj` | 0.0660 | pJ/conversion | `ngspice_or_fallback_macro_calibration` | `experiments/results/report_data/suds_adc_macro_sanity_20260512_j1_quality_boost.json` |
| `dac_pj` | 0.6861 | pJ/conversion | `literature_scaled_assumption` | `kb_markdown/01_transformer_attention_photonic/Lightening_Transformer_HPCA2024.md` |
| `hyatten_low_resolution_fraction` | 0.8500 | fraction | `literature_anchored_assumption` | `kb_markdown/01_transformer_attention_photonic/2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md` |
| `control_pj_per_sideband_group` | 0.5904 | pJ/group | `rtl_synthesis_proxy` | `experiments/results/report_data/suds_rtl_control_overhead_20260512_j2_quality_boost.json` |
| `phy_nominal_pass_ratio` | 0.6007 | fraction | `parametric_boundary` | `experiments/results/report_data/suds_phy_circuit_boundary_20260511_p2p3_quality.json` |
| `phy_pessimistic_laser_multiplier` | 1.1500 | x | `parametric_boundary` | `kb_markdown/01_transformer_attention_photonic/ASTRA_Stochastic_Transformer_Silicon_Photonics_TECS2025.md` |
| `lightening_adc_temporal_factor` | 0.1667 | x ADC energy | `literature_anchored_assumption` | `kb_markdown/01_transformer_attention_photonic/Lightening_Transformer_HPCA2024.md` |
| `tempo_time_multiplexing_boundary` | 1.0000 | modeled boundary row enabled | `literature_boundary` | `kb_markdown/01_transformer_attention_photonic/2402.07393_TeMPO_Transformer_Acceleration_with_Co-packaged_Silicon_Photonics.md` |
| `astra_stochastic_boundary` | 1.0000 | modeled boundary row enabled | `literature_boundary` | `kb_markdown/01_transformer_attention_photonic/ASTRA_Stochastic_Transformer_Silicon_Photonics_TECS2025.md` |
| `enlighten_l1_boundary` | 1.0000 | selector boundary enabled | `literature_boundary` | `kb_markdown/01_transformer_attention_photonic/ENLighten_Lighten_the_Transformer_Enable_Efficient_Optical_Acceleration_arXiv2510.01673.md` |
| `selected_sideband_group_cols` | 32.0000 | columns/group | `selected_design_point` | `experiments/results/report_data/suds_rtl_control_overhead_20260512_j2_quality_boost.json` |
| `selected_adc_sharing` | n/a | mode | `selected_design_point` | `kb_markdown/01_transformer_attention_photonic/Lightening_Transformer_HPCA2024.md` |

## Artifacts

- Kernel CSV: `experiments/results/report_data/suds_transformer_architecture_sim_20260513_tetc_pivot_kernels.csv`
- Summary CSV: `experiments/results/report_data/suds_transformer_architecture_sim_20260513_tetc_pivot_summary.csv`
- Parameter CSV: `experiments/results/report_data/suds_transformer_architecture_sim_20260513_tetc_pivot_parameters.csv`
- Sensitivity CSV: `experiments/results/report_data/suds_transformer_architecture_sim_20260513_tetc_pivot_sensitivity.csv`
- GLUE linkage CSV: `experiments/results/report_data/suds_glue_architecture_linkage_20260513_tetc_pivot.csv`
- Design-space CSV: `experiments/results/report_data/suds_transformer_architecture_design_space_20260513_tetc_pivot.csv`
- Design-space JSON: `experiments/results/report_data/suds_transformer_architecture_design_space_20260513_tetc_pivot.json`
- JSON: `experiments/results/report_data/suds_transformer_architecture_sim_20260513_tetc_pivot.json`

## Regeneration

```bash
.venv311-mps/bin/python experiments/tools/build_suds_transformer_architecture_sim.py --tag 20260513_tetc_pivot
```
