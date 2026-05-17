# SUDS TETC Baseline Parameter Crosswalk

Date: `2026-05-17`
Roadmap item: `R13-2_baseline_parameter_crosswalk`
Evidence label: `baseline_parameter_crosswalk`
Acceptance state: `pass`

## Scope

This report pre-answers baseline-fairness questions for Lightening,
HyAtten, TeMPO, and ASTRA using only the local KB Markdown mirrors and
the existing R4 same-simulator artifacts. It is a parameter and claim
permission crosswalk, not a new model-evaluation or device-validation run.

## Decision

- R13-2 acceptance: `pass`
- Blockers: `none`
- R4 input acceptance: `pass`
- Verified source anchors: `11`
- Lightening same-scope reference: `True`
- Alternate-fabric boundaries visible: `True`
- Not-dominance candidates visible: `True`

## Reviewer Crosswalk

| Baseline | Scope | Key assumption | SUDS mapping | Permission |
| --- | --- | --- | --- | --- |
| Lightening | same_scope_baseline | Lightening uses a deterministic, coherent DPTC fabric for dynamic full-range Transformer matrix multiplication, with temporal accumulation before O-E conversion. | SUDS adopts this Lightening-style deterministic DPTC as the same-fabric reference row and applies scheduler-derived KEEP/DEGRADE/PRUNE policy decisions on that fabric. | main comparison |
| HyAtten | boundary_baseline, not_dominance_candidate | HyAtten classifies attention signals by converter range: about 85% of analog signals use low-resolution 4-bit conversion, while higher-resolution residuals are handled by digital circuits. | R4 keeps a HyAtten-style low-resolution signal-selection row visible with degrade_ratio=0.85 and keep_ratio=0.15, using the same calibrated ADC/DAC tier table but not the selected SUDS policy semantics. | boundary context, no claim |
| TeMPO | boundary_baseline, not_dominance_candidate | TeMPO uses a time-multiplexed dynamic photonic tensor accelerator with customized slow-light MZM devices and hierarchical photocurrent and temporal integration. | R4 includes a TeMPO-style time-multiplexed readout boundary row inside the same simulator accounting surface, but the local implementation remains the selected Lightening-style DPTC fabric. | boundary context, no claim |
| ASTRA | boundary_baseline, not_dominance_candidate | ASTRA replaces deterministic DPTC operation with stochastic signed optical multiplication, compute-capable transducers, DAC-removal assumptions, and temporal analog accumulation. | R4 includes an ASTRA-style stochastic boundary row to expose alternate-fabric efficiency, while SUDS remains a deterministic DPTC scheduler/control proposal. | boundary context, no claim |

## Full Crosswalk Rows

| baseline | source_path | source_assumption | suds_mapping | scope_label | differing_assumption | claim_permission | r4_accuracy_evidence | r4_edp_ratio_vs_lightening_range | r4_dominance_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lightening | kb_markdown/01_transformer_attention_photonic/Lightening_Transformer_HPCA2024.md | Lightening uses a deterministic, coherent DPTC fabric for dynamic full-range Transformer matrix multiplication, with temporal accumulation before O-E conversion. | SUDS adopts this Lightening-style deterministic DPTC as the same-fabric reference row and applies scheduler-derived KEEP/DEGRADE/PRUNE policy decisions on that fabric. | same_scope_baseline | none; this is the same-scope DPTC reference | main comparison | measured_mps_glue,measured_mps_imagenet | 1.000-1.000 | not_dominating_promoted_suds |
| Lightening | experiments/results/report_data/suds_transformer_architecture_sim_20260513_tetc_pivot_parameters.csv | The simulator parameter table fixes the selected DPTC operating point: tile_dim=32, tiles=4, cores_per_tile=2, frequency_ghz=5.0, sram_global_kib=2048, sram_subarray_kib=32, adc_temporal_factor=0.166667, adc_sharing=temporal_accum. | R4 same-simulator rows use this common tile, frequency, memory, ADC/DAC, and workload-shape surface before dominance is checked. | same_scope_baseline | none; shared simulator configuration | main comparison | measured_mps_glue,measured_mps_imagenet | 1.000-1.000 | not_dominating_promoted_suds |
| HyAtten | kb_markdown/01_transformer_attention_photonic/2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md | HyAtten classifies attention signals by converter range: about 85% of analog signals use low-resolution 4-bit conversion, while higher-resolution residuals are handled by digital circuits. | R4 keeps a HyAtten-style low-resolution signal-selection row visible with degrade_ratio=0.85 and keep_ratio=0.15, using the same calibrated ADC/DAC tier table but not the selected SUDS policy semantics. | boundary_baseline | low-resolution selection and digital fallback differ from the selected deterministic DPTC scheduler policy | boundary context | literature_baseline_unmeasured_locally | 0.778-0.817 | not_same_scope_dominance_candidate |
| HyAtten | experiments/results/report_data/suds_tetc_same_sim_baselines_20260513_tetc_pivot.json | The local R4 artifact records HyAtten-style rows with literature or unmeasured local accuracy evidence rather than local MPS accuracy linkage. | The row can contextualize conversion tradeoffs, but it is not eligible as an equal-accuracy dominance candidate. | not_dominance_candidate | unmeasured local accuracy | no claim | literature_baseline_unmeasured_locally | 0.778-0.817 | not_same_scope_dominance_candidate |
| TeMPO | kb_markdown/01_transformer_attention_photonic/2402.07393_TeMPO_Transformer_Acceleration_with_Co-packaged_Silicon_Photonics.md | TeMPO uses a time-multiplexed dynamic photonic tensor accelerator with customized slow-light MZM devices and hierarchical photocurrent and temporal integration. | R4 includes a TeMPO-style time-multiplexed readout boundary row inside the same simulator accounting surface, but the local implementation remains the selected Lightening-style DPTC fabric. | boundary_baseline | time multiplexing, readout mode, and customized device/circuit assumptions differ from the selected DPTC fabric | boundary context | literature_architecture_boundary_unmeasured_locally | 0.817-0.842 | not_same_scope_dominance_candidate |
| TeMPO | experiments/results/report_data/suds_tetc_same_sim_baselines_20260513_tetc_pivot.json | The local R4 artifact records TeMPO-style rows as architecture boundary rows with no local measured accuracy linkage. | The row can flag a lower-readout-cost architecture boundary, but it cannot be used as a same-fabric or equal-accuracy dominance claim. | not_dominance_candidate | unmeasured local accuracy | no claim | literature_architecture_boundary_unmeasured_locally | 0.817-0.842 | not_same_scope_dominance_candidate |
| ASTRA | kb_markdown/01_transformer_attention_photonic/ASTRA_Stochastic_Transformer_Silicon_Photonics_TECS2025.md | ASTRA replaces deterministic DPTC operation with stochastic signed optical multiplication, compute-capable transducers, DAC-removal assumptions, and temporal analog accumulation. | R4 includes an ASTRA-style stochastic boundary row to expose alternate-fabric efficiency, while SUDS remains a deterministic DPTC scheduler/control proposal. | boundary_baseline | stochastic fabric, stochastic format, DAC-removal assumptions, and digital fallback differ from the selected DPTC fabric | boundary context | literature_architecture_boundary_unmeasured_locally | 0.408-0.488 | not_same_scope_dominance_candidate |
| ASTRA | experiments/results/report_data/suds_tetc_same_sim_baselines_20260513_tetc_pivot.json | The local R4 artifact records ASTRA-style rows as stochastic architecture boundary rows with no local measured accuracy linkage. | The row can show alternate-fabric modeled EDP pressure, but it is not a same-fabric SUDS dominator. | not_dominance_candidate | stochastic format plus unmeasured local accuracy | no claim | literature_architecture_boundary_unmeasured_locally | 0.408-0.488 | not_same_scope_dominance_candidate |

## R4 Connection

The crosswalk connects directly to the R4 fairness matrix. Lightening is
the same-scope DPTC reference. HyAtten, TeMPO, and ASTRA remain visible
as boundary rows when their converter-selection, time-multiplexed
readout, stochastic-format, or local-accuracy assumptions differ from
the selected deterministic SUDS DPTC fabric.

R4 lower-EDP boundary rows remain boundary evidence only:

| workload | condition | edp_ratio_vs_lightening | boundary_reason |
| --- | --- | --- | --- |
| bert_base_glue_seq128 | astra_boundary | 0.408 | uses ASTRA-style stochastic optical fabric and digital-fallback boundary instead of the selected deterministic DPTC fabric; accuracy label is literature_architecture_boundary_unmeasured_locally |
| mobilevit_s_transformer_blocks_256 | astra_boundary | 0.488 | uses ASTRA-style stochastic optical fabric and digital-fallback boundary instead of the selected deterministic DPTC fabric; accuracy label is literature_architecture_boundary_unmeasured_locally |
| mobilevit_s_transformer_blocks_256 | hyatten_style | 0.778 | uses HyAtten-style low-resolution signal-selection boundary; accuracy is literature/unmeasured in this local artifact; accuracy label is literature_baseline_unmeasured_locally |
| mobilevit_s_transformer_blocks_256 | tempo_time_multiplexed | 0.842 | uses TeMPO-style time-multiplexed readout boundary instead of the selected Lightening-style DPTC temporal-accumulation readout; accuracy label is literature_architecture_boundary_unmeasured_locally |

## Artifacts

- CSV: `experiments/results/report_data/suds_tetc_baseline_parameter_crosswalk_20260517_r13.csv`
- JSON: `experiments/results/report_data/suds_tetc_baseline_parameter_crosswalk_20260517_r13.json`
- Report: `docs/reports/20260517_suds_tetc_baseline_parameter_crosswalk.md`
- R4 fairness report: `docs/reports/20260513_suds_tetc_same_simulator_baselines.md`

## Regeneration

```bash
make suds-tetc-baseline-crosswalk
```
