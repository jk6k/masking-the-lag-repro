# SUDS TETC Pareto Design Space

Tag: `20260513_tetc_pivot`
Roadmap item: `R5_multi_objective_pareto_design_space`
Evidence label: `multiobjective_pareto_design_space`
Acceptance state: `pass`
Stop-condition state: `no R5 hard stop; raw energy-latency-area dominance is recorded and the claim is narrowed to the governed multi-objective evidence surface`

## Scope

This artifact extends the existing architecture design-space sweep with
R3 measured MPS accuracy linkage and R4 same-simulator baseline
fairness metadata. It is not a new model-evaluation run. The purpose is
to decide whether the selected `suds_pareto` operating point is a
defensible governed Pareto choice under an accuracy budget.

The audit deliberately separates raw energy/latency/area Pareto status
from the R5 multi-objective status. Raw PPA variants that dominate the
selected point stay visible, and the manuscript claim is narrowed away
from global PPA optimality.

## Decision

- R5 acceptance: `pass`
- Blockers: `none`
- Accuracy budget: `1.000` pp
- Selected R5 multi-objective Pareto valid: `True`
- Selected raw energy/latency/area Pareto valid: `False`
- Claim narrowing required for raw unconstrained Pareto: `True`
- Selected point freeze state: `freeze_selected_governed_multiobjective_point`

## Selected `suds_pareto` Rows

| Workload | Design | Delta pp | EDP ratio | Area | Mem pressure | Control ratio | Calibration ratio | Raw PPA front | Raw dominators | R5 front |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `bert_base_glue_seq128` | `td32_t4_c2_sg32_temporal_accum` | 0.000 | 0.666 | 10.414 | 27770880.0 | 0.000118 | 0.956 | `False` | 33 | `True` |
| `mobilevit_s_transformer_blocks_256` | `td32_t4_c2_sg32_temporal_accum` | -0.015 | 0.919 | 6.153 | 4257408.0 | 0.000373 | 0.955 | `False` | 10 | `True` |

## Raw PPA Dominance Pressure

The following examples dominate the selected row on raw energy, latency,
and area. They are not hidden; they are recorded as the reason the
claim must be stated as a governed multi-objective selection rather
than an unconstrained PPA optimum.

| Workload | Design | EDP ratio | Energy ratio | Latency ratio | Area | Sideband | ADC sharing |
|---|---|---:|---:|---:|---:|---:|---|
| `bert_base_glue_seq128` | `td64_t4_c4_sg128_temporal_accum` | 0.666 | 0.708 | 0.941 | 9.563 | 128 | `temporal_accum` |
| `bert_base_glue_seq128` | `td64_t8_c2_sg128_temporal_accum` | 0.666 | 0.708 | 0.941 | 9.563 | 128 | `temporal_accum` |
| `bert_base_glue_seq128` | `td64_t2_c4_sg128_temporal_accum` | 0.666 | 0.708 | 0.941 | 5.523 | 128 | `temporal_accum` |
| `mobilevit_s_transformer_blocks_256` | `td32_t4_c4_sg128_temporal_accum` | 0.918 | 0.917 | 1.001 | 3.855 | 128 | `temporal_accum` |
| `mobilevit_s_transformer_blocks_256` | `td32_t8_c2_sg128_temporal_accum` | 0.918 | 0.917 | 1.001 | 3.855 | 128 | `temporal_accum` |
| `mobilevit_s_transformer_blocks_256` | `td32_t4_c4_sg64_temporal_accum` | 0.919 | 0.917 | 1.002 | 5.008 | 64 | `temporal_accum` |

## R5 Objectives

| Objective | Meaning |
|---|---|
| `energy_pj` | Total modeled energy; lower is better. |
| `latency_ns` | Total modeled latency; lower is better. |
| `edp_pj_ns` | Energy-delay product; lower is better. |
| `area_mm2` | Area proxy; lower is better. |
| `memory_pressure_bytes_per_parallel_core` | Workload memory movement normalized by parallel cores; lower is better. |
| `control_overhead_energy_ratio` | Sideband-control energy share; lower is better. |
| `calibration_sensitivity_energy_ratio` | Conversion, laser, detector, and optical-link energy share used as a calibration-sensitivity proxy; lower is better. |

## Design-Space Coverage

- Input design-space rows: `4536`
- R5 enriched rows: `4536`
- Input raw Pareto rows: `594`
- R3 input acceptance: `pass`
- R4 input acceptance: `pass`

## Artifacts

- Extended design-space CSV: `experiments/results/report_data/suds_tetc_pareto_design_space_20260513_tetc_pivot.csv`
- Extended design-space JSON: `experiments/results/report_data/suds_tetc_pareto_design_space_20260513_tetc_pivot.json`
- Report: `docs/reports/20260513_suds_tetc_pareto_design_space.md`
- Architecture design-space input: `experiments/results/report_data/suds_transformer_architecture_design_space_20260513_tetc_pivot.json`
- R3 accuracy input: `experiments/results/report_data/suds_tetc_end_to_end_accuracy_20260513_tetc_pivot.json`
- R4 baseline fairness input: `experiments/results/report_data/suds_tetc_same_sim_baselines_20260513_tetc_pivot.json`

## Regeneration

```bash
make suds-tetc-pareto-design-space
```
