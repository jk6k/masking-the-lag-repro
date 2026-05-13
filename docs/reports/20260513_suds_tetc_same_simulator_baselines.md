# SUDS TETC Same-Simulator Strong Baselines

Tag: `20260513_tetc_pivot`
Roadmap item: `R4_same_simulator_strong_baselines`
Evidence label: `same_simulator_baseline_fairness`
Acceptance state: `pass`
Stop-condition state: `no R4 hard stop`

## Scope

This artifact rebuilds the baseline comparison surface from the same
TETC architecture simulator configuration and joins it to R2 scheduler
traces plus R3 MPS accuracy linkage where measured accuracy exists.
It is a fairness and dominance audit, not a new model-evaluation run.

The matrix keeps strong local selectors, uniform rows, SUDS ablations,
and alternate-fabric rows visible. Same-scope rows must share the
selected DPTC configuration. Boundary rows must state which assumption
differs before they can be used as reviewer-facing context.

## Decision

- R4 acceptance: `pass`
- Blockers: `none`
- Same-scope assumptions matched: `True`
- Boundary assumptions documented: `True`
- Accuracy labels normalized: `True`
- Promoted rows fully linked: `True`
- Same-scope dominators: `0`

## Fairness Matrix

| Workload | Condition | Scope | Accuracy evidence | EDP ratio | Delta | Fairness | Dominance |
|---|---|---|---|---:|---:|---|---|
| `bert_base_glue_seq128` | `astra_boundary` | `boundary_baseline` | `literature_architecture_boundary_unmeasured_locally` | 0.408 | n/a | `pass` | `not_same_scope_dominance_candidate` |
| `bert_base_glue_seq128` | `hyatten_style` | `boundary_baseline` | `literature_baseline_unmeasured_locally` | 0.817 | n/a | `pass` | `not_same_scope_dominance_candidate` |
| `bert_base_glue_seq128` | `tempo_time_multiplexed` | `boundary_baseline` | `literature_architecture_boundary_unmeasured_locally` | 0.817 | n/a | `pass` | `not_same_scope_dominance_candidate` |
| `bert_base_glue_seq128` | `suds_pareto` | `promoted_suds_pareto` | `measured_mps_glue` | 0.666 | 0.000 | `pass` | `promoted_reference` |
| `bert_base_glue_seq128` | `l1` | `same_scope_baseline` | `measured_mps_glue` | 0.665 | 0.000 | `pass` | `not_dominating_promoted_suds` |
| `bert_base_glue_seq128` | `lightening_dptc` | `same_scope_baseline` | `measured_mps_glue` | 1.000 | 0.000 | `pass` | `not_dominating_promoted_suds` |
| `bert_base_glue_seq128` | `random` | `same_scope_baseline` | `unmeasured_random_architecture_boundary` | 0.665 | n/a | `pass` | `not_measured_for_equal_accuracy_dominance` |
| `bert_base_glue_seq128` | `signal_only` | `same_scope_baseline` | `unmeasured_signal_architecture_boundary` | 0.754 | n/a | `pass` | `not_measured_for_equal_accuracy_dominance` |
| `bert_base_glue_seq128` | `slack_only` | `same_scope_baseline` | `measured_mps_glue` | 0.666 | 0.000 | `pass` | `not_dominating_promoted_suds` |
| `bert_base_glue_seq128` | `uniform_4bit` | `same_scope_baseline` | `unmeasured_accuracy_boundary` | 0.976 | n/a | `pass` | `not_measured_for_equal_accuracy_dominance` |
| `bert_base_glue_seq128` | `uniform_8bit` | `same_scope_baseline` | `measured_mps_glue` | 1.128 | 0.000 | `pass` | `not_dominating_promoted_suds` |
| `bert_base_glue_seq128` | `suds_l1` | `same_simulator_suds_ablation` | `measured_mps_glue` | 0.754 | 0.000 | `pass` | `not_same_scope_dominance_candidate` |
| `bert_base_glue_seq128` | `suds_only` | `same_simulator_suds_ablation` | `measured_mps_glue` | 0.754 | 0.000 | `pass` | `not_same_scope_dominance_candidate` |
| `bert_base_glue_seq128` | `suds_signal` | `same_simulator_suds_ablation` | `measured_mps_glue` | 0.754 | 0.000 | `pass` | `not_same_scope_dominance_candidate` |
| `mobilevit_s_transformer_blocks_256` | `astra_boundary` | `boundary_baseline` | `literature_architecture_boundary_unmeasured_locally` | 0.488 | n/a | `pass` | `not_same_scope_dominance_candidate` |
| `mobilevit_s_transformer_blocks_256` | `hyatten_style` | `boundary_baseline` | `literature_baseline_unmeasured_locally` | 0.778 | n/a | `pass` | `not_same_scope_dominance_candidate` |
| `mobilevit_s_transformer_blocks_256` | `tempo_time_multiplexed` | `boundary_baseline` | `literature_architecture_boundary_unmeasured_locally` | 0.842 | n/a | `pass` | `not_same_scope_dominance_candidate` |
| `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `promoted_suds_pareto` | `measured_mps_imagenet` | 0.919 | -0.015 | `pass` | `promoted_reference` |
| `mobilevit_s_transformer_blocks_256` | `l1` | `same_scope_baseline` | `measured_mps_imagenet` | 0.666 | -3.470 | `pass` | `not_dominating_promoted_suds` |
| `mobilevit_s_transformer_blocks_256` | `lightening_dptc` | `same_scope_baseline` | `measured_mps_imagenet` | 1.000 | 0.000 | `pass` | `not_dominating_promoted_suds` |
| `mobilevit_s_transformer_blocks_256` | `random` | `same_scope_baseline` | `measured_mps_imagenet` | 0.666 | -3.599 | `pass` | `not_dominating_promoted_suds` |
| `mobilevit_s_transformer_blocks_256` | `signal_only` | `same_scope_baseline` | `measured_mps_imagenet` | 0.744 | -1.338 | `pass` | `not_dominating_promoted_suds` |
| `mobilevit_s_transformer_blocks_256` | `slack_only` | `same_scope_baseline` | `measured_mps_imagenet` | 0.671 | -2.656 | `pass` | `not_dominating_promoted_suds` |
| `mobilevit_s_transformer_blocks_256` | `uniform_4bit` | `same_scope_baseline` | `unmeasured_accuracy_boundary` | 0.902 | n/a | `pass` | `not_measured_for_equal_accuracy_dominance` |
| `mobilevit_s_transformer_blocks_256` | `uniform_8bit` | `same_scope_baseline` | `measured_mps_imagenet` | 1.522 | 0.000 | `pass` | `not_dominating_promoted_suds` |
| `mobilevit_s_transformer_blocks_256` | `suds_l1` | `same_simulator_suds_ablation` | `measured_mps_imagenet` | 0.746 | -1.775 | `pass` | `not_same_scope_dominance_candidate` |
| `mobilevit_s_transformer_blocks_256` | `suds_only` | `same_simulator_suds_ablation` | `measured_mps_imagenet` | 0.746 | -2.127 | `pass` | `not_same_scope_dominance_candidate` |
| `mobilevit_s_transformer_blocks_256` | `suds_signal` | `same_simulator_suds_ablation` | `measured_mps_imagenet` | 0.746 | -1.522 | `pass` | `not_same_scope_dominance_candidate` |

## Boundary Assumptions

| Workload | Boundary row | Stated difference |
|---|---|---|
| `bert_base_glue_seq128` | `astra_boundary` | uses ASTRA-style stochastic optical fabric and digital-fallback boundary instead of the selected deterministic DPTC fabric; accuracy label is literature_architecture_boundary_unmeasured_locally |
| `bert_base_glue_seq128` | `hyatten_style` | uses HyAtten-style low-resolution signal-selection boundary; accuracy is literature/unmeasured in this local artifact; accuracy label is literature_baseline_unmeasured_locally |
| `bert_base_glue_seq128` | `tempo_time_multiplexed` | uses TeMPO-style time-multiplexed readout boundary instead of the selected Lightening-style DPTC temporal-accumulation readout; accuracy label is literature_architecture_boundary_unmeasured_locally |
| `mobilevit_s_transformer_blocks_256` | `astra_boundary` | uses ASTRA-style stochastic optical fabric and digital-fallback boundary instead of the selected deterministic DPTC fabric; accuracy label is literature_architecture_boundary_unmeasured_locally |
| `mobilevit_s_transformer_blocks_256` | `hyatten_style` | uses HyAtten-style low-resolution signal-selection boundary; accuracy is literature/unmeasured in this local artifact; accuracy label is literature_baseline_unmeasured_locally |
| `mobilevit_s_transformer_blocks_256` | `tempo_time_multiplexed` | uses TeMPO-style time-multiplexed readout boundary instead of the selected Lightening-style DPTC temporal-accumulation readout; accuracy label is literature_architecture_boundary_unmeasured_locally |

## Same-Scope Dominance Check

| Workload | Condition | Result | Reason |
|---|---|---|---|
| `bert_base_glue_seq128` | `l1` | `not_dominating_promoted_suds` | edp_better=False; accuracy_no_worse=True; baseline_edp=0.665; baseline_delta=0.000 pp |
| `bert_base_glue_seq128` | `lightening_dptc` | `not_dominating_promoted_suds` | edp_better=False; accuracy_no_worse=True; baseline_edp=1.000; baseline_delta=0.000 pp |
| `bert_base_glue_seq128` | `random` | `not_measured_for_equal_accuracy_dominance` | accuracy_evidence_label=unmeasured_random_architecture_boundary |
| `bert_base_glue_seq128` | `signal_only` | `not_measured_for_equal_accuracy_dominance` | accuracy_evidence_label=unmeasured_signal_architecture_boundary |
| `bert_base_glue_seq128` | `slack_only` | `not_dominating_promoted_suds` | edp_better=False; accuracy_no_worse=True; baseline_edp=0.666; baseline_delta=0.000 pp |
| `bert_base_glue_seq128` | `uniform_4bit` | `not_measured_for_equal_accuracy_dominance` | accuracy_evidence_label=unmeasured_accuracy_boundary |
| `bert_base_glue_seq128` | `uniform_8bit` | `not_dominating_promoted_suds` | edp_better=False; accuracy_no_worse=True; baseline_edp=1.128; baseline_delta=0.000 pp |
| `mobilevit_s_transformer_blocks_256` | `l1` | `not_dominating_promoted_suds` | edp_better=True; accuracy_no_worse=False; baseline_edp=0.666; baseline_delta=-3.470 pp |
| `mobilevit_s_transformer_blocks_256` | `lightening_dptc` | `not_dominating_promoted_suds` | edp_better=False; accuracy_no_worse=True; baseline_edp=1.000; baseline_delta=0.000 pp |
| `mobilevit_s_transformer_blocks_256` | `random` | `not_dominating_promoted_suds` | edp_better=True; accuracy_no_worse=False; baseline_edp=0.666; baseline_delta=-3.599 pp |
| `mobilevit_s_transformer_blocks_256` | `signal_only` | `not_dominating_promoted_suds` | edp_better=True; accuracy_no_worse=False; baseline_edp=0.744; baseline_delta=-1.338 pp |
| `mobilevit_s_transformer_blocks_256` | `slack_only` | `not_dominating_promoted_suds` | edp_better=True; accuracy_no_worse=False; baseline_edp=0.671; baseline_delta=-2.656 pp |
| `mobilevit_s_transformer_blocks_256` | `uniform_4bit` | `not_measured_for_equal_accuracy_dominance` | accuracy_evidence_label=unmeasured_accuracy_boundary |
| `mobilevit_s_transformer_blocks_256` | `uniform_8bit` | `not_dominating_promoted_suds` | edp_better=False; accuracy_no_worse=True; baseline_edp=1.522; baseline_delta=0.000 pp |

## Boundary Lower-EDP Rows

These rows are reported because they have lower modeled EDP than the
promoted SUDS row, but they are not same-scope dominators.

| Workload | Boundary row | EDP ratio | Boundary reason |
|---|---|---:|---|
| `bert_base_glue_seq128` | `astra_boundary` | 0.408 | uses ASTRA-style stochastic optical fabric and digital-fallback boundary instead of the selected deterministic DPTC fabric; accuracy label is literature_architecture_boundary_unmeasured_locally |
| `mobilevit_s_transformer_blocks_256` | `astra_boundary` | 0.488 | uses ASTRA-style stochastic optical fabric and digital-fallback boundary instead of the selected deterministic DPTC fabric; accuracy label is literature_architecture_boundary_unmeasured_locally |
| `mobilevit_s_transformer_blocks_256` | `hyatten_style` | 0.778 | uses HyAtten-style low-resolution signal-selection boundary; accuracy is literature/unmeasured in this local artifact; accuracy label is literature_baseline_unmeasured_locally |
| `mobilevit_s_transformer_blocks_256` | `tempo_time_multiplexed` | 0.842 | uses TeMPO-style time-multiplexed readout boundary instead of the selected Lightening-style DPTC temporal-accumulation readout; accuracy label is literature_architecture_boundary_unmeasured_locally |

## Artifacts

- Fairness CSV: `experiments/results/report_data/suds_tetc_baseline_fairness_20260513_tetc_pivot.csv`
- Same-simulator JSON: `experiments/results/report_data/suds_tetc_same_sim_baselines_20260513_tetc_pivot.json`
- Report: `docs/reports/20260513_suds_tetc_same_simulator_baselines.md`
- Architecture summary input: `experiments/results/report_data/suds_transformer_architecture_sim_20260513_tetc_pivot_summary.csv`
- R2 scheduler trace input: `experiments/results/report_data/suds_tetc_scheduler_traces_20260513_tetc_pivot.csv`
- R3 accuracy input: `experiments/results/report_data/suds_tetc_end_to_end_accuracy_20260513_tetc_pivot.json`

## Regeneration

```bash
make suds-tetc-same-sim-baselines
```
