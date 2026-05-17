# SUDS TETC R13-1 R12 Reinforcement Integration

Date: `2026-05-17`
Plan item: `R13-1`
Manuscript: `paper/suds_tetc_architecture_manuscript.tex`
Supplement: `submissions/tetc_20260517_submission/supplement/README.md`

## Purpose

R13-1 makes the existing R12 post-R11 reinforcement evidence visible without
changing the selected `suds_pareto` headline row, adding promoted workloads, or
relaxing the one-percentage-point selected accuracy target.

## Integrated Evidence

| R12 item | Artifact | Acceptance state | Integration decision |
|---|---|---|---|
| R12a | `experiments/results/report_data/suds_tetc_rtl_simulation_20260514_r12_reinforcement.json` | pass | Listed as control-plane functional simulation support. |
| R12b | `experiments/results/report_data/suds_tetc_glue_task_expansion_20260514_r12_reinforcement.json` | pass | Listed as BERT-side GLUE task robustness support. |
| R12c | `experiments/results/report_data/suds_tetc_cross_workload_transfer_20260514_r12_reinforcement.json` | boundary_recorded | Preserved as a workload-transfer boundary. |
| R12d | `experiments/results/report_data/suds_tetc_internal_adversarial_review_20260514_r12_reinforcement.json` | pass | Listed as internal adversarial-review coverage with boundary lenses retained. |
| R12e | `experiments/results/report_data/suds_tetc_mobilevit_resolution_accuracy_20260514_r12_reinforcement.json` | pass | Listed as MobileViT-S resolution robustness support. |
| R12f | `experiments/results/report_data/suds_tetc_bert_multiseed_accuracy_20260514_r12_reinforcement.json` | boundary_recorded | Preserved as a perturbation-mechanism boundary. |
| R12g | `experiments/results/report_data/suds_tetc_deit_tiny_accuracy_20260514_r12_reinforcement.json` | review_boundary | Preserved as a DeiT-Tiny vision-generality boundary. |
| R12h | `experiments/results/report_data/suds_tetc_adc_corner_cases_20260514_r12_reinforcement.json` | pass | Listed as ADC corner-case calibration support. |

## Manuscript And Supplement Edits

- Added a short main-text pointer in `Robustness Checks` that directs readers to
  the supplemental R12 integration table.
- Added a supplemental `R12 Reinforcement Integration Table` with the required
  columns: R12 item, reviewer risk addressed, artifact, outcome, and claim
  effect.
- Added the R12d internal adversarial-review artifact to the supplement evidence
  ledger so all eight R12 items are directly visible.

## Boundary Discipline

R12c, R12f, and R12g remain boundary records. They are not rewritten as
positive generality claims:

- R12c records that cross-workload policy transfer can exceed the one-point
  accuracy budget and therefore requires workload-aware calibration.
- R12f records that Gaussian-noise perturbation is seed-sensitive and separates
  it from the selected binary-column-zeroing mechanism.
- R12g records that the tested DeiT-Tiny setting loses more than one percentage
  point and remains a vision-generality boundary.

## Verification Performed

- `make repo-hygiene`: passed with `0 error(s), 0 warning(s)`.
- `cd paper && tectonic -X compile suds_tetc_architecture_manuscript.tex`:
  passed; Tectonic emitted only underfull box warnings.
- `caffeinate -dimsu make suds-tetc-final-gate`: passed with
  `promotion_decision=science_gate_pass_local_submission_candidate`.
