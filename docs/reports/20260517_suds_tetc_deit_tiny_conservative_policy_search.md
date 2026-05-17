# SUDS TETC R13-3 DeiT-Tiny Conservative Policy Search

Date: `2026-05-17`
Plan item: `R13-3`
Acceptance state: `pass`
Decision: `deit_tiny_secondary_support`

## Scope

This artifact attempts a bounded DeiT-Tiny policy search without
changing the existing `suds_pareto` headline row. It is an MPS-only
accuracy measurement plus R9 architecture join, not a silicon, layout,
device-solver, timing-closure, or bench-energy claim.

## Screening

Screening label: `screening_only`; sample count: `2048`; seed: `0`.

| Policy | Top-1 (%) | Delta (pp) | Selected for full validation |
|---|---:|---:|---|
| no_prune_keep90 | 72.9492 | 0.0488 | no |
| no_prune_keep95 | 72.8027 | -0.0977 | no |
| light_degrade_keep90 | 73.0469 | 0.1465 | yes |
| light_degrade_keep95 | 72.9004 | 0.0000 | no |
| degrade_only_signal_safe | 72.8516 | -0.0488 | yes |

## Full Validation

Selected candidates: `degrade_only_signal_safe`, `light_degrade_keep90`. Each selected policy was run
on the full 50,000-image validation split with seeds `0`, `1`, and
`2`; dense reference rows were rerun under the same command.

| Policy | Seeds | Mean Top-1 (%) | Mean delta (pp) | Worst seed delta (pp) | Min EDP improvement vs Lightening (%) | Decision |
|---|---:|---:|---:|---:|---:|---|
| degrade_only_signal_safe | 3 | 72.0960 | -0.0360 | -0.0800 | 6.901 | pass |
| light_degrade_keep90 | 3 | 72.1313 | -0.0007 | -0.0120 | 0.632 | pass |

## R9 Architecture Join

| Workload | Batch | Policy | EDP ratio vs Lightening | EDP improvement (%) |
|---|---:|---|---:|---:|
| deit_tiny_patch16_224_batch1_r9 | n/a | degrade_only_signal_safe | 0.9310 | 6.901 |
| deit_tiny_patch16_224_batch1_r9 | n/a | light_degrade_keep90 | 0.9937 | 0.632 |
| deit_tiny_patch16_224_batch4_r9 | n/a | degrade_only_signal_safe | 0.9023 | 9.775 |
| deit_tiny_patch16_224_batch4_r9 | n/a | light_degrade_keep90 | 0.9904 | 0.960 |
| deit_tiny_patch16_224_batch8_r9 | n/a | degrade_only_signal_safe | 0.8939 | 10.607 |
| deit_tiny_patch16_224_batch8_r9 | n/a | light_degrade_keep90 | 0.9894 | 1.055 |

## Decision

R13-3 finds `degrade_only_signal_safe` as a conservative secondary DeiT-Tiny support point: mean Top-1 delta -0.0360 pp, worst-seed delta -0.0800 pp, and minimum modeled EDP improvement 6.901% against the same-scope Lightening-style reference.

The earlier R12g `e2_l1` DeiT-Tiny row remains a recorded boundary;
R13-3 only supports the explicitly measured conservative no-prune
policy if its accuracy and modeled-EDP checks pass.

## Artifacts

- CSV: `experiments/results/report_data/suds_tetc_deit_tiny_conservative_policy_search_20260517_r13.csv`
- JSON: `experiments/results/report_data/suds_tetc_deit_tiny_conservative_policy_search_20260517_r13.json`
- Report: `docs/reports/20260517_suds_tetc_deit_tiny_conservative_policy_search.md`
- R9 architecture source: `experiments/results/report_data/suds_tetc_workload_expansion_20260513_tetc_pivot.json`
- R12 DeiT boundary source: `experiments/results/report_data/suds_tetc_deit_tiny_accuracy_20260514_r12_reinforcement.json`

## Regeneration

```bash
caffeinate -dimsu .venv311-mps/bin/python \
  experiments/tools/build_suds_tetc_deit_tiny_conservative_policy_search.py \
  --tag 20260517_r13 \
  --device mps
```
