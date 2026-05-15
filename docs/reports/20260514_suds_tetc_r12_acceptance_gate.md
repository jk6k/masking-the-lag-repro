# SUDS TETC R12 Acceptance Gate

Date: `2026-05-14`
Tag: `20260514_r12_reinforcement`

## Summary

- Overall status: `pass`
- Pass items: `8`
- Fail items: `0`
- Boundary-accepted items: `3`
- Required items: `8`

## Item Table

| Item | Acceptance | Status | Evidence |
|---|---|---|---|
| `R12a` | `pass` | `pass` | acceptance=pass; pass_count=31; fail_count=0 |
| `R12b` | `pass` | `pass` | acceptance=pass; total_per_seed_rows=120; tasks=8 |
| `R12c` | `boundary_recorded` | `pass` | acceptance=boundary_recorded; transfer_rows=2; delta=-3.4702500000000045; boundary_reasons=[] |
| `R12d` | `pass` | `pass` | acceptance=pass; lenses=8; unresolved=[]; missing_hashes=[] |
| `R12e` | `pass` | `pass` | acceptance=pass; expected_rows=24; worst_mean_delta=-0.04466666666666667 |
| `R12f` | `boundary_recorded` | `pass` | acceptance=boundary_recorded; total_rows=28; seeds_per_condition=7; boundary_reasons=[] |
| `R12g` | `review_boundary` | `pass` | acceptance=review_boundary; mean_delta=-1.428; max_abs_delta=1.548; boundary_reasons=['delta_exceeds_1pp'] |
| `R12h` | `pass` | `pass` | acceptance=pass; measured_rows=36; energy_ordering=True |

## Blockers

- none

## Required Artifacts

- CSV: `experiments/results/report_data/suds_tetc_r12_acceptance_gate_20260514_r12_reinforcement.csv`
- JSON: `experiments/results/report_data/suds_tetc_r12_acceptance_gate_20260514_r12_reinforcement.json`
- Report: `docs/reports/20260514_suds_tetc_r12_acceptance_gate.md`
