# SUDS TETC Internal Adversarial Review (R12d)

Date: `2026-05-14`
Tag: `20260514_r12_reinforcement`
Roadmap item: `R12d_internal_adversarial_review`

## Acceptance Summary

- Acceptance state: `pass`
- Total lenses: `8`
- Required lenses: `8`
- Fixed: `5` (L1, L3, L5, L7, L8)
- Boundary: `3` (L2, L4, L6)
- Unresolved: `none`
- Missing required: `none`
- All lenses resolved: `True`
- All required covered: `True`

## Lens Overview

| ID | Lens | Severity | State | Finding |
|---|---:|---|---|---|
| `L1` | `glue_selection_bias` | `high` | `fixed` | Original R3 only measured SST-2 and MRPC — the two easiest GLUE tasks — both showing delta=0.000 pp. A reviewer will fla... |
| `L2` | `cross_workload_transfer` | `high` | `accepted_boundary` | Without cross-workload transfer evidence, a reviewer asks: 'Does SUDS need per-workload tuning or is it general?'... |
| `L3` | `rtl_simulation_coverage` | `medium` | `fixed` | R7 RTL control plane had Yosys synthesis but no functional simulation. A reviewer may ask whether the FSM actually works... |
| `L4` | `deit_tiny_generality_gap` | `medium` | `accepted_boundary` | R9 added DeiT-Tiny simulator-only rows but had a weights/dataset blocker for measured accuracy. A reviewer asks: 'Does S... |
| `L5` | `mobilevit_resolution_sensitivity` | `medium` | `fixed` | MobileViT-S was only evaluated at its nominal 256x256 resolution. A reviewer may ask: 'Is the accuracy stable if input r... |
| `L6` | `bert_flat_delta_artifacts` | `high` | `accepted_boundary` | All BERT GLUE conditions in R3 showed delta=0.000 pp, which looks too clean. A reviewer will suspect a measurement artif... |
| `L7` | `adc_calibration_depth` | `medium` | `fixed` | ADC calibration uses a single macro sanity suite. A reviewer may ask about temperature corners, supply variation, or pro... |
| `L8` | `manuscript_claim_audit_consistency` | `high` | `fixed` | The manuscript must not contain forbidden claim language (silicon, foundry, layout, device-solver, bench-energy, univers... |

## Detailed Findings

### L1: Glue Selection Bias

**Severity:** `high`
**Resolution state:** `fixed`

**Finding:** Original R3 only measured SST-2 and MRPC — the two easiest GLUE tasks — both showing delta=0.000 pp. A reviewer will flag this as selection bias toward easy tasks.

**Evidence checked:** R12b per-task GLUE deltas (8 tasks); difficulty distribution: easy=36, medium=60, hard=24

**Resolution:** R12b expanded GLUE coverage to 8 tasks including CoLA (hard) and STS-B (medium). Non-zero deltas on harder tasks (cola, stsb) prove perturbation is not a no-op. Easy-task flatness (SST-2, MRPC) is real under binary-zeroing perturbation, not a selection artifact.

**Promotion effect:** Manuscript can claim GLUE stability across easy and hard tasks with honest per-task variation exposed.

**Consumed artifacts:** experiments/results/report_data/suds_tetc_glue_task_expansion_20260514_r12_reinforcement.json;experiments/results/report_data/suds_tetc_glue_task_expansion_20260514_r12_reinforcement.csv

**Follow-up items:** none

### L2: Cross Workload Transfer

**Severity:** `high`
**Resolution state:** `accepted_boundary`

**Finding:** Without cross-workload transfer evidence, a reviewer asks: 'Does SUDS need per-workload tuning or is it general?'

**Evidence checked:** R12c transfer matrix (2 transfer rows); BERT->MobileViT delta=-3.4702500000000045; MobileViT->BERT delta=0.0

**Resolution:** R12c records both transfer directions. BERT binary L1 -> MobileViT-S: -3.4703 pp (exceeds 1 pp budget). MobileViT-S signal/overflow -> BERT: 0.0000 pp (measured proxy, exact ratio transfer not rerun). Workload-aware calibration is the honest answer; a single universal policy is not claimed.

**Promotion effect:** Manuscript claims workload-aware SUDS calibration, not a single universal policy. Transfer boundaries are visible.

**Consumed artifacts:** experiments/results/report_data/suds_tetc_cross_workload_transfer_20260514_r12_reinforcement.json;experiments/results/report_data/suds_tetc_cross_workload_transfer_20260514_r12_reinforcement.csv;docs/reports/20260514_suds_tetc_r12_deep_reinforcement.md

**Follow-up items:** none

### L3: Rtl Simulation Coverage

**Severity:** `medium`
**Resolution state:** `fixed`

**Finding:** R7 RTL control plane had Yosys synthesis but no functional simulation. A reviewer may ask whether the FSM actually works.

**Evidence checked:** R12a iverilog testbench results: 31/31 checks, 12/12 features

**Resolution:** R12a RTL functional simulation: 31/31 checks pass, 12/12 features exercised. Claim boundary remains functional_simulation_only.

**Promotion effect:** RTL functional simulation closes the synthesis-only gap. Claim remains at functional_simulation_only.

**Consumed artifacts:** experiments/results/report_data/suds_tetc_rtl_simulation_20260514_r12_reinforcement.json;experiments/hardware/suds_control_plane_tb.v;experiments/results/runs/suds_tetc_rtl_simulation_20260514_r12_reinforcement/simulation.log

**Follow-up items:** none

### L4: Deit Tiny Generality Gap

**Severity:** `medium`
**Resolution state:** `accepted_boundary`

**Finding:** R9 added DeiT-Tiny simulator-only rows but had a weights/dataset blocker for measured accuracy. A reviewer asks: 'Does SUDS work on vision transformers beyond MobileViT?'

**Evidence checked:** R12g DeiT-Tiny MPS accuracy: baseline=72.13%, e2_l1 delta=-1.43 pp

**Resolution:** DeiT-Tiny baseline: 72.13% top-1. Under e2_l1: mean delta=-1.43 pp across 3 seeds, exceeding the 1 pp accuracy budget. Recorded as a vision generality boundary: the perturbation policy calibrated on BERT text tasks has a larger effect on vision Transformer weights.

**Promotion effect:** DeiT-Tiny delta recorded as vision generality boundary. The manuscript should not claim universal vision applicability.

**Consumed artifacts:** experiments/results/report_data/suds_tetc_deit_tiny_accuracy_20260514_r12_reinforcement.json;experiments/results/report_data/suds_tetc_deit_tiny_accuracy_20260514_r12_reinforcement.csv

**Follow-up items:** none

### L5: Mobilevit Resolution Sensitivity

**Severity:** `medium`
**Resolution state:** `fixed`

**Finding:** MobileViT-S was only evaluated at its nominal 256x256 resolution. A reviewer may ask: 'Is the accuracy stable if input resolution varies, as it does in real deployments?'

**Evidence checked:** R12e resolution sweep (4 resolutions x 3 seeds x 2 conditions); worst mean delta=-0.04466666666666667 pp at res=160

**Resolution:** R12e MobileViT-S resolution sweep: worst mean delta -0.0447 pp at resolution 160. All resolutions within 1 pp accuracy budget. SUDS policy is resolution-stable across 160-256.

**Promotion effect:** Resolution stability evidence strengthens the MobileViT-S claim or honestly records the sensitivity boundary.

**Consumed artifacts:** experiments/results/report_data/suds_tetc_mobilevit_resolution_accuracy_20260514_r12_reinforcement.json;experiments/results/report_data/suds_tetc_mobilevit_resolution_accuracy_20260514_r12_reinforcement.csv

**Follow-up items:** none

### L6: Bert Flat Delta Artifacts

**Severity:** `high`
**Resolution state:** `accepted_boundary`

**Finding:** All BERT GLUE conditions in R3 showed delta=0.000 pp, which looks too clean. A reviewer will suspect a measurement artifact or seed cherry-picking.

**Evidence checked:** R12f BERT multi-seed (7 seeds); acceptance=boundary_recorded

**Resolution:** R12f multi-seed BERT evaluation reveals a perturbation-mechanism boundary: binary column zeroing (original R3) preserves accuracy (delta=0.000 pp), while Gaussian noise injection causes large seed-sensitive drops. Both mechanisms are recorded as boundary evidence. The flat R3 delta is real for the binary-zeroing mechanism; the manuscript must specify the perturbation mechanism.

**Promotion effect:** The flat-delta finding is explained as a perturbation-mechanism property (binary zeroing vs. noise injection). The manuscript must specify the exact perturbation implementation.

**Consumed artifacts:** experiments/results/report_data/suds_tetc_bert_multiseed_accuracy_20260514_r12_reinforcement.json;experiments/results/report_data/suds_tetc_bert_multiseed_accuracy_20260514_r12_reinforcement.csv

**Follow-up items:** none

### L7: Adc Calibration Depth

**Severity:** `medium`
**Resolution state:** `fixed`

**Finding:** ADC calibration uses a single macro sanity suite. A reviewer may ask about temperature corners, supply variation, or process corners.

**Evidence checked:** R8 calibration ranges (13 total rows); ADC energy/latency rows=6; ADC macro status=measured; R12h measured rows=36; R12h energy ordering=True

**Resolution:** R8 calibration ranges cover 6 ADC energy/latency rows across ADC4/6/8 tiers, and R12h adds 36 measured corner rows with energy-tier-ordering=True. Claim boundary remains calibration/boundary only; no circuit closure is claimed.

**Promotion effect:** ADC calibration is credible for architecture-level modeling. R12h corner cases either close the ADC-depth gap or are recorded as a hard boundary.

**Consumed artifacts:** experiments/results/report_data/suds_tetc_calibration_ranges_20260513_tetc_pivot.json;experiments/results/report_data/suds_tetc_adc_corner_cases_20260514_r12_reinforcement.json;experiments/results/report_data/suds_tetc_calibration_ranges_20260513_tetc_pivot.csv

**Follow-up items:** none

### L8: Manuscript Claim Audit Consistency

**Severity:** `high`
**Resolution state:** `fixed`

**Finding:** The manuscript must not contain forbidden claim language (silicon, foundry, layout, device-solver, bench-energy, universal-interface, semantic-unimportance-proof, optical-device signoff, P&R closure) and must include required boundary markers.

**Evidence checked:** Manuscript scan: 0 forbidden patterns, 0 missing markers

**Resolution:** Manuscript claim audit: zero forbidden claim language matches, all required boundary markers present. Claim language is consistent with the evidence surface.

**Promotion effect:** Claim-audit consistency prevents overclaim and ensures reviewers see honest boundary language.

**Consumed artifacts:** paper/suds_tetc_architecture_manuscript.tex;paper/suds_tetc_architecture_reframe.md

**Follow-up items:** none


## Pending Follow-Up (Not R12d Blockers)

- **R12f** (BERT multi-seed): `complete` — perturbation-mechanism boundary recorded.
- **R12h** (ADC corner cases): `complete - corner cases measured and absorbed into L7`

## Interpretation

This expanded internal adversarial review covers 8 lenses
across the highest-risk reviewer questions. Since external independent reviewer
review is permanently abandoned for this project, the internal review must be
thorough enough to catch issues a reviewer would flag.

All 5 lenses are fixed or accepted as boundaries. 

## Required Artifacts

- CSV: `experiments/results/report_data/suds_tetc_internal_adversarial_review_20260514_r12_reinforcement.csv`
- JSON: `experiments/results/report_data/suds_tetc_internal_adversarial_review_20260514_r12_reinforcement.json`
- Report: `docs/reports/20260514_suds_tetc_internal_adversarial_review.md`

## Regeneration

```bash
make suds-tetc-internal-adversarial-review
```
