# SUDS TETC R12 Deep Reinforcement

Date: `2026-05-14`
Tag: `20260514_r12_reinforcement`
Current focus: `R12c_cross_workload_policy_transfer`

## R12c Scope

R12c asks whether the SUDS budget/policy learned on one workload can be applied
to the other without per-workload tuning. This report reads the governed R3/R12
MPS evidence surface and records both transfer directions. It does not create a
new measured-accuracy claim beyond those source artifacts.

## Transfer Matrix

| Transfer | Evidence | Target delta | State | Boundary |
|---|---|---:|---|---|
| `bert_binary_l1_policy_to_mobilevit_s` | `direct_measured_transfer` | -3.4703 pp | `transfer_boundary` | `delta_exceeds_1pp` |
| `mobilevit_signal_policy_family_to_bert` | `measured_policy_family_proxy` | 0.0000 pp | `transfer_boundary` | `exact_ratio_transfer_not_rerun` |

## Interpretation

- BERT-derived binary L1 policy transferred to MobileViT-S records
  `-3.4703` pp top-1 delta, outside the
  `1.0` pp budget, so it is a transfer
  boundary rather than a generality win.
- MobileViT-S conservative signal/overflow policy family transferred back to
  BERT records `0.0000` pp on the
  available measured BERT surface, but the exact MobileViT no-prune ratio was
  not rerun on BERT; this is recorded as a policy-family proxy boundary.
- The R12c answer is therefore not "one universal policy is enough." The
  evidence supports a workload-aware SUDS calibration claim, with conservative
  no-prune vision calibration preserved as the promoted MobileViT-S point.

## Acceptance

Acceptance state: `boundary_recorded`

- Cross-workload transfer rows with measured deltas: `2`
- Rows within 1 pp: `1`
- Boundary rows: `bert_binary_l1_policy_to_mobilevit_s`, `mobilevit_signal_policy_family_to_bert`
- Per-workload tuning required: `True`

## Required Artifacts

- CSV: `experiments/results/report_data/suds_tetc_cross_workload_transfer_20260514_r12_reinforcement.csv`
- JSON: `experiments/results/report_data/suds_tetc_cross_workload_transfer_20260514_r12_reinforcement.json`
- Report: `docs/reports/20260514_suds_tetc_r12_deep_reinforcement.md`

## Regeneration

```bash
make suds-tetc-cross-workload-transfer
```

<!-- R12E_MOBILEVIT_RESOLUTION_SWEEP_START -->
## R12e MobileViT-S Resolution Sweep

Date: `2026-05-14`
Tag: `20260514_r12_reinforcement`
Roadmap item: `R12e_mobilevit_resolution_sweep`

## MobileViT-S Resolution Accuracy Sweep

### Configuration

- Model: `mobilevit_s` (nominal input size: 256)
- Resolutions: `[160, 192, 224, 256]`
- Seeds: `[0, 1, 2]`
- Conditions: `e0_dense` (baseline), `e8_overflow` (promoted SUDS policy)
- Promoted policy: tau_low=`0.3`, tau_high=`0.95`
- Samples per run: `50000` (full ImageNet val)
- Device: `mps` only, CPU fallback forbidden
- Total runs: `24`

### Per-Resolution Summary

| Resolution | e0_dense Top-1 (mean) | e8_overflow Top-1 (mean) | Mean Δ | Seeds (e0/e8) |
|---|---:|---:|---:|---|
| `160` | 55.21% | 55.17% | -0.0447 pp | 3/3 |
| `192` | 61.04% | 61.01% | -0.0253 pp | 3/3 |
| `224` | 63.64% | 63.64% | +0.0020 pp | 3/3 |
| `256` | 65.48% | 65.49% | +0.0087 pp | 3/3 |

### Acceptance

- Acceptance state: `pass`
- Expected rows: `24`, Actual: `24`
- Expected samples per run: `50000`
- All expected samples: `True`
- All 50000 samples: `True`
- All MPS/GPU: `True`
- Accuracy target: `1.0` pp
- Worst mean delta: `-0.04466666666666667` pp at resolution `160`
- Worst single delta: `-0.12` pp (res=`192`, seed=`0`)
- Within budget: `True`

### Interpretation

MobileViT-S accuracy is resolution-stable under the promoted SUDS e8_overflow perturbation policy across all tested resolutions (160-256). The worst mean delta is within the 1.0 pp budget, supporting the claim that the SUDS policy does not depend on a single fixed input resolution.

### Required Artifacts

- CSV: `experiments/results/report_data/suds_tetc_mobilevit_resolution_accuracy_20260514_r12_reinforcement.csv`
- JSON: `experiments/results/report_data/suds_tetc_mobilevit_resolution_accuracy_20260514_r12_reinforcement.json`

### Regeneration

```bash
make suds-tetc-mobilevit-resolution-sweep
```
<!-- R12E_MOBILEVIT_RESOLUTION_SWEEP_END -->
