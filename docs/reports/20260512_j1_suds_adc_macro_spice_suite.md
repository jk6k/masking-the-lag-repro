# SUDS ADC Macro SPICE Sanity Suite

Tag: `20260512_j1_quality_boost`
Evidence label: `spice_macro`
Promotion decision: `boundary`

## Scope

This J1 artifact materializes an open-source ADC macro sanity suite for the
4/6/8-bit tier ordering used by SUDS. The suite is limited to ADC-tier energy
model calibration and stress-boundary interpretation. It is not a silicon,
foundry, PDK, extracted-layout, measured hardware-energy, photonic front-end,
or SPICE-closure claim.

## Tool Check

| Tool | PATH result |
|---|---|
| ngspice | `not_found` |
| Xyce/xyce | `not_found` |
| chosen simulator | `none` |

Execution status: `blocked_tool_missing`.

## Artifacts

- CSV: `experiments/results/report_data/suds_adc_macro_sanity_20260512_j1_quality_boost.csv`
- JSON: `experiments/results/report_data/suds_adc_macro_sanity_20260512_j1_quality_boost.json`
- Report: `docs/reports/20260512_j1_suds_adc_macro_spice_suite.md`
- Deck root: `experiments/spice/suds_adc_macro/generated/20260512_j1_quality_boost`
- Sweep matrix: `experiments/spice/suds_adc_macro/generated/20260512_j1_quality_boost/sweep_matrix.csv`

## Nominal ADC-Tier Rows

| ADC bits | Status | Expected energy ratio vs 8-bit | Measured energy ratio vs 8-bit | ENOB | SNDR |
|---:|---|---:|---:|---:|---:|
| 4 | `blocked_tool_missing` | 0.0625 | NA | NA | NA |
| 6 | `blocked_tool_missing` | 0.2500 | NA | NA | NA |
| 8 | `blocked_tool_missing` | 1.0000 | NA | NA | NA |

## Stress Coverage

The generated suite includes ramp and sinusoidal stimuli for each ADC tier, with
nominal, low-rate, high-rate, mismatch-stress, jitter-stress, and combined
stress cases. Ramp rows are intended for monotonicity and DNL/INL proxy checks;
sine rows are intended for ENOB/SNDR sanity checks.

## Promotion Decision

`boundary`. The current `20260511_suds_maxq` package remains the fallback
submission package. Because local `ngspice`/`xyce` execution is
`blocked_tool_missing`, this report does not replace the existing
`spice_proxy` ADC appendix artifact or justify any main-text hardware-closure
wording.

## Compact Anchor Policy

Do not add a large main-text SPICE section. After a simulator-backed run
completes, the only appropriate main-text integration is a compact
`ADC-Tier Calibration` anchor saying that an open-source SPICE macro sweep
sanity-checks the ADC-tier energy ordering and is used only to calibrate the
modeled trend, not to claim foundry, extracted-layout, silicon, measured
hardware-energy, or SPICE closure.

## Regeneration

```bash
python3 experiments/tools/run_suds_adc_macro_spice_suite.py --tag 20260512_j1_quality_boost
```

Use `--simulator ngspice` or `--simulator xyce` to force a specific tool, and
`--require-simulator` to fail closed instead of writing boundary artifacts when
no simulator is available.
