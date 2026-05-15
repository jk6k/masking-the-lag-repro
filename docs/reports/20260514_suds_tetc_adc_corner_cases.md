# SUDS TETC R12h ADC Corner-Case SPICE

Date: `2026-05-14`
Tag: `20260514_r12_reinforcement`
Roadmap item: `R12h_adc_corner_cases`
Evidence label: `adc_macro_corner_spice`
Chosen simulator: `ngspice`

## Boundary

Open-source ADC macro corner evidence for ADC-tier calibration only; not PDK, foundry, extracted-layout, silicon, measured hardware energy, photonic front-end, or SPICE closure.

## Acceptance

- Acceptance state: `pass`
- Planned rows: `36`
- Measured rows: `36`
- Failed rows: `0`
- Blocked rows: `0`
- Ramp monotonicity all pass: `True`
- Energy tier ordering all pass: `True`
- Worst ENOB: `3.5061` in `adc4_ramp_combined_stress`
- Worst SNDR: `22.8665` dB in `adc4_ramp_combined_stress`
- Max energy: `1.2772` pJ in `adc8_sine_vdd_high`
- Max latency: `1000.0000` ps in `adc8_sine_vdd_low`
- Blockers: `none`

## Energy Tier Ordering

| Corner | Stimulus | Measured energy per conversion |
|---|---|---|
| `nominal` | `ramp` | ADC4=0.0660 pJ, ADC6=0.2639 pJ, ADC8=1.0556 pJ |
| `nominal` | `sine` | ADC4=0.0660 pJ, ADC6=0.2639 pJ, ADC8=1.0556 pJ |
| `low_temp` | `ramp` | ADC4=0.0638 pJ, ADC6=0.2553 pJ, ADC8=1.0212 pJ |
| `low_temp` | `sine` | ADC4=0.0638 pJ, ADC6=0.2553 pJ, ADC8=1.0212 pJ |
| `high_temp` | `ramp` | ADC4=0.0759 pJ, ADC6=0.3035 pJ, ADC8=1.2139 pJ |
| `high_temp` | `sine` | ADC4=0.0759 pJ, ADC6=0.3035 pJ, ADC8=1.2139 pJ |
| `vdd_low` | `ramp` | ADC4=0.0534 pJ, ADC6=0.2138 pJ, ADC8=0.8550 pJ |
| `vdd_low` | `sine` | ADC4=0.0534 pJ, ADC6=0.2138 pJ, ADC8=0.8550 pJ |
| `vdd_high` | `ramp` | ADC4=0.0798 pJ, ADC6=0.3193 pJ, ADC8=1.2772 pJ |
| `vdd_high` | `sine` | ADC4=0.0798 pJ, ADC6=0.3193 pJ, ADC8=1.2772 pJ |
| `combined_stress` | `ramp` | ADC4=0.0615 pJ, ADC6=0.2458 pJ, ADC8=0.9832 pJ |
| `combined_stress` | `sine` | ADC4=0.0615 pJ, ADC6=0.2458 pJ, ADC8=0.9832 pJ |

## Required Artifacts

- CSV: `experiments/results/report_data/suds_tetc_adc_corner_cases_20260514_r12_reinforcement.csv`
- JSON: `experiments/results/report_data/suds_tetc_adc_corner_cases_20260514_r12_reinforcement.json`
- Report: `docs/reports/20260514_suds_tetc_adc_corner_cases.md`
- Deck root: `experiments/spice/suds_adc_macro/generated/20260514_r12_reinforcement_r12h_corners`
- Run root: `experiments/results/runs/suds_tetc_adc_corner_cases_20260514_r12_reinforcement`

## Regeneration

```bash
make suds-tetc-adc-corner-cases
```
