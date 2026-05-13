# SUDS TETC Calibration Ranges

Tag: `20260513_tetc_pivot`
Roadmap item: `R8_adc_and_photonic_calibration_deepening`
Evidence label: `adc_photonic_calibration_ranges`
Acceptance state: `pass`
Stop-condition state: `no R8 hard stop; ranges remain calibration and boundary evidence only`

## Scope

R8 turns single-point ADC and photonic assumptions into explicit
nominal, optimistic, pessimistic, and boundary ranges. The artifact
uses the local ADC macro sanity suite, the PHY pass/fail boundary sweep,
the R6 system-sensitivity audit, and the R7 RTL-control status as inputs.
It does not claim device-solver, foundry, extracted-layout, silicon,
bench-energy, Lumerical, Spectre, or P&R closure.

## Decision

- R8 acceptance: `pass`
- Blockers: `none`
- Device-solver required: `False`
- Calibration-only boundary retained: `True`

## Range Table

| Parameter | Group | Nominal | Pessimistic | Boundary | Unit | Evidence |
|---|---|---:|---:|---:|---|---|
| `adc4_energy_pj` | `adc_tier_energy` | 0.0660 | 0.0660 | 0.0660 | `pJ/conversion` | `spice_macro` |
| `adc4_latency_ps` | `adc_tier_latency` | 1000.0000 | 1414.2136 | 1414.2136 | `ps/conversion` | `spice_macro` |
| `adc6_energy_pj` | `adc_tier_energy` | 0.2639 | 0.2639 | 0.2639 | `pJ/conversion` | `spice_macro` |
| `adc6_latency_ps` | `adc_tier_latency` | 1000.0000 | 1154.7005 | 1154.7005 | `ps/conversion` | `spice_macro` |
| `adc8_energy_pj` | `adc_tier_energy` | 1.0556 | 1.0556 | 1.0556 | `pJ/conversion` | `spice_macro` |
| `adc8_latency_ps` | `adc_tier_latency` | 1000.0000 | 1000.0000 | 1000.0000 | `ps/conversion` | `spice_macro` |
| `modulator_extinction_ratio_db` | `photonic_noise` | 6.0000 | 3.0000 | 3.0000 | `dB` | `parametric_boundary` |
| `detector_crosstalk_db` | `photonic_noise` | -25.0000 | -15.0000 | -15.0000 | `dB` | `parametric_boundary` |
| `phy_pass_ratio` | `photonic_boundary` | 0.6007 | 0.2708 | 0.2708 | `pass fraction` | `parametric_boundary` |
| `phy_laser_power_mw` | `photonic_boundary` | 0.1180 | 0.1999 | 0.9043 | `mW` | `parametric_boundary` |
| `laser_multiplier` | `photonic_sensitivity` | 1.0000 | 1.1500 | 4.0000 | `x` | `system_sensitivity_boundary` |
| `optical_link_loss_scale` | `photonic_sensitivity` | 1.0000 | 1.2500 | 4.0000 | `x` | `system_sensitivity_boundary` |
| `dac_mzm_energy_scale` | `photonic_conversion` | 1.0000 | 1.2500 | 64.0000 | `x` | `system_sensitivity_boundary` |

## Interpretation

- ADC tier energy and latency now have measured corner ranges from the J1 macro suite.
- Modulator/detector noise is represented by ER and crosstalk axes from the PHY boundary sweep.
- Laser, optical-link, and DAC/MZM ranges are tied to the R6 named regimes and boundary sweeps.
- The architecture paper may cite these rows as calibration ranges and boundary evidence only.

## Artifacts

- Calibration CSV: `experiments/results/report_data/suds_tetc_calibration_ranges_20260513_tetc_pivot.csv`
- Calibration JSON: `experiments/results/report_data/suds_tetc_calibration_ranges_20260513_tetc_pivot.json`
- Report: `docs/reports/20260513_suds_tetc_calibration_ranges.md`
- Architecture parameters: `experiments/results/report_data/suds_transformer_architecture_sim_20260513_tetc_pivot_parameters.csv`
- ADC macro input: `experiments/results/report_data/suds_adc_macro_sanity_20260512_j1_quality_boost.csv`
- PHY boundary input: `experiments/results/report_data/suds_phy_circuit_boundary_20260511_p2p3_quality.csv`
- R6 sensitivity input: `experiments/results/report_data/suds_tetc_system_sensitivity_20260513_tetc_pivot.json`

## Regeneration

```bash
make suds-tetc-calibration-ranges
```
