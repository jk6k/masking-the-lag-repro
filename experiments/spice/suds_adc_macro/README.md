# SUDS ADC Macro SPICE Suite

Evidence label: `spice_macro`
Promotion decision: `boundary` until a local open-source simulator run completes.

This directory contains the generated ngspice/Xyce macro decks for the SUDS J1
ADC-tier sanity suite. The decks are behavioral/macro-level calibration assets
for 4/6/8-bit ADC tier energy ordering, latency, ramp monotonicity, DNL/INL
proxy, and sine ENOB/SNDR stress checks.

They are not PDK, extracted-layout, silicon, measured hardware-energy, or
SPICE-closure evidence. The main manuscript may only use them through a compact
ADC-Tier Calibration anchor after simulator-backed rows exist; full deck and
stress details belong in appendix, supplement, or reports.

Regenerate decks and report-data artifacts with:

```bash
python3 experiments/tools/run_suds_adc_macro_spice_suite.py --tag 20260512_j1_quality_boost
```

Force a specific simulator when installed:

```bash
python3 experiments/tools/run_suds_adc_macro_spice_suite.py --tag 20260512_j1_quality_boost --simulator ngspice
python3 experiments/tools/run_suds_adc_macro_spice_suite.py --tag 20260512_j1_quality_boost --simulator xyce
```

Use `--require-simulator` for release gates that should fail closed instead of
writing a machine-readable tool blocker.
