# SUDS RTL Control-Plane Overhead

Tag: `20260512_j2_quality_boost`
Evidence label: `rtl_synthesis`
Promotion decision: `appendix`

## Scope

This P3 lane materializes a synthesizable SUDS control-plane block:
threshold registers, tier comparator, tier sideband, and validity tags. The
timing model treats it as a one-cycle sideband path. If Yosys is installed,
the script attempts synthesis; otherwise it keeps a transparent proxy estimate.

Yosys status: `pass`
Yosys cell count: `155`

## Summary

- Worst proxy timing slack: `380.0 ps`
- Max proxy area: `2392.0 GE`
- Max proxy dynamic power: `2239.20 uW`
- RTL source: `experiments/hardware/suds_control_plane.v`
- Yosys log: `experiments/results/runs/suds_rtl_control_overhead_20260512_j2_quality_boost/yosys_synthesis.log`
- Synthesized Verilog: `experiments/results/runs/suds_rtl_control_overhead_20260512_j2_quality_boost/suds_control_plane_synth.v`

## Representative Rows

| Slack bits | Columns | Target MHz | Sideband bits | Delay ps | Timing slack ps | Area GE | Power uW | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 12 | 32 | 1000 | 96 | 620.0 | 380.0 | 712.0 | 655.20 | `pass` |
| 12 | 64 | 1000 | 192 | 620.0 | 380.0 | 1048.0 | 972.00 | `pass` |
| 12 | 128 | 1000 | 384 | 620.0 | 380.0 | 1720.0 | 1605.60 | `pass` |
| 12 | 192 | 1000 | 576 | 620.0 | 380.0 | 2392.0 | 2239.20 | `pass` |

## Interpretation

- The control path is treated as sideband metadata. It must not be described as
  accelerating the optical datapath.
- The synthesis result shows the RTL is accepted by Yosys and maps to generic
  cells. The timing, GE, and power columns are still proxy estimates because no
  liberty-backed OpenROAD or placed-and-routed flow is used here.
- TCAS promotion still requires placed/routed or liberty/tool-calibrated area
  and power. This artifact supports JETC appendix overhead accounting and
  reviewer response.

## Artifacts

- CSV: `experiments/results/report_data/suds_rtl_control_overhead_20260512_j2_quality_boost.csv`
- JSON: `experiments/results/report_data/suds_rtl_control_overhead_20260512_j2_quality_boost.json`
- Report: `docs/reports/20260512_j2_quality_boost_suds_rtl_control_overhead.md`
- RTL: `experiments/hardware/suds_control_plane.v`

## Regeneration

```bash
.venv311-mps/bin/python experiments/tools/build_suds_rtl_control_overhead.py --tag 20260512_j2_quality_boost
```
