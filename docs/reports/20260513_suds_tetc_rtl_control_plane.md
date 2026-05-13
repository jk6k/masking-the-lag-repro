# SUDS TETC RTL Control-Plane Upgrade

Tag: `20260513_tetc_pivot`
Roadmap item: `R7_rtl_control_plane_upgrade`
Evidence label: `rtl_synthesis`
Acceptance state: `pass`
Stop-condition state: `no R7 hard stop`

## Scope

R7 upgrades the sideband evidence from a comparator-only proxy to a
synthesizable control path with configuration/budget registers, slack
tiering, sideband command encoding, a tile-command handshake, and a small
SUDS issue state machine. The evidence remains architecture-level and
does not claim placed/routed timing closure, foundry signoff, device
closure, silicon, or bench energy.

## RTL Coverage

| Feature | Present |
|---|---:|
| `budget_registers` | `True` |
| `sideband_encoder` | `True` |
| `tile_command_path` | `True` |
| `suds_state_machine` | `True` |
| `queue_pressure_guard` | `True` |
| `selector_score_guard` | `True` |

## Synthesis And Proxy Contract

- Yosys status: `pass`
- Yosys cell count: `597`
- Proxy area: `1636.4 GE`
- Proxy critical-path delay: `721.0 ps`
- Proxy critical-path slack at 1 GHz: `279.0 ps`
- Command latency model: `3` cycles
- Active logic energy proxy: `0.219640 pJ/command`
- Simulator control term: `0.590400 pJ/sideband group`

The simulator-facing term uses the documented conservative rule:
`max(previous driver-inclusive 32-column sideband anchor, 2.5x R7 active-toggle logic proxy)`.

## Event-Simulator Linkage

| Workload | Arch groups | Event groups | Event control pJ | RTL active pJ | Control share | Margin |
|---|---:|---:|---:|---:|---:|---:|
| `bert_base_glue_seq128` | 13824 | 9720 | 8161.690 | 3036.303 | 0.000118 | 2.688 |
| `mobilevit_s_transformer_blocks_256` | 7204 | 7204 | 4253.242 | 1582.287 | 0.000373 | 2.688 |

## Acceptance

- Acceptance: `pass`
- Max promoted control-energy share: `0.000373`
- Negligible-share threshold: `0.0100`
- Stop condition: `no R7 hard stop`

Because the promoted control-energy share stays below the R7 negligible
threshold, R7 does not trigger the hard stop to rerun the full PPA/science
gate. The event trace still carries the control-sideband energy explicitly,
and the R6 sensitivity lane already records the boundary where exaggerated
control/conversion scaling can erode the claim.

## Artifacts

- RTL: `experiments/hardware/suds_control_plane.v`
- JSON: `experiments/results/report_data/suds_tetc_rtl_control_plane_20260513_tetc_pivot.json`
- Report: `docs/reports/20260513_suds_tetc_rtl_control_plane.md`
- Yosys log: `experiments/results/runs/suds_tetc_rtl_control_plane_20260513_tetc_pivot/yosys_synthesis.log`
- Synthesized Verilog: `experiments/results/runs/suds_tetc_rtl_control_plane_20260513_tetc_pivot/suds_control_plane_synth.v`

## Regeneration

```bash
make suds-tetc-rtl-control-plane
```
