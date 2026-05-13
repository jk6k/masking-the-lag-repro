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
- Contract-vector coverage: `11/11` pass
- Simulation backend: `not_available`

The simulator-facing term uses the documented conservative rule:
`max(previous driver-inclusive 32-column sideband anchor, 2.5x R7 active-toggle logic proxy)`.

## Static Contract Vectors

The local environment has no Verilog simulator available, so R7 records
static RTL contract vectors instead of claiming an event-driven RTL
simulation. These vectors cover the command/state semantics used by the
architecture control contract.

| Vector | Category | Status | Expected behavior |
|---|---|---|---|
| `r7_contract_reset_defaults` | `state_reset` | `pass` | reset initializes thresholds, budgets, sideband group, ready state, and output zeros |
| `r7_contract_config_update` | `configuration` | `pass` | cfg_valid_i updates thresholds, budgets, sideband group, queue limit, and score guard |
| `r7_contract_keep_low_slack` | `tiering` | `pass` | low slack maps to KEEP |
| `r7_contract_degrade_mid_slack` | `tiering` | `pass` | mid slack with degrade budget maps to DEGRADE |
| `r7_contract_prune_high_slack` | `tiering` | `pass` | high slack with prune budget or queue pressure maps to PRUNE |
| `r7_contract_score_guard_keep` | `guard` | `pass` | selector-score guard forces KEEP |
| `r7_contract_queue_pressure_prune` | `guard` | `pass` | queue pressure can force PRUNE and is encoded into command bit 31 |
| `r7_contract_overflow_fallback` | `overflow` | `pass` | empty budget fallback raises overflow flag and command bit 30 |
| `r7_contract_tile_not_ready_wait` | `handshake` | `pass` | ISSUE moves to WAIT when tile_ready_i is low |
| `r7_contract_ready_only_idle` | `handshake` | `pass` | ready_o is asserted only while state is IDLE |
| `r7_contract_command_bit_layout` | `encoding` | `pass` | command bits carry tier, kernel id, sideband group, deadline delta, overflow, and queue pressure |

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
- Contract vectors CSV: `experiments/results/report_data/suds_tetc_rtl_control_plane_contract_vectors_20260513_tetc_pivot.csv`
- Report: `docs/reports/20260513_suds_tetc_rtl_control_plane.md`
- Yosys log: `experiments/results/runs/suds_tetc_rtl_control_plane_20260513_tetc_pivot/yosys_synthesis.log`
- Synthesized Verilog: `experiments/results/runs/suds_tetc_rtl_control_plane_20260513_tetc_pivot/suds_control_plane_synth.v`

## Regeneration

```bash
make suds-tetc-rtl-control-plane
```
