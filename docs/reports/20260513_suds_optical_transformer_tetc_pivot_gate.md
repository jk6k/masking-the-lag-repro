# SUDS Optical Transformer TETC Pivot Gate

Tag: `20260513_tetc_pivot`
Evidence label: `audit`
Promotion decision: `tetc_submission_ready`

## Decision

- Primary route: `IEEE TETC architecture-first optical Transformer accelerator`
- Stretch route: `IEEE Transactions on Computers`
- Required failures: `none`
- Required partials: `none`
- Highest-priority next step: Run make suds-tetc-science-gate before treating the route as a local submission candidate; external red-team remains advisory.

## Gate Table

| Gate | Required | Status | Evidence | Blockers | Next action |
|---|---:|---|---|---|---|
| G0 Route lock and TETC-only active package | `True` | `pass` | plan=docs/coordination/active/SUDS_OPTICAL_TRANSFORMER_TETC_PIVOT_PLAN.md; reframe=paper/suds_tetc_architecture_reframe.md | `` | Keep the active package locked to the IEEE TETC route. |
| G1 Full TETC manuscript source | `True` | `pass` | tetc_source=paper/suds_tetc_architecture_manuscript.tex; tetc_title=Scheduler-Derived Budget Interfaces for Dynamic Photonic Transformer Accelerators; g1_release=pass; red_team=pass; public_repro=pass/pass | `` | Keep the architecture-first TETC source promoted only while manuscript, red-team, and public-repro alignment artifacts pass. |
| G2 Two Transformer workloads on governed MPS | `True` | `pass` | MobileViT rows=192/192; GLUE tasks=sst2,mrpc,mnli,qqp,qnli,rte; devices=mps; BERT architecture link rows=126 | `` | Keep GLUE as measured MPS accuracy and architecture-modeled energy; rerun only if perturbation policy changes. |
| G3 System-level PPA advantage | `True` | `pass` | artifact=experiments/results/report_data/suds_transformer_architecture_sim_20260513_tetc_pivot.json; workloads=bert_base_glue_seq128,mobilevit_s_transformer_blocks_256; terms=adc,dac_mzm,detector_tia,laser,memory,optical_link,control_sideband,digital_fallback; min_pessimistic_edp_gain=14.602%; design_space_rows=4536; pareto_rows=594 | `` | Use the architecture simulator summary table as the promoted system-level PPA surface. |
| G4 Strong baselines and ablations | `True` | `pass` | MobileViT conditions=e0_dense,e2_l1,e3_slack,e4_suds,e5_random,e6_signal,e7_overlay,e8_overflow; architecture conditions=astra_boundary,hyatten_style,l1,lightening_dptc,random,signal_only,slack_only,suds_l1,suds_only,suds_pareto,suds_signal,tempo_time_multiplexed,uniform_4bit,uniform_8bit; HyAtten artifact conditions=composition_summary,e6_signal,e7_overlay,e8_overflow | `` | Keep signal-only wins as composition-boundary evidence, not slack-only superiority. |
| G5 SPICE/RTL/PHY calibration tied to architecture parameters | `True` | `pass` | ADC bits=4,6,8; RTL cells=155; PHY pass ratio=0.601; architecture params=adc4_pj,adc6_pj,adc8_pj,astra_stochastic_boundary,control_pj_per_sideband_group,cores_per_tile,dac_pj,enlighten_l1_boundary,frequency_ghz,hyatten_low_resolution_fraction,lightening_adc_temporal_factor,phy_nominal_pass_ratio,phy_pessimistic_laser_multiplier,selected_adc_sharing,selected_sideband_group_cols,sram_global_kib,sram_subarray_kib,tempo_time_multiplexing_boundary,tile_dim,tiles | `` | Preserve ADC/RTL/PHY as calibration and boundary evidence only. |
| G6 Target journal fit | `False` | `pass` | Primary route is IEEE TETC Technical Track; TC remains a separate future stretch target. | `` | Recheck CAS/JCR partition in institutional database before final submission. |

## Evidence Snapshot

- G1 release artifact: status `pass`, manuscript `pass` with `1495` lines, red-team `pass` across `4` lenses, external status `explicitly_abandoned`, public-repro `pass` with validation `pass`.
- MobileViT measured matrix: `192/192` rows, models `mobilevit_s,mobilevit_xs,mobilevit_xxs`, conditions `e0_dense,e2_l1,e3_slack,e4_suds,e5_random,e6_signal,e7_overlay,e8_overflow`.
- MobileViT-S composition boundary: max absolute Top-1 drop among E6/E7/E8 is `1.775` pp at minimum ADC ratio `0.463`.
- BERT/GLUE measured validation: tasks `sst2,mrpc,mnli,qqp,qnli,rte`, devices `mps`, max aggregate delta `0.0` pp; slack source analytical = `True`.
- Architecture simulator: status `pass`, workloads `bert_base_glue_seq128,mobilevit_s_transformer_blocks_256`, conditions `astra_boundary,hyatten_style,l1,lightening_dptc,random,signal_only,slack_only,suds_l1,suds_only,suds_pareto,suds_signal,tempo_time_multiplexed,uniform_4bit,uniform_8bit`, minimum pessimistic EDP gain `14.602`%.
- Architecture design space: `4536` rows, `594` Pareto rows, artifact `experiments/results/report_data/suds_transformer_architecture_design_space_20260513_tetc_pivot.json`.
- BERT GLUE architecture linkage: `126` linked rows, conditions `l1,lightening_dptc,slack_only,suds_l1,suds_only,suds_pareto,suds_signal`.
- Circuit calibration: ADC macro status `measured`, ADC tiers `4,6,8`, RTL Yosys pass `True`, PHY pass ratio `0.601`.

## Interpretation

The pivot route now has an architecture-level Transformer/DPTC simulator when
G3 is passing. This pivot gate is an artifact-pack gate: it checks that the
local architecture, manuscript, red-team substitute, public-repro alignment,
and calibration artifacts exist and agree on claim boundaries. It is still
not the final science-strength gate. Final local submission-candidate wording
must additionally pass `make suds-tetc-science-gate`. The simulator is modeled
system PPA, not bench-energy or circuit signoff. Legacy methodology-route
materials are archival provenance only and are not part of the active TETC
submission route for this artifact gate.

## Regeneration

```bash
make suds-optical-transformer-pivot-gate
```
