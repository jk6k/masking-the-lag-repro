# SUDS TETC Major-Revision Response Scaffold

Date: `2026-05-14`
Tag: `20260513_tetc_pivot`
Purpose: `pre-write response hooks for likely TETC reviewer objections`

## Response Table

| Reviewer concern | Response stance | Evidence artifacts | Claim boundary |
|---|---|---|---|
| The simulator is static accounting, not architecture evaluation. | Point to R1 event traces, R2 scheduler traces, and R6/R10 sensitivity; offer to expand event traces in revision. | `suds_tetc_event_sim_20260513_tetc_pivot.json`; `suds_tetc_scheduler_traces_20260513_tetc_pivot.json`; `suds_tetc_uncertainty_20260513_tetc_pivot.json` | Modeled architecture/event-level evidence only. |
| SUDS may be dominated by simpler selectors. | Show R4 same-scope equal-accuracy matrix and R5 selected Pareto rationale; keep L1/signal wins visible as boundary or ablation evidence. | `suds_tetc_same_sim_baselines_20260513_tetc_pivot.json`; `suds_tetc_pareto_design_space_20260513_tetc_pivot.json` | Promote only `suds_pareto`. |
| The calibration story overclaims circuit closure. | Reiterate ADC macro, RTL, and PHY artifacts as calibration/proxy/boundary inputs. | `suds_tetc_rtl_control_plane_20260513_tetc_pivot.json`; `suds_tetc_calibration_ranges_20260513_tetc_pivot.json` | No foundry, layout, silicon, bench-energy, or device-solver claim. |
| Workload generality is narrow. | Present R9 as architecture-only generality expansion and state measured accuracy is limited to governed MPS BERT/GLUE and MobileViT-S rows. | `suds_tetc_workload_expansion_20260513_tetc_pivot.json`; `suds_tetc_end_to_end_accuracy_20260513_tetc_pivot.json` | New DeiT-Tiny and sequence/batch rows are not measured-accuracy promotion rows. |
| Failure cases are hidden. | Point to R10 failure families and boundary-regime uncertainty crossing zero; narrow target-regime claims accordingly. | `suds_tetc_failure_suite_20260513_tetc_pivot.csv`; `suds_tetc_uncertainty_20260513_tetc_pivot.json` | SUDS should not be used in named low-slack or boundary regimes without fallback. |
| The public artifact may leak private data or legacy route language. | Point to the whitelist manifest, checksum manifest, public-repro validator, and R11 text-surface scan. | `configs/public_repro_manifest.json`; `checksums_manifest.json`; `suds_tetc_external_red_team_20260513_tetc_pivot.json` | Public package is a compact reader benchmark, not a full private rerun workspace. |

## Revision Discipline

Do not raise headline claims in the response unless the relevant R-phase
artifact and `make suds-tetc-science-gate` are regenerated and pass. If a
reviewer asks for unavailable silicon, foundry, P&R, or bench evidence,
answer by narrowing the claim rather than inventing closure.
