# SUDS TETC Science-Strength Gate

Tag: `20260513_tetc_pivot`
Evidence label: `science_strength_gate`
Promotion decision: `science_gate_pass_local_submission_candidate`

## Decision

- Pivot artifact gate decision: `tetc_submission_ready`
- Required failures: `none`
- Required partials: `none`
- External red-team required: `False`

This gate is intentionally stricter than the artifact pivot gate. It treats
buildable reports, public-repro alignment, and internal red-team artifacts
as necessary but not sufficient for TETC submission readiness.

## Gate Table

| Gate | Required | Status | Evidence | Blockers | Next action |
|---|---:|---|---|---|---|
| S0 Artifact gate is necessary but not sufficient | `True` | `pass` | pivot_decision=tetc_submission_ready; architecture_status=pass | `` | Keep the pivot gate as artifact-pack readiness only. |
| S1 Promoted SUDS Pareto row is competitive against measured same-fabric baselines | `True` | `pass` | bert_base_glue_seq128: suds_pareto edp=0.666 delta=0.000; mobilevit_s_transformer_blocks_256: suds_pareto edp=0.919 delta=-0.015 | `` | Rework the policy, operating point, or claim so SUDS is not dominated by measured same-fabric selectors. |
| S1b R4 same-simulator strong-baseline fairness matrix passes | `True` | `pass` | r4=pass; stop=no R4 hard stop; rows=28; same_scope=14; boundary=6; dominators=0 | `` | Regenerate R4 and revise the claim if any same-scope baseline dominates SUDS under equal accuracy. |
| S2 Promoted accuracy loss stays within a TETC-grade budget | `True` | `pass` | worst_promoted_suds_delta_pp=-0.015 | `` | Target <=1 pp loss for promoted rows or show a full accuracy/EDP Pareto instead of a single headline. |
| S2b R3 end-to-end perturbation accuracy is linked to PPA policy | `True` | `pass` | r3=pass; stop=no R3 hard stop; worst_delta=-0.01450000000000351; rows=18; promoted_rows=2 | `` | Regenerate R3 and ensure promoted MPS rows match architecture tier ratios and R2 trace IDs. |
| S3 Transformer workload grounding is hardware-derived enough for the main claim | `True` | `pass` | glue_tasks=sst2,mrpc,mnli,qqp,qnli,rte; devices=mps; mobilevit_completion=1.0; pivot_glue_link_rows=126; schedule_kernel_rows=1008; schedule_link_rows=126; schedule_sources=dptc_photonic_tile_schedule; original_glue_analytical_slack=True | `` | Replace analytical BERT slack with hardware-derived schedule metadata before calling the route submission-ready. |
| S3b R9 workload generality expansion is visible and bounded | `True` | `pass` | r9=pass; stop=no R9 hard stop; DeiT-Tiny measured accuracy setup blocker is recorded and simulator-only traces are emitted; rows=90; new_workloads=deit_tiny_patch16_224_batch1_r9,deit_tiny_patch16_224_batch4_r9,deit_tiny_patch16_224_batch8_r9; seq=64,128,256,512; batch=1,4,8; mps_metadata=True; min_new_edp_improvement_pct=6.901431998653473 | `` | Regenerate R9 and keep new workload rows as boundary evidence unless governed MPS accuracy exists. |
| S4 Manuscript maturity reaches full journal-paper density | `True` | `pass` | line_count=1081; references=True; figures=True | `` | Expand the TETC manuscript into a complete IEEE journal article with figures and references. |
| S5 Calibration boundaries remain traceable without circuit overclaim | `True` | `pass` | ADC/RTL/PHY parameters are checked against the pivot architecture summary. | `` | Keep SPICE/RTL/PHY evidence as calibration, proxy, or boundary evidence only. |
| S5a R5 governed Pareto selection rationale is explicit | `True` | `pass` | r5=pass; stop=no R5 hard stop; raw energy-latency-area dominance is recorded and the claim is narrowed to the governed multi-objective evidence surface; selection_rationale_rows=30; r5_valid=True; raw_ppa_valid=False; claim_narrowing_required=True; r3=pass; r4=pass | `` | Regenerate R5 and keep raw-PPA non-optimality visible in the selection rationale. |
| S5b R6 memory, conversion, and link sensitivity preserves the bounded claim | `True` | `pass` | r6=pass; stop=no R6 hard stop; nominal and pessimistic named regimes preserve the bounded promoted EDP claim; min_pessimistic_edp_improvement_pct=12.102428098701768; not_beneficial_sweep_rows=2; thin_margin_sweep_rows=6; combined_not_beneficial_rows=16; stacked_boundary_not_beneficial_rows=0; claim_narrowing_required=True | `` | Regenerate R6 and narrow the claim if a realistic memory/conversion/link regime erases the benefit. |
| S5c R7 RTL control-plane upgrade remains bounded and simulator-linked | `True` | `pass` | r7=pass; stop=no R7 hard stop; cells=597; area_ge=1636.4; slack_ps=279.0; control_pj_per_group=0.5904; max_control_share=0.00037261230692448683; contract_vectors=11/11; coverage=yosys_plus_static_contract_vectors; sim_backend=not_available | `` | Regenerate R7 and rerun PPA/science gate if control overhead is no longer negligible. |
| S5d R8 ADC and photonic calibration ranges remain bounded | `True` | `pass` | r8=pass; stop=no R8 hard stop; ranges remain calibration and boundary evidence only; rows=13; adc_macro=measured; phy_pass=346; phy_fail=230; r6=pass; r7=pass; checksums=True; provenance=True; monotonic=True; claim_scan=pass; forbidden_matches=0 | `` | Regenerate R8 and keep the paper wording at calibration/boundary level if device closure is absent. |
| S5e R10 failure cases and uncertainty bound the promoted claim | `True` | `pass` | r10=pass; stop=no R10 hard stop; target-regime uncertainty preserves positive EDP advantage; rows=29; families=activation_sensitive_layers,conversion_dominated_boundary,high_memory_pressure,long_sequence,low_slack,signal_only_dominant,small_batch; should_not_use_rows=28; target_min_ci95_lower_pct=5.838487076004698; target_crosses_zero=False; boundary_crosses_zero=True; accuracy_bootstrap=True | `` | Regenerate R10 and narrow the manuscript claim if target-regime uncertainty crosses zero advantage. |
| S6 External red-team advisory | `False` | `pass` | internal=pass; external=explicitly_abandoned; external_required=False; equivalence=not_equivalent_to_external_review | `` | External red-team remains useful but is not a hard local blocker per user instruction. |
| S7 R11 public mini-benchmark and internal closure record passes | `True` | `pass` | r11=pass; stop=no R11 hard stop; manifest=pass; public_repro=pass; validation_errors=0; text_scan=pass; text_matches=0; external=explicitly_abandoned; policy=all_fixed_internally; fixed=4; accepted_risks=0 | `` | Regenerate R11 and rerun public-repro build/check/render/check before treating the public package as reviewer-checkable. |

## Same-Fabric Dominance Findings

| Workload | Promoted SUDS Pareto row | Dominating measured baseline rows |
|---|---|---|
| `bert_base_glue_seq128` | `suds_pareto edp=0.666 delta=0.000` | `none` |
| `mobilevit_s_transformer_blocks_256` | `suds_pareto edp=0.919 delta=-0.015` | `none` |

## Interpretation

All required science-strength gates pass locally. This is the only state in which the package may be called a local submission candidate.
External red-team review is advisory only in this gate and is not counted
as a required failure or partial. The internal substitute is recorded
as useful but not equivalent to independent external review.

## Regeneration

```bash
make suds-tetc-science-gate
```
