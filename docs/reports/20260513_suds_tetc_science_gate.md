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
| S4 Manuscript maturity reaches full journal-paper density | `True` | `pass` | line_count=1353; references=True; figures=True | `` | Expand the TETC manuscript into a complete IEEE journal article with figures and references. |
| S5 Calibration boundaries remain traceable without circuit overclaim | `True` | `pass` | ADC/RTL/PHY parameters are checked against the pivot architecture summary. | `` | Keep SPICE/RTL/PHY evidence as calibration, proxy, or boundary evidence only. |
| S6 External red-team advisory | `False` | `pass` | internal=pass; external=not_available_in_local_execution; external_required=False; equivalence=not_equivalent_to_external_review | `` | External red-team remains useful but is not a hard local blocker per user instruction. |

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
