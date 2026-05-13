# SUDS Optical Transformer TETC Pivot Plan

Date: `2026-05-13`
Status: `active G1 promotion contract`
Primary route: `IEEE TETC architecture-first optical Transformer accelerator`
Stretch route: `IEEE Transactions on Computers after stronger architecture baselines`
Fallback route: `JSA or protected 20260512 JETC/JSA-methodology package`

## 0. Purpose

This document implements the accepted pivot from a scheduler-interface
methodology paper into an architecture-first optical Transformer accelerator
paper. The new thesis is:

> SUDS is a scheduler-derived budget interface for dynamic photonic Transformer
> accelerators. It exposes scheduler timing slack to DPTC-style optical compute
> tiles as KEEP/DEGRADE/PRUNE budgets, then composes those budgets with local
> model or signal selectors for ADC precision, degradation, and pruning
> decisions.

The old JETC package remains a protected fallback. New work must not weaken or
overwrite the protected fallback unless the TETC gate promotes it.

## 1. Hard Rules

1. Accelerator-backed local runs must use `.venv311-mps`, `--device mps`, and
   `caffeinate -dimsu`.
2. CPU fallback is forbidden for model evaluation, accuracy sweeps, or
   accelerator-backed validation.
3. Circuit macro evidence is calibration evidence unless a future foundry or
   extracted physical-design flow is explicitly added and independently
   reviewed.
4. No manuscript may claim fabricated-chip validation, physical-design closure,
   foundry closure, bench energy measurement, device-solver closure, or
   deployment status.
5. TETC main-text evidence must include at least two Transformer workloads and
   a system-level PPA/accuracy tradeoff, not only ADC-only accounting.
6. SUDS must be framed as a scheduler-derived budget interface. L1, signal
   amplitude, overflow, or accuracy proxies may still choose exact columns.
7. The new route must include strong baselines: uniform 8-bit, uniform 4-bit,
   random, L1, slack-only, SUDS+L1, SUDS+signal/HyAtten-style, and a
   Lightening-Transformer-style DPTC reference model.
8. If a result is incomplete, it stays `boundary` or `appendix`; do not smooth
   it into a stronger story.

## 2. Target Contribution Stack

| ID | Contribution | Required evidence | Main-text status |
|---|---|---|---|
| T1 | Transformer optical accelerator mapping | Attention and FFN mapped to DPTC-style dynamic photonic tiles with dataflow, conversion, memory, and scheduler interfaces | Required |
| T2 | Scheduler-derived SUDS budget interface | Layer/column slack extraction, tier budget policy, and sideband control overhead | Required |
| T3 | Architecture-level PPA simulator and design-space sweep | Latency, energy, area, conversion, memory, optical-link, control, sensitivity accounting, and selected operating point | Required |
| T4 | Transformer workload validation | BERT/GLUE plus DeiT or MobileViT Transformer blocks on governed MPS, with accuracy/energy Pareto | Required |
| T5 | Strong baseline comparison | Uniform, random, L1, signal/HyAtten-style, slack-only, SUDS variants, Lightening-style DPTC, TeMPO boundary, and ASTRA boundary | Required |
| T6 | Circuit calibration | ADC macro, RTL sideband synthesis, PHY boundary sweeps tied back to simulator parameters | Supporting |

## 3. Execution Waves

### Wave A: Route Lock And Manuscript Split

Deliverables:

- `paper/suds_tetc_architecture_reframe.md` defines the new title, abstract,
  contribution order, figure/table map, and wording restrictions.
- `docs/reports/20260513_suds_optical_transformer_tetc_pivot_gate.md` records
  the current readiness state.
- `paper/suds_paper_acmart.tex` stays protected as the fallback manuscript
  until the gate promotes a new full TETC manuscript source.

Acceptance:

- The gate decision is at least `pivot_scaffolded_not_submission_ready`.
- Current status points to this plan as the active route.

### Wave B: Architecture Simulator Closure

Deliverables:

- A Transformer-specific DPTC architecture simulator or extension that reports
  per-layer/per-kernel latency, energy, area, conversion cost, memory movement,
  optical-link cost, and SUDS-control overhead.
- A Lightening-Transformer-style baseline model with matched tile dimensions,
  frequency assumptions, ADC/DAC settings, and workload mapping.
- A manifest tying each simulator parameter to literature, SPICE macro,
  RTL/PHY proxy, or explicit assumption.

Acceptance:

- Simulator outputs are deterministic CSV/JSON artifacts.
- Energy/EDP improvements are system-level, not ADC-only.
- A pessimistic sensitivity setting is included and does not erase the core
  advantage versus the strongest same-scope baseline.

### Wave B+: Architecture Research And Operating-Point Closure

Status: `complete`

Deliverables:

- `docs/reports/20260513_suds_tetc_architecture_optimization_research.md`
  records the Lightening, HyAtten, TeMPO, ASTRA, and ENLighten architecture
  audit.
- `experiments/results/report_data/suds_transformer_architecture_design_space_20260513_tetc_pivot.csv`
  and matching JSON sweep `tile_dim`, tile count, cores per tile, sideband
  group columns, and ADC-sharing mode.
- Selected operating point is frozen at `tile_dim=32`, `tiles=4`,
  `cores_per_tile=2`, `sideband_group_cols=32`, and
  `adc_sharing=temporal_accum`.
- `tempo_time_multiplexed` and `astra_boundary` are modeled boundary rows.

Acceptance:

- `G2/G3/G4/G5` pass in `make suds-optical-transformer-pivot-gate`.
- `G1` remains partial only because manuscript integration/red-team is not yet
  complete.
- No broad additional architecture research is required unless red-team finds a
  fairness or missing-baseline blocker.

### Wave B++: Delta Literature Audit

Status: `complete`

Deliverables:

- `docs/reports/20260513_suds_tetc_delta_literature_audit.md` records the
  latest 2026 adjacent items: ASTRA arXiv refresh, Light-Bound Transformers,
  electro-optic attention nonlinearities, PRISM, and MXFormer.

Acceptance:

- New items are classified as related-work, scope-boundary, or reviewer-pressure
  evidence.
- No item forces a new same-fabric simulator baseline before G1 promotion.

### Wave C: Transformer Workload Closure

Deliverables:

- BERT-base GLUE validation with hardware-derived photonic schedule metadata,
  not only analytical BERT slack.
- One vision Transformer-family workload: preferred `DeiT-T`; acceptable
  fallback `MobileViT-S Transformer blocks` if DeiT support is not ready.
- MPS run metadata for all promoted accuracy rows.

Acceptance:

- At least two Transformer workloads complete under `--device mps`.
- Mean task degradation is within the accepted paper budget or is reported as a
  Pareto curve rather than a single over-strong claim.
- Every row records device, git hash, command, seed, model, dataset/split,
  condition, and linked architecture-simulator output.

### Wave D: Baselines And Ablations

Deliverables:

- Uniform 8-bit, uniform 4-bit, random, L1, slack-only, signal-only,
  SUDS-only, SUDS+L1, and SUDS+signal/overflow conditions.
- Lightening-style DPTC and HyAtten-style selector comparisons under matched
  parameter settings.
- Ablation table separating budget allocation from exact-column selection.

Acceptance:

- The strongest same-scope baseline is named in the main comparison table.
- Any case where signal-only beats SUDS+signal is kept as boundary evidence.
- The paper claims composition benefit only when the matching ablation supports
  it.

### Wave E: Circuit Calibration Closure

Deliverables:

- ADC macro SPICE suite remains runnable and tied to ADC tier parameters.
- RTL sideband synthesis remains tied to control overhead in the architecture
  simulator.
- PHY boundary sweep remains tied to optical-link assumptions.
- A calibration table appears in the main text or early appendix, not as a
  single invisible sentence.

Acceptance:

- SPICE/RTL/PHY artifacts map into the architecture parameter table.
- Each artifact carries the correct evidence label and limitation note.
- No circuit-facing evidence is phrased as closure.

### Wave F: TETC Readiness Gate And Red-Team

Status: `active`

Deliverables:

- `make suds-optical-transformer-pivot-gate` passes.
- Internal red-team covers architecture, photonic/circuit, systems, and
  reviewer-skeptic lenses.
- External red-team is preferred; if unavailable, subagent/internal review is
  recorded as a substitute, not an equivalent replacement.

Acceptance:

- Gate decision is `tetc_submission_ready`.
- No `required` gate remains `fail` or `partial`.
- Manuscript, figures, public repro, and report-data artifacts agree on the
  same claim boundaries.

## 4. Current Known State

- Current fallback manuscript is still protected.
- Current TETC manuscript source exists at
  `paper/suds_tetc_architecture_manuscript.tex`, but G1 remains partial until
  figure/table integration and red-team promotion.
- Current measured evidence includes MobileViT-S and six-task BERT/GLUE
  validation linked to hardware-derived DPTC schedules.
- Current ADC/RTL/PHY evidence is adequate for calibration and boundary support,
  not for circuit or device signoff.
- Current strongest acceptance blocker is manuscript maturity, not architecture
  evidence. Wave B/B+ now supplies the Transformer DPTC simulator, design-space
  sweep, selected operating point, and Lightening/HyAtten/TeMPO/ASTRA boundary
  rows.

## 5. Regeneration

```bash
make suds-optical-transformer-pivot-gate
```

Heavy follow-up runs must be launched with governed MPS commands, for example:

```bash
caffeinate -dimsu .venv311-mps/bin/python experiments/tools/run_suds_glue_bert_eval.py \
  --device mps \
  --tasks sst2,mrpc,mnli,qqp,qnli,rte \
  --seeds 0,1,2
```
