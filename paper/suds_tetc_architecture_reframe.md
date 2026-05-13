# SUDS TETC Architecture Reframe

Date: `2026-05-13`
Status: `active IEEE TETC manuscript reframe blueprint`
Active manuscript: `paper/suds_tetc_architecture_manuscript.tex`

## Working Title

Scheduler-Derived Budget Interfaces for Dynamic Photonic Transformer Accelerators

## One-Sentence Thesis

SUDS exposes scheduler timing slack as a quality-budget interface for dynamic
photonic Transformer accelerators, allowing DPTC-style optical tiles to choose
ADC precision, degradation, and pruning budgets jointly with local signal or
model selectors.

## Core Research Question

Can a dynamic photonic Transformer accelerator minimize modeled system energy
under kernel deadlines by converting scheduler slack, a kernel DAG, and current
DPTC tile state into a KEEP/DEGRADE/PRUNE quality budget?

This is the active architecture problem. SUDS is not an empirical rule that
declares slack to be a semantic importance oracle. The scheduler provides
deadline pressure, SUDS emits the budget, and local L1/signal/overflow
selectors spend that budget on exact columns. The formal claim contract is
recorded in
`docs/reports/20260513_suds_tetc_formal_claim_contract.md`, with
machine-readable rows in
`experiments/results/report_data/suds_tetc_formal_claim_contract_20260513_tetc_pivot.csv`
and `.json`.

## Draft Abstract

Dynamic photonic tensor cores can accelerate Transformer attention and
feed-forward layers, but their system efficiency is constrained by signal
conversion, operand serialization, and the lack of a control path that connects
scheduler timing pressure to optical resource decisions. We present SUDS, a
scheduler-derived budget interface for dynamic photonic Transformer
accelerators. SUDS derives layer and column slack from the accelerator schedule,
maps slack into KEEP/DEGRADE/PRUNE quality budgets, and composes those budgets
with local L1 or signal-amplitude selectors for ADC precision and pruning
choices. The architecture model maps Transformer attention and FFN kernels to
DPTC-style tiles and accounts for optical compute, conversion, memory movement,
control sidebands, and link sensitivity. Existing evidence already
shows measured MobileViT-S and BERT/GLUE validation, ADC macro tier
ordering, RTL sideband synthesis, and PHY boundary sweeps; the TETC version
will promote only the subset that closes at system level against matched
Lightening-style and HyAtten-style baselines, with TeMPO-style and ASTRA-style
rows retained as boundary fabrics. The work is an architecture and simulation
study, not a fabrication, physical-design, foundry, or bench-energy claim.

## Formal Input/Output Contract

| Contract element | Definition |
|---|---|
| Input: scheduler slack | Hardware-derived normalized slack after DPTC kernel mapping, release/deadline assignment, and tile-latency accounting. |
| Input: kernel DAG | Transformer GEMM dependency graph with matrix dimensions, precedence edges, release times, and deadlines. |
| Input: tile state | DPTC operating point, tile availability, sideband grouping, ADC-sharing mode, memory/link state, and calibration parameters. |
| Output: budget | Per-kernel or per-column-group counts/ratios for `KEEP`, `DEGRADE`, and `PRUNE`. |
| Non-output | Exact column masks; those are selected by local L1, signal, overflow, slack-only, or random policies inside the budget. |

Objective: minimize modeled system energy subject to deadline, precedence,
resource, budget-conservation, accuracy-risk, and calibration-domain
constraints. EDP remains a reported audit metric and pessimistic-gate metric,
but the core problem is deadline-constrained energy minimization.

## Contribution Order

1. **Transformer optical mapping:** Map MHA and FFN GEMM kernels onto a
   DPTC-style dynamic photonic tile with explicit E/O, O/E, memory, and
   scheduler interfaces.
2. **Scheduler-derived SUDS budget interface:** Use scheduler-derived layer and
   column slack as a quality-budget signal for ADC precision, degradation, and
   pruning.
3. **Architecture-level PPA and design-space model:** Report latency, energy,
   area, conversion, memory, optical-link, and control-sideband costs with
   sensitivity bounds and Pareto CSV/JSON artifacts.
4. **Transformer workload validation:** Promote BERT/GLUE plus DeiT or
   MobileViT Transformer-block validation only when both are backed by governed
   MPS runs and linked architecture outputs.
5. **Strong baselines and ablations:** Compare against uniform ADC settings,
   random, L1, slack-only, signal-only, the promoted `suds_pareto` row,
   SUDS-only ablation, SUDS+L1, SUDS+signal, Lightening-style DPTC,
   HyAtten-style selector, TeMPO-style time-multiplexed boundary, and
   ASTRA-style stochastic boundary rows. Only `suds_pareto` is a headline
   SUDS row after the science gate; SUDS+L1 and SUDS+signal remain ablations
   and boundary context.
6. **Circuit calibration:** Use ADC macro, RTL synthesis, and PHY sweeps as
   parameter calibration and boundary evidence, not signoff.

## Headline Claim Contract

| Headline claim | Evidence requirement |
|---|---|
| SUDS is a scheduler-derived budget interface | Manuscript/reframe definition plus architecture evidence-flow figure. |
| SUDS input is slack + DAG + tile state, output is KEEP/DEGRADE/PRUNE budget | Formal contract report plus machine-readable R0 contract CSV/JSON and architecture kernel CSV/JSON with schedule metadata. |
| SUDS has modeled system-level PPA advantage at the selected DPTC point | Architecture simulator report, JSON, summary CSV, sensitivity CSV, and science gate. |
| Promoted accuracy is governed by measured MPS evidence | GLUE measured validation, MobileViT-S conservative Pareto artifact, and GLUE architecture linkage. |
| Baselines and alternate fabrics are visible boundaries | Pivot gate G4 and architecture simulator condition matrix. |
| ADC/RTL/PHY evidence calibrates parameters only | ADC macro, RTL sideband, PHY boundary reports, and architecture parameter table. |

## Required Main Tables And Figures

| Slot | Purpose | Source requirement |
|---|---|---|
| Fig. 1 | Dynamic photonic Transformer accelerator with SUDS control path | New architecture schematic |
| Fig. 2 | Transformer MHA/FFN mapping to DPTC tiles | Simulator mapping artifact |
| Fig. 3 | Slack-to-budget policy and selector composition | Existing SUDS policy, revised caption |
| Fig. 4 | System-level energy/latency/EDP versus baselines | New architecture simulator |
| Fig. 5 | Accuracy-energy Pareto on two Transformer workloads | Governed MPS results |
| Fig. 6 | Sensitivity under ADC/PHY/control assumptions | ADC/RTL/PHY-linked parameters |
| Table 1 | Comparison to Lightening, HyAtten, TeMPO, ASTRA, ENLighten | KB-backed related work |
| Table 2 | Architecture parameters and calibration sources | Literature + SPICE + RTL + PHY |
| Table 3 | Selected tile/control/ADC-sharing operating point | Design-space Pareto artifacts |
| Table 4 | Acceptance gate summary | Pivot artifact gate plus science-strength gate |

## Delta Literature Audit

The 2026-05-13 delta audit does not reopen the architecture design. It adds
related-work pressure points that must be handled in boundary language:

| Work | How to use it |
|---|---|
| ASTRA 2026 arXiv refresh | Cite as the current public stochastic-photonic Transformer surface; keep `astra_boundary` as a boundary fabric. |
| Light-Bound Transformers | Use as robustness/noise-training adjacent work, not as a same-fabric PPA baseline. |
| Integrated electro-optic attention nonlinearities | Use as non-GEMM attention-function boundary; current simulator is GEMM/PPA focused. |
| PRISM | Use as long-context/KV-cache block-selection scope boundary. |
| MXFormer | Use as reviewer-pressure against broad accelerator superiority claims; not photonic/DPTC same-scope. |

## Circuit Macro Placement Rule

Do not hide circuit macro calibration in one sentence. The TETC manuscript needs a visible
calibration paragraph and parameter table:

> We use ngspice ADC macro decks only to calibrate ADC-tier ordering and stress
> sensitivity. These rows determine the relative 4/6/8-bit conversion-energy
> and latency parameters used by the architecture simulator. They do not claim
> foundry readiness, extracted physical design, photonic-front-end signoff, or
> bench-energy measurement.

## Forbidden Wording

- silicon validation claims
- circuit closure from macro-only evidence
- physical-design completion
- hardware energy measurement
- `first photonic Transformer accelerator`
- `state-of-the-art` unless the comparison is matched and complete
- `slack alone is better`
- `GLUE hardware validation` until BERT has a hardware-derived photonic schedule
