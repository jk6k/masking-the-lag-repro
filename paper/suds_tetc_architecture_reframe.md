# SUDS TETC Architecture Reframe

Date: `2026-05-13`
Status: `manuscript reframe blueprint`
Protected fallback manuscript: `paper/suds_paper_acmart.tex`

## Working Title

Scheduler-Derived Budget Interfaces for Dynamic Photonic Transformer Accelerators

## One-Sentence Thesis

SUDS exposes scheduler timing slack as a quality-budget interface for dynamic
photonic Transformer accelerators, allowing DPTC-style optical tiles to choose
ADC precision, degradation, and pruning budgets jointly with local signal or
model selectors.

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
   random, L1, slack-only, signal-only, SUDS-only ablation, SUDS+L1,
   SUDS+signal, Lightening-style DPTC, HyAtten-style selector, TeMPO-style
   time-multiplexed boundary, and ASTRA-style stochastic boundary rows.
6. **Circuit calibration:** Use ADC macro, RTL synthesis, and PHY sweeps as
   parameter calibration and boundary evidence, not signoff.

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
| Table 4 | Acceptance gate summary | Pivot gate report |

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
