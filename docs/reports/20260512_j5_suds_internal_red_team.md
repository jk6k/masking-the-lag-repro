# SUDS J5 Internal Multi-Lens Red-Team

Tag: `20260512_j5_quality_boost`
Review type: `internal_red_team`
Evidence label: `audit`
Promotion decision: `audit`

This is an internal pre-submission quality-control gate. It must not be cited as
external validation, independent peer review, reviewer approval, silicon
validation, SPICE closure, measured hardware-energy evidence, or deployment
readiness.

## Acceptance

- Lenses completed: `4`
- Findings: `8`
- `do_not_use` findings: `0`
- Accepted-risk blockers: `0`
- Gate result: `pass`

## Claim Boundary Memo

SUDS remains a methodology and sideband-interface paper. Slack allocates quality
budgets; local model or physics proxies select exact columns. The SPICE macro
suite calibrates ADC-tier ordering only, while the RTL lane demonstrates a
synthesizable sideband block with proxy timing/area/power accounting. The paper
does not claim foundry/PDK evidence, extracted layout, silicon measurement,
measured hardware energy, SPICE closure, placed-and-routed implementation, or
workload-general deployment readiness.

## Required Questions

| ID | Question |
|---|---|
| Q1 | Is novelty clear after acknowledging HyAtten, ENLighten, Lightening-Transformer, ASTRA, SCATTER, serving schedulers, and approximate computing? |
| Q2 | Does the paper ever imply slack is semantic importance? |
| Q3 | Does SPICE evidence reduce concern, or invite circuit-reviewer rejection by overclaiming? |
| Q4 | Is E6 beating E7 handled as a strength of the composition story? |
| Q5 | Are measured accuracy and modeled energy clearly separated? |
| Q6 | Is the paper too long or too appendix-heavy for JETC? |
| Q7 | Would the paper survive if C1 were downgraded from novelty proof to scoped motivation? |
| Q8 | Does the ADC-Tier Calibration anchor make SPICE visible enough without inviting circuit-closure review? |

## Artifact Snapshot

- ADC-tier anchor count: `1`
- Compact ADC anchor present: `True`
- P3 boundary table present: `True`
- J1 ADC macro: `spice_macro` / `appendix` / `measured`
- J2 RTL: `rtl_synthesis` / `appendix` / `pass`

## Architecture Ai Accelerator

| Finding | Questions | Severity | Disposition | Promotion effect |
|---|---|---|---|---|
| Novelty is strongest when stated as a scheduler-to-accelerator budget interface, not as priority over slack/deadline scheduling, approximate computing, or photonic pruning. | `Q1;Q2;Q4;Q7` | `medium` | `fixed` | main_text wording is eligible; no broader novelty claim promoted |
| E6 L1-signal beating E7 could look like a failed SUDS result unless framed as evidence for composition with local selectors. | `Q4` | `low` | `fixed` | strengthens honesty of measured validation |

Required actions:

- `J5-A1`: Keep adjacent-work boundary table and composition wording; do not say slack is semantic importance or slack-only superiority.
- `J5-A2`: Leave E6 stronger than E7 in the abstract and limitations; present the measured claim as composition, not slack-only superiority.

## Photonic Circuit

| Finding | Questions | Severity | Disposition | Promotion effect |
|---|---|---|---|---|
| The ngspice ADC macro evidence reduces ADC-energy-model skepticism only if it remains macro calibration, not silicon, PDK, extracted-layout, measured hardware-energy, or SPICE closure evidence. | `Q3;Q5;Q8` | `high` | `fixed` | appendix support only; no circuit-closure promotion |
| Yosys synthesis materially improves RTL existence evidence, but timing, gate-equivalent area, and power are still proxy because there is no liberty/OpenROAD/place-and-route lane. | `Q3;Q5` | `medium` | `boundary` | appendix overhead evidence; not implementation closure |

Required actions:

- `J5-C1`: Keep one compact ADC-Tier Calibration anchor in main text; keep decks, stress rows, traces, and regeneration command in appendix/report artifacts.
- `J5-C2`: Use rtl_synthesis for the block existence check and keep timing/GE/power wording as proxy overhead accounting.

## Systems Serving

| Finding | Questions | Severity | Disposition | Promotion effect |
|---|---|---|---|---|
| Serving-scheduler reviewers may read SUDS as a scheduling algorithm unless the interface boundary remains explicit. | `Q1;Q2;Q7` | `medium` | `accepted-risk` | non-blocking risk; paper remains methodology contribution |
| The appendix is dense, but moving full SPICE decks or stress matrices into the main body would hurt JETC fit. | `Q5;Q6` | `low` | `fixed` | main text stays focused |

Required actions:

- `J5-S1`: Maintain wording that serving schedulers manage requests/batches while SUDS exposes an accelerator-internal control surface they could call.
- `J5-S2`: Keep only compact ADC-tier anchor in the main text and leave full deck/stress/regeneration details in report/supplement.

## Reproducibility Artifact

| Finding | Questions | Severity | Disposition | Promotion effect |
|---|---|---|---|---|
| Public reproduction must regenerate the ADC macro suite without private paths, private data, weights, commercial EDA, or PDK dependencies. | `Q5;Q8` | `high` | `fixed` | supports supplement/public repro inclusion |
| C1 remains useful if downgraded from novelty proof to scoped motivation because the paper already argues an interface gap within a frozen technical subset. | `Q6;Q7` | `medium` | `accepted-risk` | non-blocking positioning risk |

Required actions:

- `J5-R1`: Keep deck trace paths repo-relative; expose ngspice as an optional open-source dependency; record Xyce absence as a tool blocker rather than silent fallback.
- `J5-R2`: Do not strengthen C1 beyond the frozen 167-item SUDS technical subset; keep venue/routing cards out of C1 evidence.


## Regeneration

```bash
.venv311-mps/bin/python experiments/tools/build_suds_internal_red_team.py --tag 20260512_j5_quality_boost
```
