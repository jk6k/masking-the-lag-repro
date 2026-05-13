# SUDS TETC Pre-Review Major Revision

Date: `2026-05-13`
Target venue: `IEEE Transactions on Emerging Topics in Computing`
Review stance: `pre-emptive major revision before initial submission`
Decision: `revise as TETC-only; do not submit as a mixed-route package`

## Venue Gate Interpreted for This Package

The active target is IEEE TETC Technical Tracks. The relevant official gate is
that a manuscript must be research-type and emerging in nature, have
methodological generality, include clear simulation and/or implementation
evaluation, and compare with state-of-the-art solutions. The practical
pre-review translation is:

1. The manuscript must read as an architecture paper, not a scheduler-method
   note with architecture decoration.
2. SUDS must be positioned as a scheduler-derived budget interface for dynamic
   photonic Transformer accelerators.
3. Evidence must include at least two Transformer workloads, system-level PPA,
   measured accuracy linkage, calibration boundaries, and strong baselines.
4. Same-fabric dominance and alternate-fabric wins must be visible rather than
   hidden.
5. Historical methodology-route language must not appear as an active
   submission option in the TETC package.

Primary official source checked:
`https://www.computer.org/digital-library/journals/ec/technical-tracks`.

## Major-Revision Findings

| ID | Severity | Finding | Revision action |
|---|---|---|---|
| MR1 | high | The package could read as "possible review/major revision" rather than "risk-free acceptance" because the PPA layer is modeled and the external red-team is absent. | Keep the conclusion bounded: local submission candidate, not guaranteed acceptance. External red-team remains the first pre-submission risk reducer. |
| MR2 | high | Active docs still exposed legacy fallback-route language, which weakens the TETC positioning. | Active coordination docs and the pivot gate now describe a TETC-only active route; legacy methodology materials are archival provenance only. |
| MR3 | high | The strongest reviewer attack is local-selector dominance: L1/signal/HyAtten/ASTRA can look stronger under some surfaces. | Promote only `suds_pareto`; retain SUDS+L1/SUDS+signal and alternate-fabric rows as ablation or boundary context. |
| MR4 | high | TETC requires clear evaluation and state-of-the-art comparison, so an ADC-only or single-workload story would be desk-rejection-prone. | Keep BERT/GLUE and MobileViT-S, system PPA terms, design-space sweep, Lightening/HyAtten/TeMPO/ASTRA/ENLighten boundaries, and calibration ties in the main package. |
| MR5 | medium | Public repro could accidentally carry legacy route wording through old reports. | The active public-repro manifest no longer exports old route reports or legacy internal red-team artifacts with route-specific wording. |
| MR6 | medium | Internal red-team is useful but not independent. | Gate wording states internal substitute is not equivalent to external review. |

## Submission Positioning After Revision

The correct positioning is:

> This is an IEEE TETC architecture-first submission candidate with credible
> modeled system PPA, measured MPS accuracy linkage, and calibration evidence.
> It has a plausible path to review and major revision. It is not a
> no-risk/steady-acceptance package.

The manuscript should not claim:

- fabricated chip validation;
- physical-design or foundry closure;
- SPICE-level full-system closure;
- bench-measured hardware energy;
- universal dominance over all local selectors or photonic fabrics.

## Remaining Pre-Submission Major-Revision Work

1. Run an external red-team review if scheduling permits; record it separately
   from the internal substitute.
2. Do a final TETC-specific metadata pass: title, abstract, keywords, author
   anonymity, IEEEtran formatting, references, and supplementary naming.
3. Re-run the strict local gate and public-repro check after any wording or
   artifact change.
4. Before upload, scan the final TETC bundle for legacy route names, personal
   paths, private datasets, private KB paths, and overclaim terms.

## Self-Review Outcome

Proceed with the pre-emptive major revision now. The highest-value change is
not adding more aggressive headline numbers; it is making the package read like
one coherent IEEE TETC architecture submission with visible limits, strong
baselines, and no mixed-route fallback language in the active bundle.
