# SUDS TETC R13-4 Final Freeze

Date: `2026-05-17`
Plan item: `R13-4`
Acceptance state: `pass`
Release commit: `not_requested`

## Scope

R13-4 freezes the active TETC submission-candidate artifacts after the R13-1
R12 evidence integration and R13-2 baseline parameter crosswalk. R13-3 was not
executed in this freeze; the existing R12g DeiT-Tiny result remains a recorded
vision-generality boundary and no DeiT-Tiny promotion is claimed.

## Gate Results

| Command | Result | Key output |
|---|---|---|
| `make project-status` | pass | `route_consistency=pass`; active figure pack `figures/suds_tetc_20260516_submission_figure_pack` |
| `make repo-hygiene` | pass | `0 error(s), 0 warning(s)` |
| `caffeinate -dimsu make suds-tetc-final-gate` | pass | pivot gate `tetc_submission_ready`; science gate `science_gate_pass_local_submission_candidate` |
| `caffeinate -dimsu make suds-tetc-submission-figure-pack` | pass | regenerated Fig4-Fig7 in the active submission figure pack |
| `python3 experiments/tools/check_figure_numbering_registry.py --pack_dir figures/suds_tetc_20260516_submission_figure_pack` | pass | `active=7`; `main_active=Fig1..Fig7` |
| `caffeinate -dimsu make public-repro-build` | pass | wrote `<public_repro>` |
| `make public-repro-check` before render | pass | `0 error(s)` |
| `caffeinate -dimsu make public-repro-render` | pass | regenerated public Fig4-Fig7 under `build/rendered_figures` |
| `make public-repro-check` after render | pass | `0 error(s)` |
| `cd paper && tectonic -X compile suds_tetc_architecture_manuscript.tex` | pass | wrote `paper/suds_tetc_architecture_manuscript.pdf`; underfull box warnings only |
| exact forbidden-claim phrase scan | pass | no matches for the R13 forbidden claim sentences in the manuscript or supplement |
| `make dirty-audit` | pass | `37 dirty path(s)`, `37 managed`, `0 unmanaged` |

## Claim Discipline

The exact forbidden R13 positive claim sentences from the active plan were
absent from the active manuscript and supplement. The freeze report avoids
restating those sentences verbatim so future text scans can include this file
without self-matching the evidence note.

The manuscript and supplement still include explicit negative guardrails about
silicon, layout, bench-energy, universal transfer, and slack-only semantics.
Those are boundary statements, not positive unsupported claims.

## Dirty State

No release commit was created because the user did not request one. The
post-freeze dirty audit reported only managed roots:

- managed figures: `16`
- managed experiments: `13`
- managed docs: `5`
- managed Makefile: `1`
- managed paper: `1`
- managed submissions: `1`
- unmanaged paths: `0`

The dirty set is therefore attributable to the managed R13/final-freeze bundle
and regenerated artifacts. Upload should still use the active manuscript,
active seven-figure pack, supplement, and public reproduction package generated
by this freeze.

## Final Decision

R13-4 passes. The TETC package is locally submission-ready subject to the
remaining venue-specific IEEE TETC formatting, bibliography, submission
metadata, and institutional target-journal partition checks.
