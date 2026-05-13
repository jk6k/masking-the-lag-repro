# SUDS TETC Conservative Pareto Artifact

Tag: `20260513_tetc_pivot`
Evidence label: `measured_mps_imagenet_conservative_pareto`
Promotion decision: `conservative_pareto_ready`

## Scope

This artifact adds a measured MobileViT-S SUDS operating point for the
TETC pivot: `tau_low=0.30`, `tau_high=0.95`, signal-overflow selection,
and zero pruning after mapped tier application. It is promoted only as an
accuracy-guarded Pareto point, not as a relabeling of the previous
aggressive SUDS rows.

## Aggregate Result

| Seeds | Device | Top-1 | Delta Top-1 | ADC ratio | Keep | Degrade | Prune |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | `mps` | 65.466 | -0.015 pp | 0.205 | 0.152 | 0.848 | 0.000 |

## Per-Seed Measurements

| Seed | Top-1 | Top-5 | Delta Top-1 | Processed | Elapsed s |
|---:|---:|---:|---:|---:|---:|
| 0 | 65.434 | 87.084 | -0.046 pp | 50000 | 305.2 |
| 1 | 65.580 | 87.178 | 0.100 pp | 50000 | 294.5 |
| 2 | 65.452 | 87.138 | -0.028 pp | 50000 | 283.7 |
| 3 | 65.486 | 87.104 | 0.006 pp | 50000 | 282.2 |
| 4 | 65.490 | 87.184 | 0.010 pp | 50000 | 282.3 |
| 5 | 65.414 | 87.118 | -0.066 pp | 50000 | 275.5 |
| 6 | 65.394 | 87.066 | -0.086 pp | 50000 | 275.3 |
| 7 | 65.474 | 87.216 | -0.006 pp | 50000 | 275.5 |

## Same-Fabric Context Retained

| Condition | Top-1 mean | Delta Top-1 mean | ADC ratio | Keep | Degrade | Prune |
|---|---:|---:|---:|---:|---:|---:|
| `e0_dense` | 65.480 | 0.000 pp | 1.000 | 1.000 | 0.000 | 0.000 |
| `e2_l1` | 62.010 | -3.470 pp | 0.700 | 0.700 | 0.000 | 0.300 |
| `e3_slack` | 62.825 | -2.656 pp | 0.703 | 0.703 | 0.000 | 0.297 |
| `e6_signal` | 64.142 | -1.338 pp | 0.463 | 0.440 | 0.364 | 0.196 |
| `e7_overlay` | 63.705 | -1.775 pp | 0.463 | 0.440 | 0.364 | 0.195 |
| `e8_overflow` | 63.959 | -1.522 pp | 0.463 | 0.440 | 0.364 | 0.195 |

## Interpretation

The conservative point clears the <=1 pp promoted-accuracy target on the
measured MobileViT-S evidence surface. It does not erase the fact that
stronger energy rows exist at lower accuracy, so downstream claims must
be Pareto-framed rather than single-point superiority claims.

## Regeneration

```bash
make suds-tetc-conservative-pareto
```
