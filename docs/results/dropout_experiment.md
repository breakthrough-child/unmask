# Experiment 2: within-clip swap dropout, results

Protocol: [PASSIVE_EXPERIMENT.md](../PASSIVE_EXPERIMENT.md) (Experiment 2, pre-registered, run once). **One participant, 3 sessions, 12 clips (hand_pass + head_turn; 6 real, 6 swapped), 6 paired recordings.** Hypothesis: swapped − real > 0 for all three features.

## Per-pair direction (same recording: swapped − real)

| Feature | Session | Challenge | Real | Swapped | Δ (swapped − real) |
|---|---|---|---|---|---|
| `d_tex_change` | daylight | hand_pass | -0.069 | -0.043 | +0.026 |
| `d_tex_change` | daylight | head_turn | -0.497 | -0.416 | +0.082 |
| `d_tex_change` | dim | hand_pass | 0.061 | 0.135 | +0.074 |
| `d_tex_change` | dim | head_turn | -0.560 | -0.408 | +0.151 |
| `d_tex_change` | roomlight | hand_pass | -0.219 | -0.114 | +0.104 |
| `d_tex_change` | roomlight | head_turn | -0.321 | -0.418 | -0.097 |
| `d_tex_maxjump` | daylight | hand_pass | 1.306 | 1.148 | -0.157 |
| `d_tex_maxjump` | daylight | head_turn | 0.666 | 0.600 | -0.067 |
| `d_tex_maxjump` | dim | hand_pass | 1.165 | 0.730 | -0.435 |
| `d_tex_maxjump` | dim | head_turn | 0.755 | 0.450 | -0.305 |
| `d_tex_maxjump` | roomlight | hand_pass | 1.126 | 0.960 | -0.165 |
| `d_tex_maxjump` | roomlight | head_turn | 0.423 | 0.413 | -0.011 |
| `d_out_frac` | daylight | hand_pass | 0.413 | 0.478 | +0.065 |
| `d_out_frac` | daylight | head_turn | 0.200 | 0.000 | -0.200 |
| `d_out_frac` | dim | hand_pass | 0.175 | 0.103 | -0.072 |
| `d_out_frac` | dim | head_turn | 0.160 | 0.120 | -0.040 |
| `d_out_frac` | roomlight | hand_pass | 0.156 | 0.174 | +0.018 |
| `d_out_frac` | roomlight | head_turn | 0.000 | 0.000 | +0.000 |

## Separation with intervals

| Feature | Pairs Δ > 0 (95% exact CI) | Median Δ (95% bootstrap CI over pairs) | Session-mean Δ (daylight / dim / roomlight) | Sessions with Δ > 0 | Criterion 1 |
|---|---|---|---|---|---|
| `d_tex_change` | 5/6 (36–100%) | +0.078 (-0.036 to +0.128) | +0.054 / +0.113 / +0.004 | 3/3 | met |
| `d_tex_maxjump` | 0/6 (0–46%) | -0.161 (-0.370 to -0.039) | -0.112 / -0.370 / -0.088 | 0/3 | not met |
| `d_out_frac` | 2/6 (4–78%) | -0.020 (-0.136 to +0.042) | -0.067 / -0.056 / +0.009 | 1/3 | not met |

## Leave-one-session-out, clip level (flag if score ≥ 0.5)

| Set | Swapped flagged | Real flagged | Balanced accuracy | Per session (daylight / dim / roomlight) |
|---|---|---|---|---|
| D (all three) | 4/6 = 67% [30–90] | 1/6 = 17% [3–56] | 0.75 | 0.75 / 1.00 / 0.50 |
| `d_tex_change` alone | 3/6 = 50% [19–81] | 3/6 = 50% [19–81] | 0.50 | 0.50 / 0.50 / 0.50 |
| `d_tex_maxjump` alone | 4/6 = 67% [30–90] | 3/6 = 50% [19–81] | 0.58 | 0.50 / 0.75 / 0.50 |
| `d_out_frac` alone | 4/6 = 67% [30–90] | 1/6 = 17% [3–56] | 0.75 | 0.75 / 1.00 / 0.50 |

## Pre-registered success criterion

1. a feature with Δ > 0 in the session mean of all 3 sessions and in ≥ 5 of 6 pairs: **True** (by feature: {'d_tex_change': True, 'd_tex_maxjump': False, 'd_out_frac': False})
2. set D balanced accuracy ≥ 0.75: **True** (0.75); all 3 held-out sessions ≥ 0.75: **False** ({'daylight': 0.75, 'dim': 1.0, 'roomlight': 0.5})

**Criterion met: NO**
