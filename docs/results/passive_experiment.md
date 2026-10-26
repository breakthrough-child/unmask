# Passive-detection experiment: results (pre-registered)

Protocol: [docs/PASSIVE_EXPERIMENT.md](../PASSIVE_EXPERIMENT.md) (pre-registered). **One participant, 3 lighting sessions, 24 clips**; leave-one-session-out; clip-level 95% Wilson intervals are optimistic (clips of a session are correlated).

## Clip-level passive detection (leave-one-session-out, flag if score ≥ 0.5)

| Set | Features | Detection (swapped clips flagged) | False alarms (real clips flagged) | Balanced accuracy | Per-session balanced accuracy (daylight / dim / roomlight) |
|---|---|---|---|---|---|
| T: texture only | 1 | 8/12 = 67% [39–86] | 6/12 = 50% [25–75] | 0.58 | 0.50 / 0.62 / 0.62 |
| B: baseline passive | 7 | 6/12 = 50% [25–75] | 6/12 = 50% [25–75] | 0.50 | 0.50 / 0.50 / 0.50 |
| N: new only | 13 | 10/11 = 91% [62–98] | 10/11 = 91% [62–98] | 0.50 | 0.38 / 0.62 / 0.50 |
| C: baseline + new (primary) | 20 | 9/12 = 75% [47–91] | 10/12 = 83% [55–95] | 0.46 | 0.50 / 0.38 / 0.50 |

## Pre-registered success criterion (set C)

1. balanced accuracy ≥ 0.75: **False** (0.46)
2. ≥ 2 of 3 held-out sessions with balanced accuracy ≥ 0.75: **False** (0/3)
3. C beats baseline B by ≥ 0.10: **False** (C 0.46 vs B 0.50)

**Passive detection improved: NO**

## Per-feature separation (paired: same recording, swapped − real; 12 pairs, 3 sessions)

| Feature | Clips with feature | Median real | Median swapped | Median Δ | Pairs with majority sign | Sessions with that sign | Pre-registered sign | Hypothesis held | Pooled AUC* | Single-feature LOSO bal. acc. |
|---|---|---|---|---|---|---|---|---|---|---|
| `seam_edge` | 24/24 | 0.698 | 0.724 | +0.035 | 8/12 (+) | 2/3 | – | – | 0.56 | 0.54 |
| `seam_flicker` | 24/24 | 0.851 | 0.877 | +0.016 | 8/12 (+) | 2/3 | – | – | 0.56 | 0.54 |
| `seam_edge_cv` | 24/24 | 0.134 | 0.135 | +0.003 | 6/12 (+) | 2/3 | – | – | 0.48 | 0.50 |
| `tex_lap_ratio` | 24/24 | 0.156 | -0.588 | -0.653 | 12/12 (−) | 3/3 | − | yes | 0.31 | 0.58 |
| `tex_hf_ratio` | 24/24 | 0.833 | 0.575 | -0.227 | 11/12 (−) | 3/3 | – | – | 0.39 | 0.50 |
| `landmark_jitter` | 24/24 | 0.027 | 0.026 | -0.005 | 10/12 (−) | 3/3 | – | – | 0.41 | 0.46 |
| `det_score_std` | 24/24 | 0.011 | 0.008 | -0.001 | 10/12 (−) | 3/3 | – | – | 0.45 | 0.58 |
| `p_res_pyr_neck` | 6/24 | -0.329 | -0.378 | -0.049 | 2/3 (−) | 1/1 | − | yes | 0.33 | nan |
| `p_res_pyr_surround` | 22/24 | -0.809 | -0.875 | -0.066 | 7/11 (−) | 2/3 | − | yes | 0.46 | 0.50 |
| `p_res_fft_neck` | 6/24 | -0.739 | -1.091 | -0.318 | 2/3 (−) | 1/1 | − | yes | 0.44 | nan |
| `p_noise_face_vs_neck` | 6/24 | 0.174 | 0.212 | +0.072 | 2/3 (+) | 1/1 | − | no | 0.67 | nan |
| `p_noise_face_vs_bg` | 22/24 | 0.712 | 0.631 | -0.140 | 8/11 (−) | 2/3 | − | yes | 0.42 | 0.50 |
| `p_noise_tvar_ratio` | 4/24 | 0.542 | 0.631 | +0.088 | 2/2 (+) | 1/1 | – | – | 1.00 | nan |
| `p_noise_tcorr_diff` | 6/24 | 0.259 | 0.324 | -0.018 | 2/3 (−) | 1/1 | – | – | 0.44 | nan |
| `p_skin_de_neck` | 20/24 | 28.605 | 26.698 | -1.081 | 8/10 (−) | 3/3 | + | no | 0.44 | 0.55 |
| `p_skin_de_neck_ab` | 20/24 | 11.199 | 11.594 | +0.276 | 5/10 (+) | 1/3 | + | yes | 0.54 | 0.50 |
| `p_skin_de_forehead_ab` | 22/24 | 4.493 | 5.917 | +0.200 | 6/11 (+) | 2/3 | + | yes | 0.51 | 0.50 |
| `p_drop_sep` | 14/24 | 3.565 | 3.626 | -0.119 | 4/7 (−) | 2/3 | + | no | 0.49 | 0.36 |
| `p_drop_minor_frac` | 14/24 | 0.067 | 0.125 | +0.000 | 3/7 (+) | 1/3 | – | – | 0.54 | 0.43 |
| `p_drop_switch_rate` | 14/24 | 0.172 | 0.172 | +0.000 | 1/7 (+) | 0/3 | + | yes | 0.53 | 0.57 |

*Pooled AUC is confounded by session offsets (informational only).

## Controls (face-independent; the swapped clips were encoded one extra time)

| Control | Median real | Median swapped | Median Δ | Pairs with majority sign | Sessions |
|---|---|---|---|---|---|
| `p_ctl_bg_noise` | -0.958 | -0.924 | +0.004 | 6/11 | 1/3 |
| `p_ctl_bg_hf` | -3.322 | -3.348 | +0.025 | 6/11 | 2/3 |
