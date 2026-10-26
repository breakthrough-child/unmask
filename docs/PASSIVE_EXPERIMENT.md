# Passive-detection experiment: pre-registration

**Written and committed before any new feature was computed** (git history is the evidence). Time-box: stop by
17:00 Lagos on 2026-10-26, whatever the outcome. Data: only the existing 24 clips (3 sessions × 4 clips × {real, swapped}),
one participant. No new recordings or swaps. Privacy rule unchanged (only `divine_*` sessions in public outputs).

Question: can passive signals computed on frontal, unoccluded frames of *any* of the 24 clips separate real from
swapped better than the existing passive signals did (§6.1 of the report: detection 2/3, false alarms 2/3)?

## Frames used

A frame is **eligible** when: face score ≥ 0.8; |yaw| ≤ 20° (the pipeline's yaw estimate); not occluded (for `hand_pass`
clips: Farneback/MOG2 motion fraction inside the face box < 0.08, as in the existing occlusion code; other challenges
have no occluder by design); and all patches below lie inside the frame. Every 2nd eligible frame, at most 60 frames per
clip. A clip gets a feature only if it has ≥ 8 eligible frames (≥ 12 for the temporal/dropout features); otherwise the
feature is missing for that clip and the coverage is reported.

## Regions (pixel coordinates of the ≤ 640 px-wide analysis frame; face box x, y, w, h; cx, cy = box centre)

Matched scale = every comparison uses patches of the **same pixel size s = clamp(round(0.30·w), 24, 64)** taken from the
same frame (same optics, no rescaling).

- **Face patch F**: s×s centred at (cx, cy + 0.10·h) (cheek/nose area).
- **Neck patch N**: s×s centred at (cx, y + 1.18·h).
- **Surround patches S**: two s×s patches centred at (cx ∓ 0.75·w, cy) (hair, ears, background beside the head).
- **Background patches G**: the four frame-corner s×s patches.
- **Central face** (colour): ellipse radii (0.25·w, 0.25·h) at (cx, cy + 0.05·h). **Forehead edge**: the annulus
  0.85–1.0 of the face ellipse (radii 0.5·w, 0.5·h) restricted to y < cy − 0.15·h. **Neck** (colour): the existing neck mask.

## Features (all clip-level: median over eligible frames unless stated; log = natural log)

Resolution mismatch (swap models render at low internal resolution and upscale): expected **lower** for swapped.
- `p_res_pyr_neck` = log r(F) − log r(N), `p_res_pyr_surround` = log r(F) − mean(log r(S)), where r = E₀/E₁ is the ratio of
  mean-square energy of the finest to the next Laplacian-pyramid band (cv2.pyrDown/pyrUp) of the grey patch.
- `p_res_fft_neck` = log HF(F) − log HF(N), HF = power at normalised radius > 0.5 over power at 0 < radius ≤ 1 of the
  Hann-windowed 2-D FFT of the grey patch (DC excluded).

Sensor-noise mismatch: noise level n(P) = 25th percentile over 8×8 blocks of the block standard deviation of the residual
(uint8 grey patch − cv2.medianBlur(patch, 3)). Expected: swapped face noise **different from** (probably below) its surroundings.
- `p_noise_face_vs_neck` = log n(F) − log n(N); `p_noise_face_vs_bg` = log n(F) − mean(log n(G)).
- `p_noise_tvar_ratio` = std over frames of log n(F) ÷ std over frames of log n(N).
- `p_noise_tcorr_diff` = mean correlation of the residual between consecutive eligible frames for F minus the same for N
  (only pairs whose box centre moved < 2 px).

Skin-tone mismatch (Lab, ΔE76): expected **larger** for swapped.
- `p_skin_de_neck` = ‖Lab(central face) − Lab(neck)‖ (medians of each region); `p_skin_de_neck_ab` = same on (a, b) only;
  `p_skin_de_forehead_ab` = (a, b) difference between central face and forehead edge.

Swap dropout (brief appearance jumps): per eligible frame the descriptor is [L, a, b of the central face, log Laplacian
variance of F, log HF(F)]; standardised within the clip; `cv2.kmeans` k = 2 (10 attempts, PP centres, `cv2.setRNGSeed(0)`).
- `p_drop_sep` = centroid distance ÷ pooled within-cluster RMS distance; `p_drop_minor_frac` = size of the smaller cluster ÷ n;
  `p_drop_switch_rate` = cluster-label changes per second in time order.

Existing signal kept: `tex_lap_ratio` (face vs scene Laplacian variance).

Controls (reported, never used for scoring): `p_ctl_bg_noise` = mean log n(G), `p_ctl_bg_hf` = mean log HF(G). The swapped clips went through
**one extra video encode** (swap output re-encoded), which changes noise and high-frequency content of the *whole* frame. If these
face-independent controls differ between real and swapped, absolute features are confounded by that encode and only the
relative (face-vs-surround) features are interpretable.

## Analysis protocol (fixed in advance)

- Unit: the clip (24). Cross-validation: **leave-one-session-out** (3 folds): thresholds/direction/weights for a fold are fitted
  only on the other two sessions' clips (8 real + 8 swapped) with the existing rule (`unmask/scoring/tune.py`: centre = midpoint
  of class medians, scale from robust spread, weight = clip(d′, 0.2, 2)), and applied to the held-out session's 8 clips. Clips of a
  session never appear in both training and testing.
- Clip score = weighted mean of the feature sigmoid scores; a clip is **flagged** if score ≥ 0.5. Missing features are skipped.
- Feature sets: **T** = {tex_lap_ratio}; **B** = baseline passive = {seam_edge, seam_flicker, seam_edge_cv, tex_lap_ratio,
  tex_hf_ratio, landmark_jitter, det_score_std} computed as before; **N** = all new `p_*` features except controls;
  **C** = B ∪ N (the primary set, because it is what would be wired into the scorer). Also each feature alone.
- Metrics: detection (swapped clips flagged), false-alarm rate (real clips flagged), balanced accuracy, with 95 % Wilson intervals
  (clip level, n = 12 per class; optimistic because clips of one session are correlated), and per-session balanced accuracy.
- Per-feature separation: for each of the 12 recordings, d = swapped − real (same recording, same session); report median d,
  the number of pairs with the hypothesised sign (k/12), the number of sessions whose median d has that sign (k/3), pooled AUC
  (informational only: it is confounded by session offsets), and the single-feature leave-one-session-out balanced accuracy.
  No multiple-comparison correction is applied (there are 13 candidate features; with 3 sessions any single "significant"
  feature is weak evidence).

## Success criterion (fixed in advance)

Passive detection is said to **improve** only if, for the primary set **C**: (1) leave-one-session-out balanced accuracy ≥ 0.75,
AND (2) at least 2 of the 3 held-out sessions have per-session balanced accuracy ≥ 0.75, AND (3) C beats the baseline set B by
≥ 0.10 balanced accuracy. Otherwise the result is "no improvement". If it improves: wire the new signals into the scorer and
the agent's uncertainty logic, deploy, and check the live swapped sample. If not: revert the live demo to the v0 priors.
Either way the docs report the result and state that this is one participant, three sessions.

## Amendments

Implementation fixes made **before** any result was computed (crash/index bugs found by the unit tests, not definition changes):
`frame_masks` was called with a 3-channel shape; the FFT frequency grid failed for odd patch sizes (now `fftfreq`, same
definition: normalised radius = frequency / Nyquist); the noise level has a numerical floor of 0.05 (avoids log 0).
Also: the temporal-correlation feature uses all adjacent eligible frame pairs (not only the stride-2 sample).

**A1 (post-hoc, declared after seeing the pre-registered results).** The pre-registered result is in
[results/passive_experiment.md](results/passive_experiment.md): passive detection did **not** improve. It also showed that the
neck patch `N` (centre y + 1.18·h) lies outside the frame in 18 of 24 clips, so all features that compare the face with the neck
were measured on only 4–6 clips and their hypotheses were barely tested. Variant A1 changes **one thing**: if `N` does not fit,
it is slid up to the lowest position that fits inside the frame, but not above y0 ≥ y + 0.9·h (otherwise the neck features stay
missing). Everything else, including the protocol and the success criterion, is identical. A1 is **exploratory**: it was chosen
after looking at the data, so it can never count as confirmation and **does not trigger deployment**; only the pre-registered
run decides that. A1 results are reported next to the pre-registered ones and labelled as post-hoc.

---

# Experiment 2: within-clip swap dropout during hand_pass and head_turn (pre-registration)

**Written and committed before any Experiment-2 feature was computed.** Hard stop 16:45 Lagos on 2026-10-26. Same 24 recordings
(only the 12 `hand_pass` + `head_turn` clips are used: 6 real, 6 swapped; 3 sessions × 2 challenges), one participant, no new data.
Run **once**; nothing is tuned after seeing the result. Privacy rule unchanged.

**Hypothesis.** During a hand pass or head turn the swapper drops out and the original face shows through, so *within a single swapped
clip* the face appearance jumps between a smooth swapped look and a sharper original look, while a real clip stays stable. Because every
quantity is measured **against the same clip's own pre-challenge frames**, between-session offsets (lighting, scene) cancel, which is what
sank the absolute-threshold features in §6.1 and Experiment 1.

**Frames.** *All* frames with a face detection (score ≥ 0.6, the pipeline's "present"), **including partial occlusion and large yaw**; no
yaw or occlusion filter. Frames whose face patch does not fit in the frame are skipped for the descriptor only.
**Windows (fixed):** pre-challenge = frames with t < 1.0 s from clip start; challenge = frames with t ≥ 1.0 s. A clip yields features only
if it has ≥ 5 detected frames in the pre window and ≥ 10 in the challenge window. *Known caveat, stated in advance:* the measured response
onset in these recordings is ~0.6–0.7 s (median `response_latency_s`), so the "pre" window can contain the start of the motion, which makes
the test conservative; the window is not adjusted.

**Per-frame quantities.** `tex_t` = log((Laplacian variance inside the face mask + 1) / (Laplacian variance of the background mask + 1)) of the
grey frame, the same definition as the existing `tex_lap_ratio` but kept per frame (face mask = central ellipse 0.35·w × 0.35·h, background =
outside 0.9·w × 1.0·h and not below the neck, as in `regions.frame_masks`). Descriptor `d_t` = [L, a, b medians of the central face region
(Experiment-1 definition), `tex_t`, log HF power of the face patch F (Experiment-1 definition)].

**Features (clip level, all defined here):**
- `d_tex_change` = median(`tex_t`, challenge window) − median(`tex_t`, pre window). Hypothesised sign of (swapped − real): **+** (the original,
  sharper face shows through).
- `d_tex_maxjump` = max over consecutive detected frames in the challenge window of |`tex_t` − `tex_{t−1}`|. Hypothesised sign: **+**.
- `d_out_frac` = fraction of challenge-window frames whose appearance lies outside the pre-challenge appearance: z-score of `d_t` against the
  pre-window mean and standard deviation per dimension (standard deviation floored at 0.25 × the dimension's standard deviation over all
  detected frames of the clip); a frame is outside if sqrt(mean of squared z) > 3.0. Hypothesised sign: **+**.
No k-means is used here (the z-score option is the pre-registered one).

**Analysis (fixed).**
- Per pair (same recording, 6 pairs): d = swapped − real; report each d, the sign counts k/6 with exact (Clopper–Pearson) 95 % intervals,
  the median d with a percentile-bootstrap 95 % interval over the 6 pairs (clustered by session in practice: only 3 sessions), and the
  per-session mean of the two pairs (sign per session).
- Leave-one-session-out, clip level (12 clips, 4 per held-out session): thresholds/direction/weights fitted on the other two sessions with the
  existing rule (`tune.fit_config`), score = weighted mean of sigmoid scores of the three features (set **D**), flagged if ≥ 0.5; also each
  feature alone. Report detection/false alarms with 95 % Wilson intervals (n = 6 per class) and per-session balanced accuracy.

**Success criterion (all required; "separates in all 3 sessions").**
1. at least one of the three features has the hypothesised sign in the session-mean difference of **all 3 sessions** and in ≥ 5 of the 6 pairs;
2. set D leave-one-session-out balanced accuracy ≥ 0.75, with per-session balanced accuracy ≥ 0.75 in **all 3** held-out sessions.
(With three features tried, passing both by chance is not negligible; this is stated up front and in the report.)

**Consequences.** If met: add the signal to the scorer config and the agent's uncertainty logic, redeploy (`make remote-deploy`), run `make smoke`
on both samples and check whether the live swapped sample is caught. If not met: **change nothing in the deployment**. Report and docs either way.
Features are research-only (switched off in Lambda by default) unless the criterion is met.

## Amendments (Experiment 2)

- **Floating-point bug in the success check (found after the single run).** Set D's balanced accuracy is exactly 0.75 ((4/6 + 5/6)/2) but the
  comparison `>= 0.75` evaluated to False (0.7499999…). Fixed with a 1e-9 tolerance and the report regenerated from the cached features
  (deterministic, nothing re-measured). Effect on the outcome: criterion 2a is met, but criterion 2b (all three held-out sessions ≥ 0.75) is
  not (room light 0.50) and neither is the overall criterion, so the verdict (**not met**) is unchanged. No feature definition, window,
  threshold or analysis step was changed.
