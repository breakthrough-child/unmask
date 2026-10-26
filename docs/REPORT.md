# Unmask: technical report

**OpenCV AI Competition 2026 (powered by AWS) · Agentic Vision track**

> **Status of the numbers in this report.** Every number below was produced by the code in this
> repository from recorded footage. The headline result is **negative**: on the data collected so far
> (one participant, three lighting sessions, one swap pipeline) the scoring **does not separate real
> from swapped callers**: detection and false-positive rate are both 2 of 3 sessions (§6.1). The sample is far too
> small to say the approach cannot work, and it is too small to say it can. What the system does demonstrably do is run
> the full agentic loop end to end on AWS (§5, §6.3, §7). AWS latency figures were measured on the live deployment in
> eu-north-1 with `make smoke`.

## 1. Problem

Real-time face-swap software now runs on consumer GPUs and plugs into WhatsApp, Google Meet or Zoom
as a virtual camera. Fraudsters use it to pose as relatives in an emergency, romantic partners,
executives or recruiters, and then ask for money, codes or personal data. A single video frame of
a good swap is hard to tell apart from a real face, and the people targeted are often not experts.

A live swap does have a weakness: it is rendered frame by frame from a model trained mostly on
frontal, unoccluded, evenly lit faces. When the real person behind it **turns to profile, passes a
hand in front of the face, or changes the lighting**, the illusion tends to break: the blend seam
flickers, landmarks jump, the face is painted *over* the hand, and the face's brightness stops
following the scene. Unmask turns that observation into a guided, explainable check that anyone can
run during a call.

## 2. Users and use

The primary user is a person on a suspicious video call: a parent who gets a call from a "child"
asking for money, an employee called by the "CEO", someone in an online relationship. They open
Unmask in a browser tab beside the call, share the call window, read out the challenges Unmask
suggests, and get a cautious verdict with plain advice. Secondary users are fraud-prevention and
helpdesk staff, who can use upload mode on a recorded call.

Design constraints that follow from this:

- **No setup, no install:** a web page; window sharing via `getDisplayMedia` (desktop Chrome / Edge).
- **The human stays in control:** Unmask advises only. It never ends calls, contacts anyone, or
  identifies the caller. Every challenge can be skipped, and the check can be stopped at any time.
- **Honest uncertainty:** four outcomes (LOW_RISK / SUSPICIOUS / HIGH_RISK / **INCONCLUSIVE**), with
  advice that is safe whatever the verdict: verify through a channel you already trust.

## 3. Architecture

![architecture](diagrams/architecture.png)

Details: [architecture.md](architecture.md). In short:

| Layer | AWS service | Notes |
|---|---|---|
| Frontend | S3 (private) + CloudFront (OAC, HTTPS, security headers) | Vanilla JS, no build step |
| API | API Gateway HTTP API | CORS locked to the CloudFront origin; throttled to 10 req/s (burst 20) |
| Analysis + agent | Lambda **container image** (3008 MB, 60 s) | OpenCV 5.0 headless + YuNet baked in; the Docker build fails if `cv2.__version__` isn't 5.x or the model checksum differs |
| Clips + evidence | S3 (private, SSE, TLS-only, Block Public Access) | Pre-signed POST (10 min, ≤ 60 MB, content type locked); **lifecycle rule deletes everything after 1 day** |
| Sessions + traces | DynamoDB on-demand | Optimistic locking; **TTL 24 h** |
| Agent model | Amazon Bedrock, Converse API with tool use | Default Amazon Nova Lite (`eu.amazon.nova-lite-v1:0`), configurable |
| Observability | CloudWatch Logs (JSON) + EMF custom metrics + dashboard + alarm | Clip latency, verdicts, inconclusive rate, errors, Bedrock fallbacks |
| IaC | AWS SAM (`template.yaml`) | Least-privilege role, no secrets in code |

Video never passes through API Gateway: the browser uploads straight to S3, then calls
`POST /sessions/{id}/clips`, which runs the vision pipeline, scorer and agent in a single Lambda
invocation and returns the next action.

## 4. OpenCV 5 implementation

Verified version: **`opencv-python-headless==5.0.0.93`, `cv2.__version__ == "5.0.0"`** (checked by
`make setup`, the unit tests, the Docker build and the `/health` endpoint). Face detector: **YuNet**
`face_detection_yunet_2023mar.onnx`, SHA-256 `8f2383e4…52fa4`. All vision code is in
`unmask/vision/` (pure Python + OpenCV + NumPy, no AWS imports), so the same code runs locally,
in the eval harness and in Lambda.

**Pre-processing.** `cv2.VideoCapture` decodes the clip. Browser MediaRecorder WebM often has no
duration and a bogus FPS, so frames are timed with `CAP_PROP_POS_MSEC` when it advances (else by
FPS), resampled onto a 10 fps grid, and downscaled to ≤ 640 px wide with `INTER_AREA`.

| # | Signal | OpenCV 5 functions | What it measures |
|---|---|---|---|
| 1 | Face + landmarks | `cv2.FaceDetectorYN` | Per-frame box, 5 landmarks, confidence; single-face tracking; detection flicker per second; confidence spread |
| 2 | Head pose | `cv2.solvePnP` (EPnP → iterative), `cv2.Rodrigues`, `cv2.projectPoints` | Yaw trajectory from a generic 3D 5-point face model (blended with a geometric nose-offset proxy for robustness near profile); PnP reprojection error as a geometric-consistency check; behaviour at high yaw (confidence jerk, landmark jumps, seam gain) |
| 3 | Face-boundary seam | `cv2.warpAffine` (face-aligned crop), `cv2.Sobel`, `cv2.magnitude`, `cv2.cvtColor` (Lab) | Edge energy in an elliptical band at the face boundary vs. just inside and outside; Lab colour step across the band in 8 jaw/cheek sectors (median, robust to hair); frame-to-frame flicker in the band vs. the face interior |
| 4 | Occlusion (hand pass) | `cv2.calcOpticalFlowFarneback`, `cv2.createBackgroundSubtractorMOG2`, `cv2.morphologyEx` | Moving-pixel mask over the face; whether the face is still detected with high confidence while ≥ 20 % of it is covered; confidence drop; **texture leak** = correlation between the moving region and the pre-occlusion face (a swap paints the face over the hand); landmark jumps during occlusion |
| 5 | Lighting | `cv2.cvtColor` (Lab L), masks drawn with `cv2.ellipse` | Correlation of the face's luminance changes with the neck and the background; gain deviation; left/right asymmetry tracking (side light) |
| 6 | Texture / sharpness | `cv2.Laplacian`, NumPy FFT | Laplacian variance of the face vs. the background (log ratio); high-frequency spectral energy of the face vs. a same-scale scene patch |
| 7 | Response latency | (from 2, 4, 5) | Time from clip start (challenge issued) to the onset of the requested motion |
| 8 | Quality gates | `cv2.Laplacian`, Lab L | Too few frames, low FPS, no face, face too small (< 60 px), too dark, too blurry, **challenge not performed** (yaw range < 25°, occlusion < 25 %, light change < 6 L units) → unusable, with a specific instruction |

**Evidence frames.** Up to three annotated JPEGs per clip (`cv2.rectangle`, `cv2.ellipse2Poly` for
the seam band, `cv2.circle` for landmarks, red motion overlay, `cv2.imencode`): the strongest seam
frame, the maximum head turn, the hand-over-face moment, or the after-light frame.

**OpenCV 5 note (measured).** YuNet under OpenCV 5's new DNN graph engine took **63 ms/frame at
640×480 but 17 ms/frame at 320×240** on our laptop CPU (the engine logs that targets are not yet
supported by the new graph engine). `FaceDetectorYN` exposes no engine switch, so detection runs on
a 320 px copy and boxes and landmarks are scaled back. Farneback flow runs at 192 px. Together these
brought a 5 s clip from ~8 s to ~2–4 s locally (§6.3).

## 5. Agent design (Agentic Vision)

![agent workflow](diagrams/agent-workflow.png)

Details: [agent-workflow.md](agent-workflow.md).

**State the agent sees.** After each clip the deterministic scorer turns features into seven signal
scores (seam, texture, temporal, pose, occlusion, lighting, latency), each with an uncertainty
(`1 − |2s − 1|` scaled by feature coverage; unmeasured = 1), plus the session risk, the challenges
done, the last clip's usability and reason, and which signals each challenge resolves.

**Actions (tool use).** `issue_challenge(head_turn | hand_pass | light_change, reason)`,
`request_better_conditions(instruction, reason)`, `conclude(reason)`. The model must give a
one-sentence reason citing the signal values, which is stored in the trace and shown to the user.

**Policy.** The rule policy picks the untried challenge that maximises
`Σ signal_weight × uncertainty` over the signals it resolves (borderline seam/pose → head turn;
untested or odd lighting → light change; borderline texture/occlusion → hand pass). The Bedrock
agent receives the same state and principle. **Guardrails run first and can't be overridden:** an
unusable clip forces `request_better_conditions`; 2 unusable clips in a row or 3 in total, 4
challenges, or no challenges left force `conclude`. Every model tool call is validated (available?
not already done? within budget? has a reason?). Errors, timeouts or invalid calls fall back to the
rule policy, which is logged with the reason and counted in the `BedrockFallbacks` metric.

**The verdict never comes from the LLM.** `conclude` only stops the loop. The verdict, risk, top
reasons and advice are computed by `unmask/scoring` from a versioned JSON config.

**Model choice.** The decision is a single small tool call per step, so the default is the fast,
cheap **Amazon Nova Lite** via the EU cross-region inference profile (`eu.amazon.nova-lite-v1:0`,
used exactly as given; the stack runs in eu-north-1). It's switchable with one stack parameter. Requests
are shaped per model family: for Nova, greedy decoding (temperature 0, top-k 1) and tool schemas
without `additionalProperties`; Claude IDs still work. Tool calls are normalised (JSON-string input,
enum case) before validation, so a malformed answer becomes a logged fallback, not an error.
`toolChoice` is `auto`; the prompt asks for exactly one call and the validator enforces it.

**Proof that vision output changes decisions.** [AGENT_TRACE_EXAMPLE.json](AGENT_TRACE_EXAMPLE.json)
is an exported trace of a real recorded session (the swapped room-light session; the offline replay uses the
rule policy, Bedrock was not called):

| Step | What OpenCV reported (scores 0–1, uncertainty) | Agent decision | Why (from the trace) |
|---|---|---|---|
| 1 | Passive clip: seam 0.04 (uncertainty 0.25, clear), texture 0.73 (0.55), temporal 0.86 (0.27); risk 0.53 | `issue_challenge(hand_pass)` | Expected uncertainty resolved: hand pass 1.74 > head turn 1.64 > light change 1.54. Because the seam signal was already settled, the head turn (which mainly resolves seam/pose) was no longer the best first test |
| 2 | Hand-pass clip: occlusion 0.31 (0.61), risk 0.42 | `issue_challenge(head_turn)` | Pose unmeasured: a head turn now resolves 2.38 of uncertainty, far more than a light change (1.51) |
| 3 | Head-turn clip: pose 0.94 (uncertainty 0.11), risk 0.62 | `conclude` | Scores decisive enough; the light-change challenge was skipped |

The choice depended on what OpenCV measured. In the two room-light sessions the seam and temporal signals were already clear after the
passive scan (uncertainty 0.25–0.28), so the agent started with `hand_pass`; in the other four sessions the seam was still
uncertain (0.45–0.99) and it started with `head_turn` (`head_turn > hand_pass > light_change` ×4, `hand_pass > head_turn` ×2). The agent stopped after two
challenges in those two room-light sessions and used all three in the others. Whether these choices make the
*verdict* better is a separate question, and §6.1 shows they did not here.

## 6. Evaluation

Harness: `eval/run_eval.py`, run once on the real recordings; raw outputs in [results/](results/results.md).
Each clip is analysed once and cached. Thresholds are fitted with `unmask/scoring/tune.py` (feature centre = midpoint
of the class medians, scale from the robust spread, weight = clipped d′) in a cross-validated way so that every
scored session never contributed to its own thresholds. Leave-one-*participant*-out is the intended scheme, but **with a
single participant it is impossible, so the harness uses leave-one-*session*-out** (each lighting session is scored with
thresholds fitted on the other two), and the results file says so. The folds start from the hand-set v0 priors.
Three strategies are replayed offline on the same clips: (a) passive only, (b) a fixed sequence head turn → hand pass
→ light change, (c) the agent choosing challenges (rule policy; Bedrock is not involved in the replay).

### 6.1 Real recordings

**Data.** 24 clips = 1 participant (the project owner, with consent) × 3 lighting sessions (daylight, dim, room light) ×
4 clips (passive 10 s, head turn, hand pass, light change 6 s each) × {real, swapped}; 640×480, 30 fps laptop webcam,
2,520 frames per label. The swapped clips are the same recordings with the face replaced by an AI-generated face of no
real person, using one open-source face-swap pipeline at its default settings (tool name and version kept in private
notes, per [DATA_COLLECTION.md](DATA_COLLECTION.md)). Not covered: other people, other skin tones, other cameras,
through-a-real-call (compressed) variants, other swap tools. All 24 clips passed the quality gates.

| Strategy | Detection (swapped flagged SUSPICIOUS/HIGH_RISK) | High-risk on swapped | False positives (real flagged) | Inconclusive | Swaps rated LOW_RISK | Avg challenges |
|---|---|---|---|---|---|---|
| (a) Passive only | 2/3 (66.7 %) | 1/3 | 2/3 (66.7 %) | 2/6 sessions | 0/3 | 0 |
| (b) Fixed sequence | 2/3 (66.7 %) | 0/3 | 2/3 (66.7 %) | 0/6 | 1/3 | 3 |
| (c) Agent-chosen | 2/3 (66.7 %) | 0/3 | 2/3 (66.7 %) | 0/6 | 1/3 | 2.67 |

| Per-challenge (passive + that one challenge) | Detection | False positives | n real / n swapped |
|---|---|---|---|
| head_turn | 3/3 | 3/3 | 3 / 3 |
| hand_pass | 1/3 | 2/3 | 3 / 3 |
| light_change | 2/3 | 1/3 | 3 / 3 |

With 3 sessions per class one session moves a rate by 33 points, and the 95 % Wilson interval for 2 of 3 is roughly
21–94 %. Read these as "no evidence of separation", not as measured rates.

Per-session outcome (risk score 0–1 in brackets; thresholds from the other two sessions):

| Session | Passive only | Fixed sequence | Agent-chosen (challenges used) |
|---|---|---|---|
| real, daylight | SUSPICIOUS (0.59) | LOW_RISK (0.41) | LOW_RISK (0.41), 3 |
| **swapped**, daylight | HIGH_RISK (0.70) | LOW_RISK (0.44) ← missed | LOW_RISK (0.44) ← missed, 3 |
| real, dim | INCONCLUSIVE (0.24) | SUSPICIOUS (0.53) ← false alarm | SUSPICIOUS (0.53) ← false alarm, 3 |
| **swapped**, dim | INCONCLUSIVE (0.23) | SUSPICIOUS (0.46) | SUSPICIOUS (0.46), 3 |
| real, room light | SUSPICIOUS (0.47) | HIGH_RISK (0.72) ← false alarm | HIGH_RISK (0.75) ← false alarm, 2 |
| **swapped**, room light | SUSPICIOUS (0.53) | SUSPICIOUS (0.62) | SUSPICIOUS (0.62), 2 |

![ablation](results/ablation.png)

**What the data says, plainly**

1. **No separation.** In two of the three sessions the *real* clips received a higher risk score than the swapped
   clips (dim 0.53 vs 0.46; room light 0.75 vs 0.62); only in daylight was the swap marginally higher (0.44 vs 0.41).
   Detection and false-positive rate are equal, which is what a coin flip looks like.
2. **Challenges did not help accuracy.** Passive-only already flags the same 2 of 3 real and 2 of 3 swapped
   sessions (with two dim sessions INCONCLUSIVE); the challenges made verdicts more cautious (swaps went from
   HIGH_RISK to SUSPICIOUS or LOW_RISK) without separating the classes.
3. **The agent saved effort, not accuracy.** It reached the same verdict class as the fixed sequence in all six
   sessions using 2.67 instead of 3 challenges on average (it stopped after two in both room-light sessions). That is a
   real, if modest, benefit for the person doing the check, but it is not a detection improvement.
4. **Which signals carry anything?** Per-session feature values (real vs swapped; "swapped higher" = number of
   sessions out of 3 in which the swapped clip scored higher than the real one):

   | Feature | What it measures | Swapped higher in | Comment |
   |---|---|---|---|
   | `tex_lap_ratio` (passive) | face sharpness vs. scene | **0/3** (swapped *smoother* every time) | The only feature with a consistent direction (−0.98 vs −0.20, 1.04 vs 1.70, −0.55 vs −0.11). With n=3 the chance of this by luck is 1 in 4 (either direction); and the session-to-session spread is larger than the effect, so a fixed threshold cannot use it |
   | `turn_landmark_jump` (head turn) | landmark jumps during the turn | 2/3 | Real room-light turn (0.129) > swapped (0.106): session effects (lighting, how far the head turned, 80° vs 177°) dominate |
   | `turn_reproj_err_gain` | PnP reprojection error at high yaw | 1/3 | Inconsistent |
   | `seam_flicker`, `occ_texture_leak`, `occ_persist`, `light_corr`, `light_lr_corr` | seam flicker, face "leaking" over a hand, lighting following the scene | 1–2/3, tiny differences | No separation observed |

   The in-sample fit on all 24 clips (`scoring_v1.json`) reports a head-turn landmark jump d′ of 5.3.
   Given the per-session numbers above that is an **overfitting artefact of 3 vs 3 samples**, not evidence; v1 has
   no out-of-sample validation, so it is **not deployed** (the live demo uses the hand-set v0 priors, see §6.5).
5. **Why the original hypotheses may have failed here (hypotheses, not findings).** The swap is a 256-px model
   run offline on clean webcam footage, not a real-time swap on compressed call video. When a hand fully covers the
   face the face detector finds no face, so no swap is applied in those frames (the swapped clip differs less from the
   original in those frames, measured), which looks the same as a real face being hidden: the "face persists over the hand" tell-tale
   never occurred. Seam flicker was not visible at this quality. None of this says how weaker or real-time swaps, or
   heavily compressed call video, would behave.

### 6.2 Failure cases (agent strategy, all from the owner's own sessions)

| Case | Verdict | Why it failed |
|---|---|---|
| Swapped, daylight | **LOW_RISK (0.44), missed swap** | Only the texture-smoothness signal fired; the head turn scored as consistent and lighting as following the scene (top reasons in the results file) |
| Real, dim | **SUSPICIOUS (0.53), false alarm** | The head-turn signal ("broke up or jumped") and the response-latency signal fired on a real face |
| Real, room light | **HIGH_RISK (0.75), false alarm** | Head-turn, hand-occlusion ("leaked over the hand") and seam-flicker signals all scored high on a real face |

Evidence frames (annotated: face box, landmarks, seam band, motion overlay):

| Missed swap (daylight, head turn) | False alarm (dim, head turn) | False alarm (room light, hand pass) |
|---|---|---|
| ![](results/failures/p01_swapped_daylight_head_turn_0.jpg) | ![](results/failures/p01_real_dim_head_turn_0.jpg) | ![](results/failures/p01_real_roomlight_hand_pass_0.jpg) |

The false alarms matter as much as the miss: the advice for SUSPICIOUS is deliberately "verify through another
channel", never "accuse", but a tool that cries wolf on a real caller in warm light is not usable as is.

### 6.3 Latency

| Clip (frames after 10 fps resampling) | Local p50 | Local p95 | n |
|---|---|---|---|
| passive (10 s, 100 frames) | 4.77 s | 5.31 s | 6 |
| head_turn (6 s, 60 frames) | 2.53 s | 2.56 s | 6 |
| hand_pass (6 s, 60 frames) | 5.00 s | 5.52 s | 6 |
| light_change (6 s, 60 frames) | 3.98 s | 4.24 s | 6 |

Local = Ubuntu laptop CPU (8 threads), measured by the eval harness on the 12 real + 12 swapped recordings, while other
processes were also using the CPU. All are well inside API Gateway's 30 s integration limit. A MediaRecorder WebM
recorded in the browser decoded and analysed correctly through the local API.

**Lambda (eu-north-1, 3008 MB, x86_64), measured with `make smoke` on 2026-10-26** through the live API with the two
recorded demo samples (real and swapped daylight sessions, 8 clips, [lambda_latency.json](results/lambda_latency.json)):

| Clip | OpenCV analysis on Lambda |
|---|---|
| passive (100 frames) | 1.4 s warm; **8.5 s** on the first clip after a fresh deploy (cold container) |
| head_turn (60 frames) | 0.8 s |
| hand_pass (60 frames) | 1.4–2.3 s |
| light_change (60 frames) | 1.1–1.7 s |

Client round trips for the `/clips` call (analysis + S3 download + Nova decision + DynamoDB + network from Lagos, upload
excluded) were 3.5–11.2 s warm (median over all 8 clips 5.0 s); the single cold-start clip took **23.5 s**, uncomfortably close
to API Gateway's 30 s limit. The first invocation of a freshly deployed image previously exceeded Lambda's 10 s init
limit and timed out; an EventBridge keep-warm ping every 5 minutes and a one-time client retry for safe calls cover normal
use, but the first request after a deploy can still be slow.

**Agent on Bedrock (live).** Across four smoke sessions Amazon Nova Lite (`eu.amazon.nova-lite-v1:0`) made 12 decisions: 11
accepted and **1 rejected by the validator** (it asked for `hand_pass`, which was already completed, so the rule policy took
over and the fallback was logged; the other 4 steps were guardrail `conclude`s when no challenge was left). Latency, first two
sessions: median **0.72 s**, max 1.14 s per decision. Nova mostly chose `light_change` first, whereas the rule policy replay opens
with `head_turn` for daylight sessions; both are valid under the guardrails. Its stated reasons were generic one-liners ("to resolve
the most uncertain signals …") rather than quoting the signal values the prompt asks for. Whether Nova's order changes verdict
quality was not measured (the offline evaluation used the rule policy).

**Live demo check.** With the deployed hand-set v0 priors the real daylight sample ends **LOW_RISK (risk 0.285)** and the
swapped daylight sample **LOW_RISK (0.288)**: the deployed demo does **not** flag the swap, matching the missed-swap failure case
in §6.2. (An earlier deployment with the in-sample `scoring_v1` gave 0.32 and 0.44, also LOW_RISK for both; v1 was reverted because
it has no out-of-sample validation, §6.5.)

### 6.4 Failure handling, observability and human control

| Aspect | How it is evaluated |
|---|---|
| Task success | Ablation detection / FPR / inconclusive / challenges-to-verdict (§6.1–6.2) |
| Failure handling | Unit tests: unusable clip → `request_better_conditions` with the gate's reason; 2 unusable in a row → INCONCLUSIVE; ≤ 4 challenges; Bedrock timeout / invalid tool / unavailable challenge → rule fallback; guardrail overrides the model on unusable clips; bare model ID rejected → regional inference profile retried and cached (55 tests in `tests/`, incl. mocked Nova responses) |
| Observability | Each step's inputs, scores, action, reason, policy (bedrock / rule_fallback / guardrail / human), model ID and latency are stored in DynamoDB and downloadable as JSON. CloudWatch EMF metrics: `ClipLatencyMs`, `AnalysisLatencyMs`, `AgentLatencyMs`, `Verdicts`, `InconclusiveRate`, `BedrockFallbacks`, `Errors`, `ClipsAnalysed`; dashboard and error alarm defined in `template.yaml` |
| Human control | Skip any challenge (`/skip`), stop any time (`/finish`), delete everything now (`DELETE /sessions/{id}`); the agent cannot end calls or contact anyone |

### 6.5 Passive-signal experiment (pre-registered; result: no improvement)

**Question.** §6.1 showed passive detection at chance, with texture smoothness the only consistent direction. Can new passive
signals, measured on frontal, unoccluded frames of *every* clip (not just `passive.mp4`), do better? The features, protocol and
success criterion were written and committed **before** any new feature was computed
([PASSIVE_EXPERIMENT.md](PASSIVE_EXPERIMENT.md), commit `0474841`); results:
[results/passive_experiment.md](results/passive_experiment.md).

**New signals (OpenCV 5; `unmask/vision/passive.py`).** Resolution mismatch (Laplacian-pyramid fine-band energy and FFT
high-frequency power of a face patch vs. neck and vs. the hair/ear/background surround, at the same pixel scale); sensor-noise
mismatch (median-filter residual, 25th-percentile block noise level in face vs. neck vs. background, its variation over time and
the frame-to-frame residual correlation); skin-tone mismatch (Lab ΔE between central face and neck / forehead edge, plus
chroma-only); swap dropout (`cv2.kmeans`, k = 2, on per-frame Lab + texture descriptors: cluster separation, minor-cluster size,
switch rate); plus the existing texture signal. Eligible frames: face confidence ≥ 0.8, |yaw| ≤ 20°, no hand over the face.
Controls (face-independent background noise/HF) check whether the one extra video encode of the swapped clips is a confound.

**Protocol.** 24 clips (12 real, 12 swapped), one participant, **leave-one-session-out** (thresholds fitted on two sessions, applied
to the third; same fitting rule as the scorer), clip-level decision (flag if score ≥ 0.5), 95 % Wilson intervals (optimistic: clips of
one session are correlated). Feature sets: **T** texture only, **B** existing passive signals, **N** new only, **C** = B + N (primary).

| Set (pre-registered run) | Detection (swapped flagged) | False alarms (real flagged) | Balanced accuracy | per session (daylight / dim / room light) |
|---|---|---|---|---|
| T: texture only | 8/12 = 67 % [39–86] | 6/12 = 50 % [25–75] | 0.58 | 0.50 / 0.62 / 0.62 |
| B: existing passive signals | 6/12 = 50 % [25–75] | 6/12 = 50 % [25–75] | 0.50 | 0.50 / 0.50 / 0.50 |
| N: new signals only | 10/11 = 91 % [62–98] | 10/11 = 91 % [62–98] | 0.50 | 0.38 / 0.62 / 0.50 |
| **C: B + N (primary)** | 9/12 = 75 % [47–91] | 10/12 = 83 % [55–95] | **0.46** | 0.50 / 0.38 / 0.50 |

**Pre-registered criterion: not met.** It required (1) balanced accuracy ≥ 0.75 (got 0.46), (2) ≥ 2 of 3 held-out sessions at ≥ 0.75
(got 0 of 3) and (3) a gain of ≥ 0.10 over B (got −0.04). So nothing was wired into the scorer or the agent's uncertainty logic, and
the live demo was **reverted to the v0 priors**. (Set N flags 10 of 11 scored clips of *both* classes, i.e. it does not discriminate.)

**Per-feature separation** (paired: swapped − real of the same recording; 12 pairs from 3 sessions; "sessions" = how many of the 3 sessions
have the majority sign):

| Feature | Clips with feature | Pairs with majority sign | Sessions | Single-feature LOSO balanced accuracy |
|---|---|---|---|---|
| `tex_lap_ratio` (kept) | 24/24 | 12/12, swapped smoother | 3/3 | 0.58 |
| `tex_hf_ratio` (existing) | 24/24 | 11/12, swapped lower | 3/3 | 0.50 |
| `p_res_pyr_surround` (resolution vs. surround) | 22/24 | 7/11, swapped lower | 2/3 | 0.50 |
| `p_noise_face_vs_bg` (noise vs. background) | 22/24 | 8/11, swapped lower | 2/3 | 0.50 |
| `p_skin_de_neck` (ΔE face–neck) | 20/24 | 8/10, swapped **smaller** (opposite of the hypothesis) | 3/3 | 0.55 |
| `p_skin_de_neck_ab` / `p_skin_de_forehead_ab` | 20/24, 22/24 | 5/10, 6/11 | 1/3, 2/3 | 0.50 / 0.50 |
| `p_drop_sep`, `p_drop_minor_frac`, `p_drop_switch_rate` (dropout) | 14/24 | 4/7, 3/7, 1/7 | 2/3, 1/3, 0/3 | 0.36 / 0.43 / 0.57 |
| `p_res_*_neck`, `p_noise_face_vs_neck`, `p_noise_t*` | 4–6/24 | not testable | – | – |

**A design flaw this exposed.** The pre-registered neck patch lay outside the frame in 18 of 24 clips (the face fills the picture), so
every face-vs-neck comparison, the core of the resolution and noise hypotheses, was measured on only 4–6 clips and was effectively
untested. The face-independent controls show **no** encode confound (background noise Δ +0.004, background high-frequency Δ +0.025 log
units, real vs swapped), so the relative features are interpretable where they exist.

**Post-hoc variant A1 (exploratory; declared in the pre-registration file before it was run).** Sliding the neck patch up to fit the
frame (one change; everything else identical) raised neck-feature coverage to 20/24. It was chosen *after* seeing the data, so it can
never count as confirmation and could not trigger deployment. Result: the hypothesised direction held for resolution mismatch at the
neck in **3 of 3 sessions** (`p_res_pyr_neck` 9/10 pairs, `p_res_fft_neck` 8/10: relative to the same person's neck, the swapped face carries less fine
detail than the real face does) and, less clearly, for noise (`p_noise_face_vs_neck` 7/10, 2/3 sessions), **but classification stays at chance**: C 11/12
detection with 11/12 false alarms (balanced accuracy 0.50, 0/3 sessions ≥ 0.75), single neck features 0.50–0.55 balanced accuracy.
Full table: [results/passive_experiment_a1.md](results/passive_experiment_a1.md).

**Reading.** Within a recording the swap *is* measurably different in consistent directions (texture 12/12 pairs; resolution at the neck 9/10
pairs, post-hoc), which means "compare against a reference of the same person in the same lighting" would work. A single-clip detector
needs absolute thresholds, and the offsets between sessions (lighting, scene, how the person sits) are larger than the real/swapped
effect, so thresholds learned from two sessions do not transfer to the third. The 12 pairs are not 12 independent trials: they come from 3
sessions, so "3 of 3 sessions with the same sign" happens by chance 1 time in 4. One participant, three sessions and one swap tool mean
this neither proves the idea wrong nor right.

## 7. AWS deployment and reproducibility

```bash
make setup && make test                     # pinned deps, YuNet checksum, 55 tests
make check-bedrock REGION=eu-north-1         # which Bedrock models/profiles the account can call
sam build && sam deploy --guided             # whole stack
make deploy-frontend && make smoke           # upload site, end-to-end test, record Lambda latency
```

No local Docker? `make remote-deploy` builds the image in CodeBuild (from the pushed GitHub commit)
and deploys it; that is how the live stack was deployed over a slow connection.

Live endpoint: **https://d2xv52l4vknsio.cloudfront.net** (API `https://flqstgd5x1.execute-api.eu-north-1.amazonaws.com`, stack
`unmask` in eu-north-1). Verified live: S3 lifecycle rule `delete-everything-after-1-day` enabled,
Block Public Access on, DynamoDB TTL on `expires_at` enabled, CloudWatch dashboard `unmask-unmask`
receiving the custom metrics. `make serve` runs the exact Lambda handler locally against
file storage, so the app can be reviewed without an AWS account.

## 8. Limitations

- **The detector does not work on the data we have (§6.1).** On one participant and one swap pipeline, detection and
  false-positive rate are both 2/3 sessions, real clips scored *higher* risk than swapped ones in 2 of 3 sessions, and a
  follow-up with four new families of passive signals (§6.5) did not improve it. The deployed thresholds are the hand-set v0 priors; the
  fitted `scoring_v1.json` (in-sample, no out-of-sample validation) is not deployed. Do not rely on Unmask's verdict for a
  real call; as shipped it is a demonstration of the agentic architecture, not a validated deepfake detector.
- **Tiny, narrow sample.** 6 sessions of one person (three lighting conditions), one camera, one swap tool at default
  settings, clean webcam footage. No other people, no skin-tone coverage, no through-a-real-call (compressed) clips,
  no real-time swaps. Results will not transfer to other people, cameras, call apps or swap tools.
- **Single participant, leave-one-session-out.** Thresholds were cross-validated leave-one-*session*-out, not
  leave-one-*participant*-out. That is a weaker check: the sessions share the same face, camera and swap source, so
  it *overstates* how well thresholds would generalise to new people, and even so it showed no separation.
- **Replays used the rule policy.** The offline evaluation does not call Bedrock. Nova Lite's decisions were checked only
  in live smoke sessions (§6.3), not for their effect on verdict quality.
- **Compression:** call apps re-encode video heavily, which can hide seam and texture artefacts or create false ones.
  This was not measured.
- **Adversaries adapt:** better swap models, refusing challenges, or occluding deliberately can defeat individual signals.
- **Detector bias:** face detection and texture measures vary with skin tone, exposure and camera quality; the quality
  gates push poor conditions towards INCONCLUSIVE, but coverage is limited (one skin tone).
- **Browser support:** window sharing works in desktop Chrome/Edge (and recent Firefox), not on mobile; upload mode covers those.
- **OpenCV 5 DNN speed:** YuNet runs at reduced resolution for CPU latency, which halves landmark precision on small faces.

### What would make it better (not done)

More participants (≥ 6) and sessions so leave-one-participant-out is possible; compressed through-a-call recordings;
several swap tools and strengths; paired (within-session) features such as face-vs-scene sharpness rather than absolute
thresholds, since between-session variation was larger than the real/swapped effect.

## 9. Responsible use

See [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md): 24 h deletion (S3 lifecycle + DynamoDB TTL + delete
button), no face database, no recognition or identification, consent-only test data with
pseudonyms, AI-generated swap faces of no real person (never a celebrity), no face-swap tooling in
the repository, an explicit INCONCLUSIVE outcome, and advice that holds whatever the verdict.
