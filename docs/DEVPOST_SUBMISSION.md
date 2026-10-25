# Devpost submission text

_Paste each section into the matching Devpost field. Items marked **⏳ PENDING** still need a value
that doesn't exist yet (deployment, real-data evaluation, video). Fill them from the stack outputs
and `docs/results/results.md`; don't paste any number that isn't there._

**Project name:** Unmask

**Tagline:** An OpenCV 5 agent that catches real-time face swaps on video calls by asking the caller to move.

**Links:** Live demo ⏳ PENDING (`FrontendUrl` output after `sam deploy`) · Repo https://github.com/breakthrough-child/unmask · Video ⏳ PENDING (YouTube URL)

---

## Inspiration

Video calls used to be the way to check that someone was who they said they were. Not any more:
real-time face-swap software runs on a consumer GPU and plugs into WhatsApp, Meet or Zoom as a
virtual camera. Scammers use it to pose as a relative in an emergency, a romantic partner, or a
company executive, and then ask for money. The people targeted are often the least able to spot
a fake. We wanted something that a non-expert can use *during the call*, that tells them what to
do next, and that is honest when it can't tell.

## What it does

Unmask runs in a browser tab next to your call. You share the call window with it, and it:

1. records a 10-second **passive scan** of the caller and measures their face with OpenCV 5;
2. lets an **agent** (Amazon Nova Lite on Amazon Bedrock, via Converse tool use) pick the physical challenge that
   best resolves what the measurements are unsure about: *turn your head fully left*, *pass a hand
   slowly across your face*, or *move closer to a light*. It shows the challenge as a large prompt
   for you to read out, with a one-sentence reason;
3. records the caller's 5-second response, measures it, updates the scores, and repeats (at most
   four challenges, stopping early when the scores are decisive);
4. gives a cautious verdict (**LOW RISK / SUSPICIOUS / HIGH RISK / INCONCLUSIVE**) with annotated
   evidence frames, the top reasons in plain language, and advice such as "Don't send money. Call
   them back on a number you already know."

Bad video (too dark, face too small, too blurry, challenge not performed) gets a specific
instruction to fix it; repeated failure gives INCONCLUSIVE instead of a guess. There's also an
upload mode for recorded calls, with one-click sample clips for judges. The whole decision trail
(what OpenCV saw, what the agent chose, why, and whether Bedrock or the fallback decided) is shown
as a timeline and can be downloaded as JSON.

## How we built it

- **OpenCV 5.0** (`opencv-python-headless==5.0.0.93`) does all the vision: YuNet
  (`cv2.FaceDetectorYN`) for faces and 5 landmarks per frame; `solvePnP` + `projectPoints` against
  a generic 3D face for a yaw trajectory and geometric consistency; a face-boundary **seam** score
  (Sobel edge energy and Lab colour step in an elliptical band, plus its frame-to-frame flicker);
  **texture** mismatch (Laplacian variance and FFT high-frequency energy, face vs scene);
  **occlusion** analysis for the hand pass (Farneback optical flow + MOG2 background subtraction,
  checking whether the face "survives" a hand and whether face texture leaks over it);
  **lighting** consistency (Lab L on face vs neck/background over time); response latency; and
  quality gates.
- A **deterministic scorer** maps features to per-signal suspicion scores with thresholds fitted
  leave-one-participant-out, stored as versioned JSON. The verdict comes only from here.
- The **agent** gets the scores and uncertainties and makes one Bedrock Converse tool call:
  `issue_challenge`, `request_better_conditions` or `conclude`. Guardrails validate every call;
  a rule-based policy with the same logic takes over on errors, timeouts or invalid output.
- **AWS**: S3 + CloudFront frontend; API Gateway HTTP API; a Lambda container image (3008 MB) with
  OpenCV 5 and YuNet baked in; clips uploaded straight to a private, encrypted S3 bucket via
  pre-signed POST, deleted after 1 day; DynamoDB sessions and traces with a 24 h TTL; structured
  logs, EMF custom metrics, and a CloudWatch dashboard, all from one SAM template.
- **Frontend**: vanilla JS. `getDisplayMedia` → optional crop to the caller's tile → canvas →
  `MediaRecorder`.

## Challenges we ran into

- **Browser video is messy.** MediaRecorder WebM files often have no duration and a bogus frame
  rate, so we time frames from decoder timestamps and resample to ~10 fps ourselves.
- **Separating "the swap broke" from "the person did something hard".** A real face also loses
  detection confidence at profile and gets hidden by a hand. We had to measure *how* things break:
  smooth versus jumpy, hidden versus drawn-over.
- **Keeping the LLM in its lane.** We wanted real agentic choices without letting a language model
  decide whether someone is a fraudster. Splitting "what to test next" (agent) from "what the
  evidence says" (deterministic scorer), with guardrails and a fallback, solved it.
- **OpenCV 5's new DNN engine is slower on CPU.** YuNet took 63 ms per 640×480 frame but 17 ms at
  320×240, so we detect on a downscaled copy and scale the landmarks back. With 10 fps sampling and
  optical flow at 192 px, a 5 s clip analyses in about 2–4 s on a laptop CPU (p50 1.9–3.8 s),
  far inside API Gateway's 30 s limit. Lambda timing: ⏳ PENDING (`make smoke`).
- ⏳ PENDING: what surprised us once real recordings went through the pipeline.

## Accomplishments that we're proud of

- An agent loop where OpenCV output demonstrably changes the next tool call, with an exported
  trace that shows it.
- An honest evaluation: ablation of passive-only vs fixed sequence vs agent-chosen, leave-one-
  participant-out thresholds, failure cases with frames, and an explicit INCONCLUSIVE outcome.
- Responsible defaults: 24 h deletion, no face database, no identification, synthetic swap faces
  only, a delete-now button.

## What we learned

- Challenge-response is what makes a decision possible: in our synthetic pipeline check (cartoon fixtures, not real swaps), passive analysis
  alone left half the sessions inconclusive, while challenges let the scorer commit, and the agent
  got there with fewer challenges (2.5 vs 2.9) than a fixed sequence. Which challenge matters most
  on real swaps: ⏳ PENDING (per-challenge table from real recordings).
- Quality gates matter as much as detectors: a clip with no visible face has to turn into a
  specific instruction, not a verdict. Main source of errors on real calls: ⏳ PENDING (real data).
- Tool use is a great interface for constrained agent decisions: small action space, validated
  arguments, full trace.

## What's next

- A browser extension that captures the call tile automatically.
- More participants, devices and call apps; per-condition calibration and published bias metrics.
- Audio–lip-sync and voice-clone checks as additional challenges.
- On-device (WebAssembly OpenCV) passive scan for privacy, with the cloud agent only for challenges.

## Built with

`opencv` `opencv-5` `yunet` `python` `numpy` `aws-lambda` `amazon-bedrock` `amazon-nova` `amazon-api-gateway`
`amazon-s3` `amazon-cloudfront` `amazon-dynamodb` `amazon-cloudwatch` `aws-sam` `docker` `javascript`
`webrtc` `mediarecorder` `matplotlib` `pytest`
