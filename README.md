# Unmask

**Check whether the face on your video call is a real-time face swap, before you trust it.**

Unmask runs in a browser tab beside your call (WhatsApp Web, Google Meet, Zoom, …). You share the
call window with it; it records short clips of the caller, and an agent tells you which physical
challenge to ask for next ("turn your head fully to the left", "pass a hand slowly across your
face", "move closer to a light"). OpenCV 5 measures each response, the agent picks the next
challenge based on what the measurements show, and a deterministic scorer gives a cautious verdict
(**LOW_RISK / SUSPICIOUS / HIGH_RISK / INCONCLUSIVE**) with evidence frames and plain advice
("Do not send money. Call them back on a number you already know.").

> Unmask gives a risk assessment, **not proof**. It never ends calls, contacts anyone, or
> identifies the caller. Clips are deleted within 24 hours. See [docs/RESPONSIBLE_USE.md](docs/RESPONSIBLE_USE.md).

Built for the OpenCV AI Competition 2026 (powered by AWS), targeting the **Agentic Vision** award:
OpenCV output decides the agent's next tool call.

| | |
|---|---|
| Live demo | **https://d2xv52l4vknsio.cloudfront.net** (eu-north-1) |
| Technical report | [docs/REPORT.md](docs/REPORT.md) |
| Architecture | [docs/architecture.md](docs/architecture.md) |
| Agent workflow | [docs/agent-workflow.md](docs/agent-workflow.md) |
| Example decision trace | [docs/AGENT_TRACE_EXAMPLE.json](docs/AGENT_TRACE_EXAMPLE.json) |
| Evaluation results | [docs/results/results.md](docs/results/results.md) (real recordings, leave-one-session-out) |

## Results at a glance (honest)

Evaluated on real recordings: **1 participant × 3 lighting sessions × 4 clips**, each also face-swapped with an
AI-generated face of no real person (24 clips, 2,520 frames per label; leave-one-session-out, the weaker check that is
all one participant allows). Full tables, per-session numbers, failure cases and caveats: [docs/REPORT.md](docs/REPORT.md) §6.

| Strategy | Swaps flagged | Real flagged (false alarms) | Avg challenges |
|---|---|---|---|
| Passive only | 2/3 | 2/3 | 0 |
| Fixed challenge sequence | 2/3 | 2/3 | 3 |
| Agent-chosen sequence | 2/3 | 2/3 | 2.67 |

**The detector does not separate real from swapped callers on this data** (in 2 of 3 sessions the real clip scored the
higher risk), and 3 sessions per class is far too few to conclude either way. What works end to end: the agentic loop
(OpenCV measurements change which challenge is asked next; the agent reached the same verdicts with fewer challenges),
the AWS deployment, observability and the 24 h deletion. Do not use the verdict as a real fraud check.

## Verified versions

| Component | Version |
|---|---|
| OpenCV | **5.0.0** (`opencv-python-headless==5.0.0.93`; `cv2.__version__ == "5.0.0"`, checked by `make setup`, the tests, the Docker build, and `/health`) |
| YuNet face detector | `face_detection_yunet_2023mar.onnx` from opencv_zoo, SHA-256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| Python | 3.12 (local and the `public.ecr.aws/lambda/python:3.12` base image) |
| Other pins | [requirements-lambda.txt](requirements-lambda.txt), [requirements-dev.txt](requirements-dev.txt) |

OpenCV 5 APIs used: `cv2.FaceDetectorYN` (YuNet), `cv2.solvePnP` / `cv2.projectPoints` /
`cv2.Rodrigues`, `cv2.calcOpticalFlowFarneback`, `cv2.createBackgroundSubtractorMOG2`,
`cv2.Laplacian`, `cv2.Sobel`, `cv2.cvtColor` (Lab), `cv2.warpAffine`, `cv2.VideoCapture`,
`cv2.imencode`, drawing functions for evidence frames.

## Quick start (local, no AWS needed)

```bash
make setup          # venv + pinned deps + YuNet model (checksum verified)
make test           # unit tests for every signal, scorer, agent, pipeline
make samples        # build the one-click demo clip sets
make serve          # http://localhost:8000: frontend + the same Lambda handler, local storage
```

Analyse clips from the command line:

```bash
.venv/bin/python -m unmask.cli analyze path/to/clip.webm --challenge head_turn --evidence-dir /tmp/ev
.venv/bin/python -m unmask.cli session data/swapped/p01 --trace /tmp/trace.json   # full agent replay
.venv/bin/python -m unmask.cli session data/swapped/p01 --bedrock                 # decisions via Bedrock
```

## Deploy to AWS

Prerequisites: AWS CLI v2 with credentials, [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html),
Docker (to build the Lambda container image), and access to Amazon Nova Lite in Amazon Bedrock.

```bash
make check-bedrock REGION=eu-north-1          # which Nova/Claude models and inference profiles you can call
sam build
sam deploy --guided                           # stack name: unmask; Region: eu-north-1 (your project Region)
make deploy-frontend STACK=unmask REGION=eu-north-1
make smoke STACK=unmask REGION=eu-north-1      # end-to-end test against the deployed API
```

**No Docker, or a slow connection?** `make remote-deploy` (after `git push`) deploys a small
`unmask-build` stack (ECR repository + CodeBuild project, `build.yaml`), builds the image inside AWS
from the pushed commit, and runs `sam deploy` with that image, so no local Docker and no large
upload. The live demo was deployed this way. Credentials from `aws login` need `awscrt`
(in `requirements-dev.txt`) for the Python tools.

`sam deploy` creates: CloudFront + private S3 frontend bucket, API Gateway HTTP API, the Lambda
container (3008 MB, 60 s), a private encrypted clip bucket with a 1-day lifecycle rule, a DynamoDB
table with 24 h TTL, least-privilege IAM, a CloudWatch dashboard and an error alarm. Outputs
include `FrontendUrl`, `ApiUrl` and `DashboardUrl`.

Bedrock: the agent only makes one small tool call per step, so the default model is the fast, cheap
**Amazon Nova Lite** through the EU cross-region inference profile, `eu.amazon.nova-lite-v1:0`, used
exactly as given (your project Region is eu-north-1). Change it with the `BedrockModelId` parameter
(e.g. `eu.amazon.nova-pro-v1:0`). Model ID, region and inference-profile handling are stack
parameters (`BedrockModelId`, `BedrockRegion`, `BedrockProfile`) passed to the function as
`BEDROCK_MODEL_ID` / `BEDROCK_REGION` / `BEDROCK_PROFILE`. An ID that already carries a profile
prefix (`eu.`, `us.`, `apac.`, `global.`) or an ARN is used unchanged; for a bare ID,
`BedrockProfile=auto` tries it as given, then the region's profile, then `global.`. The request is
shaped per model family (Nova: greedy decoding, no `additionalProperties` in tool schemas). If the model is not enabled, times out or
returns an invalid action, the agent falls back to the deterministic policy, logs the reason, and
increments the `BedrockFallbacks` metric. Set `UseBedrock=false` to run rules only.

## Evaluate

```bash
make eval-synth     # pipeline check on cartoon fixtures -> eval/out/synthetic-results/ (never published)
make eval           # your recordings in data/ (see docs/DATA_COLLECTION.md) -> docs/results/
.venv/bin/python eval/run_eval.py --data data --write-config --export-trace   # refit thresholds
```

The harness reports per-challenge detection and false-positive rates, inconclusive rate, average
challenges to verdict, per-clip latency, and an ablation of passive-only vs fixed sequence vs
agent-chosen sequence (offline replay of the same clips), with leave-one-participant-out
thresholds (leave-one-session-out when there is only one participant, which is a weaker check; the results
say so). Failure cases are collected with their evidence frames, for `divine_*` sessions only: other people's
sessions feed aggregate numbers but never appear in public outputs.

## API

| Method | Path | |
|---|---|---|
| GET | `/health` | OpenCV version, scoring config, Bedrock settings |
| POST | `/sessions` | `{mode: "live"\|"upload", available_challenges?}` → session + first action |
| POST | `/sessions/{id}/upload-url` | `{challenge, content_type}` → pre-signed POST for the clip |
| POST | `/sessions/{id}/clips` | `{key, challenge}` → analyse, score, run the agent, return next action |
| POST | `/sessions/{id}/skip` | user can't / won't do the requested challenge |
| POST | `/sessions/{id}/finish` | stop now and get the verdict so far |
| GET | `/sessions/{id}` | state, verdict, evidence URLs, timeline |
| GET | `/sessions/{id}/trace` | full decision trace (JSON) |
| DELETE | `/sessions/{id}` | delete clips, evidence and session immediately |

## Repository layout

```
unmask/vision/     OpenCV 5 signals, quality gates, evidence frames (no AWS deps)
unmask/scoring/    deterministic scorer, versioned configs, threshold fitting
unmask/agent/      rule policy + guardrails, Bedrock Converse tool use, session/trace engine
unmask/synth.py    cartoon fixtures used only by the unit tests and `make eval-synth`
backend/           Lambda handler, storage backends, local dev server, observability
frontend/          static web app (live window-share mode + upload mode)
eval/              evaluation harness
scripts/           deploy-frontend, smoke test, Bedrock check, samples, diagrams
docs/              report, diagrams, results, responsible use, data collection, video script, Devpost text
template.yaml      AWS SAM template      Dockerfile   Lambda container image
```

## Limitations

- **Not a validated detector.** Thresholds were fitted on one participant (three sessions) and showed no real/swapped
  separation out of sample; performance on other people, cameras, call apps and swap tools is unknown. See
  [docs/REPORT.md](docs/REPORT.md) §6 and §8 for the numbers, failure cases and what would fix it.
- Heavy call compression, low light, small video tiles and uncooperative callers reduce what can be
  measured; Unmask then says INCONCLUSIVE rather than guessing.
- Window sharing (`getDisplayMedia`) works in desktop Chrome and Edge (and recent Firefox); not on
  mobile browsers. Use upload mode there.
- A determined adversary with a better swap model can defeat individual signals.

## Responsible use

24 h deletion, no face database, no identification, consent-only test data, synthetic swap faces,
and an explicit INCONCLUSIVE outcome. Details and known bias limits: [docs/RESPONSIBLE_USE.md](docs/RESPONSIBLE_USE.md).
