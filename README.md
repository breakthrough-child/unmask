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
| Live demo | _see `FrontendUrl` after deploy (filled in docs/DEVPOST_SUBMISSION.md)_ |
| Technical report | [docs/REPORT.md](docs/REPORT.md) |
| Architecture | [docs/architecture.md](docs/architecture.md) |
| Agent workflow | [docs/agent-workflow.md](docs/agent-workflow.md) |
| Example decision trace | [docs/AGENT_TRACE_EXAMPLE.json](docs/AGENT_TRACE_EXAMPLE.json) |
| Evaluation results | [docs/results/results.md](docs/results/results.md) |

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
Docker (to build the Lambda container image), and access to a Claude model in Amazon Bedrock.

```bash
make check-bedrock REGION=us-east-1          # which Claude models / inference profiles you can call
sam build
sam deploy --guided                           # stack name: unmask; set BedrockModelId if needed
make deploy-frontend STACK=unmask REGION=us-east-1
make smoke STACK=unmask REGION=us-east-1      # end-to-end test against the deployed API
```

`sam deploy` creates: CloudFront + private S3 frontend bucket, API Gateway HTTP API, the Lambda
container (3008 MB, 60 s), a private encrypted clip bucket with a 1-day lifecycle rule, a DynamoDB
table with 24 h TTL, least-privilege IAM, a CloudWatch dashboard and an error alarm. Outputs
include `FrontendUrl`, `ApiUrl` and `DashboardUrl`.

Bedrock: the agent only makes one small tool call per step, so the default model is the fast, cheap
`anthropic.claude-haiku-5-5` (switch to e.g. `anthropic.claude-sonnet-5-5` with the `BedrockModelId`
parameter). Model ID, region and inference-profile handling are stack parameters (`BedrockModelId`,
`BedrockRegion`, `BedrockProfile`) passed to the function as `BEDROCK_MODEL_ID` / `BEDROCK_REGION` /
`BEDROCK_PROFILE`. With `BedrockProfile=auto` the agent tries the ID as given, then the region's
cross-region inference profile (`us.` / `eu.` / `apac.`), then `global.`, and caches whichever works;
a full profile ID or ARN is used unchanged. If the model is not enabled, times out or
returns an invalid action, the agent falls back to the deterministic policy, logs the reason, and
increments the `BedrockFallbacks` metric. Set `UseBedrock=false` to run rules only.

## Evaluate

```bash
make eval-synth     # pipeline check on generated fixtures -> eval/out/synthetic-results/
make eval           # your recordings in data/ (see docs/DATA_COLLECTION.md) -> docs/results/
.venv/bin/python eval/run_eval.py --data data --write-config --export-trace   # refit thresholds
```

The harness reports per-challenge detection and false-positive rates, inconclusive rate, average
challenges to verdict, per-clip latency, and an ablation of passive-only vs fixed sequence vs
agent-chosen sequence (offline replay of the same clips), with leave-one-participant-out
thresholds. Failure cases are collected with their evidence frames.

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
unmask/synth.py    synthetic fixtures (cartoons) for tests before real data exists
backend/           Lambda handler, storage backends, local dev server, observability
frontend/          static web app (live window-share mode + upload mode)
eval/              evaluation harness
scripts/           deploy-frontend, smoke test, Bedrock check, samples, diagrams
docs/              report, diagrams, results, responsible use, data collection, video script, Devpost text
template.yaml      AWS SAM template      Dockerfile   Lambda container image
```

## Limitations

- Thresholds are tuned on a small set of consenting participants; performance on other people,
  cameras and call apps will differ. See [docs/REPORT.md](docs/REPORT.md) for measured numbers and failure cases.
- Heavy call compression, low light, small video tiles and uncooperative callers reduce what can be
  measured; Unmask then says INCONCLUSIVE rather than guessing.
- Window sharing (`getDisplayMedia`) works in desktop Chrome and Edge (and recent Firefox); not on
  mobile browsers. Use upload mode there.
- A determined adversary with a better swap model can defeat individual signals.

## Responsible use

24 h deletion, no face database, no identification, consent-only test data, synthetic swap faces,
and an explicit INCONCLUSIVE outcome. Details and known bias limits: [docs/RESPONSIBLE_USE.md](docs/RESPONSIBLE_USE.md).
