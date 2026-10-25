# Responsible use

Unmask exists to protect people from video-call fraud (romance, "family emergency", CEO and
recruiter scams that use real-time face swaps). A tool in this space can also cause harm: by
giving false reassurance, by wrongly accusing a real person, or by becoming a surveillance
tool. These are the design choices we made, and the limits users should know about.

## What Unmask does and does not do

- **Advises only.** It never ends a call, never contacts anyone, never posts or reports anything.
  The person using it stays in control: they decide whether to relay a challenge, can skip any
  challenge, and can stop at any time ("Stop and show result").
- **No identification.** Unmask does not recognise faces and does not compare the caller with
  anyone. YuNet is a face *detector* (where is a face?), not a recogniser (whose face is it?).
  There is no face database, no embeddings, no biometric templates.
- **Risk assessment, not proof.** The four outcomes are LOW_RISK, SUSPICIOUS, HIGH_RISK and
  **INCONCLUSIVE**. When the video is too dark, small, blurry, choppy, or the caller did not
  perform the challenge, Unmask says so and asks for better conditions; after repeated failures it
  returns INCONCLUSIVE rather than guessing. Every verdict comes with plain advice that holds
  either way: verify through a channel you already trust, and never send money or codes on the
  strength of a video call alone.
- **The verdict is deterministic.** It is computed from OpenCV measurements by a versioned,
  inspectable scoring config. The LLM only chooses *which challenge to try next*; it cannot set or
  change the verdict, and its choices are checked against hard guardrails.

## Data handling

- Clips go straight from the browser to a **private, encrypted S3 bucket** through a short-lived
  pre-signed upload (10 min, size-limited, content-type-locked). Public access is blocked; HTTPS is
  enforced by bucket policy.
- **Everything is deleted within 24 hours**: an S3 lifecycle rule expires all clips and evidence
  frames after 1 day, and DynamoDB TTL expires session records and traces after 24 h. Users can
  delete their session immediately with "Delete my clips now" (`DELETE /sessions/{id}`).
- The decision trace stores numbers (signal scores, features, decisions), not video.
- Logs contain session IDs and measurements, not images or personal data. Log retention is 14 days.
- The local mode (`make serve`) applies the same 24 h purge to its local store.

## Evaluation data

- Only consenting participants (see `docs/DATA_COLLECTION.md`), pseudonymised as p01, p02, …
- Swapped clips use an **AI-generated face of no real person**, never a celebrity or acquaintance.
- No face-swap software, model weights, configs or source faces are in this repository.

## Known limitations and bias

- **Small participant pool.** Thresholds are tuned on a handful of participants with
  leave-one-participant-out validation; results will not transfer perfectly to other people,
  cameras, or call apps. See the numbers and failure cases in `docs/REPORT.md`.
- **Skin tone and lighting coverage.** Face detection confidence, seam contrast and texture
  measures all depend on skin tone, exposure and camera quality. Darker skin in dim light is a
  known weak spot for face detectors in general. Our coverage of skin tones and lighting is
  limited and is reported honestly in the report; treat results for under-represented
  conditions as less reliable. The quality gates (too dark / too small / too blurry) push such
  cases towards INCONCLUSIVE rather than towards an accusation.
- **Adversaries adapt.** A better face swap, a caller who refuses challenges, or heavy call
  compression can defeat individual signals. Unmask raises the cost of a convincing fake; it does
  not make one impossible.
- **False positives hurt real people.** A real caller with a poor camera may be flagged
  SUSPICIOUS. The advice for SUSPICIOUS is deliberately to *verify*, not to accuse.
- **Accessibility of challenges.** Some callers cannot turn their head or raise a hand (disability,
  injury). Users can skip any challenge; skipping is not itself treated as suspicious.

## Abuse considerations

- Unmask's outputs (signal scores) describe artefacts of a given video; they are not a recipe for
  building better swaps, and we do not publish swap tooling. The deterministic signals are
  standard, well-known image-forensics measurements.
- API throttling (10 req/s, burst 20) and short-lived upload URLs limit misuse of the hosted demo.
