# Data collection guide

This is the protocol for recording the evaluation set used to tune Unmask's thresholds
and produce the numbers in `docs/REPORT.md`. Please read the safety rules first.

## Safety and consent rules (non-negotiable)

1. **Only consenting participants.** Every person whose real face appears gives informed,
   written consent to be recorded *for this evaluation*, knows their clips may be shown in the
   competition demo video, and can withdraw at any time (their folder is then deleted and the
   eval is re-run). Keep the signed consent forms outside this repository.
2. **Swapped clips use an AI-generated face of no real person.** Generate the "source" face with a
   synthetic-face generator and check it does not resemble anyone you know. **Never** a celebrity,
   a politician, a colleague, or any other real, identifiable person.
3. **No face-swap tooling, models, configs or source faces go into this repository**, its issues,
   or the Devpost submission. `data/` is git-ignored; keep it that way. Do not publish swapped
   clips except the 2–3 short samples you choose for the judges' one-click demo.
4. Swapped clips are only of participants who consented to being swapped.
5. No children. No recording of third parties in the background.

## Folder layout

```
data/
  labels.csv
  real/<participant>/passive.webm
  real/<participant>/head_turn.webm
  real/<participant>/hand_pass.webm
  real/<participant>/light_change.webm
  swapped/<participant>/passive.webm  ... (same four)
```

Any of `.webm .mp4 .mov .mkv` works. For extra conditions (see below) use a sub-folder per
condition, e.g. `real/p01/call_whatsapp/passive.webm`, and list it in `labels.csv`:

```csv
participant,label,challenge,path,condition,via_call
p01,real,passive,real/p01/passive.webm,default,no
p01,real,head_turn,real/p01/head_turn.webm,default,no
p01,swapped,hand_pass,swapped/p01/hand_pass.webm,default,no
p01,real,passive,real/p01/call_whatsapp/passive.webm,call_whatsapp,yes
```

`participant` is a pseudonym (`p01`, `p02`, …), never a name.

## What to record per participant

Record the **real** set and the **swapped** set under the same conditions (same room, same
camera, same lighting, same session), so the only difference is the swap.

| Clip | Duration | What the participant does |
|---|---|---|
| `passive` | 10 s | Looks at the camera, talks naturally, small natural head movement. No challenge. |
| `head_turn` | 5 s | Starts frontal; at ~1 s turns the head slowly **fully** to the left (profile, ~70–90°), then fully to the right, then back. |
| `hand_pass` | 5 s | Starts frontal; at ~1 s passes one open hand slowly across the face from one side to the other, fully covering it for ~1 s. |
| `light_change` | 5 s | Starts in normal light; at ~1 s moves clearly closer to a lamp/window, **or** a helper switches a side lamp on, **or** shines a phone torch on one side of the face. |

The clip starts when the instruction is given, so leave the ~1 s reaction delay in the clip:
Unmask measures response latency from clip start.

### Framing

- Head and shoulders visible, face roughly 1/4 to 1/3 of the frame width; the **neck must be
  visible** (the lighting check compares face vs neck).
- Camera at eye level, 50–80 cm away. Webcam or phone both fine; 720p or higher, ≥ 15 fps.
- Some background visible around the head (not a flat wall filling everything) helps the
  texture comparison, but also record at least one participant against a plain wall.

### Variety to cover (as much as time allows)

Aim for **at least 6 participants** (more is better; leave-one-participant-out needs ≥ 3 to mean anything).
With a single participant the harness uses leave-one-session-out instead and flags it as a limitation.
Name each session folder `<owner>_<condition>` (e.g. `divine_dim`); only folders starting with `divine_`
may ever appear in public outputs, everyone else's footage is aggregate-only.

- **Skin tones:** deliberately include a range (at least light, medium and dark skin tones).
  Report what you achieved in `docs/REPORT.md` — the bias section depends on it.
- **Lighting:** at least one participant each in (a) bright even light, (b) dim evening light,
  (c) strong side light / window behind.
- **Glasses / beard / hair over forehead:** include some.
- **Devices:** laptop webcam and phone.

### Through-a-real-call variants (important)

Real calls compress video heavily, which can hide (or fake) artefacts. For at least 2 participants,
re-record the four clips **through an actual video call** and capture them the way Unmask sees them:

1. Participant calls you on WhatsApp Web / Google Meet / Zoom (real face, then swapped face fed
   in as a virtual camera).
2. On your laptop, open Unmask's live mode or any screen recorder, share/record **only the call
   window**, and save each 5–10 s segment as above.
3. Put them under `condition` = `call_whatsapp` / `call_meet` / `call_zoom` and `via_call=yes`.

### Swapped variants

Use any real-time face-swap pipeline running locally on your own machine as a virtual camera, with
the AI-generated source face (rule 2). Record the swapped set with exactly the same actions. Keep
the tool's default settings (that is what a scammer would use) and note the tool's name and version
in a private notes file, **not** in this repository.

## After recording

```bash
make eval          # analyse every clip, LOPO thresholds, ablation, charts -> docs/results/
.venv/bin/python eval/run_eval.py --data data --write-config --export-trace
make test          # tests still pass with the new config
```

Then pick 2–3 sample sets for the judges (`make samples`, see `scripts/make_samples.py`), choosing
participants who consented to public use.
