# Demo video script (≤ 5:00)

Target length 4:30, leaving 30 s of slack. Record at 1080p. Screen + voice-over; face cam only in the intro.
Times are cumulative. **[Bracketed]** = what's on screen. Every number below is in `docs/REPORT.md` /
`docs/results/results.md`; **«name»** is the only thing to fill in. The story is deliberately honest: a working agentic
system on AWS, and a detector that does **not** yet separate real from swapped callers on our small data set.

---

### 0:00 – 0:20 · Team intro
**[Face cam, then title card "Unmask: is the face on your video call real?"]**

> Hi, I'm «name» «and this is «teammate»». We built Unmask for the OpenCV AI Competition: an agent that helps you
> check, during a video call, whether the caller might be a face swap. I'll show how it works, and I'll be straight
> about where it doesn't work yet.

### 0:20 – 0:55 · Problem
**[B-roll: news headlines about deepfake video-call scams (your own screenshots, cite sources on screen); then a phone
ringing with a "family member" on video]**

> Face-swap software now runs on ordinary laptops and plugs into WhatsApp or Zoom as a virtual camera. Scammers use it to
> pose as relatives or bosses and ask for money. One frame of a good swap is hard to judge. But a swap has to cope when the
> real person moves, so Unmask asks the caller to move: turn your head, pass a hand across your face, change the light.

### 0:55 – 2:15 · Demo on the live site
**[Browser at https://d2xv52l4vknsio.cloudfront.net; click "Analyse a recording"; a line on screen: "Samples are my own
recordings; the swap uses an AI-generated face of no real person."]**

1. **[Click "Real caller (recorded, daylight light)"; the agent session runs, ~30 s; jump-cut if needed]**
   > This is a real recording of me. Unmask records a passive scan, then the agent, Amazon Nova Lite on Bedrock, chooses
   > the next challenge and tells me why. Each clip goes through OpenCV 5 on Lambda: YuNet face detection, head pose
   > with solvePnP, seam and texture measures, optical flow for the hand pass, lighting correlation.
2. **[Verdict card: LOW RISK 32/100; scroll the timeline "saw → chose → why"; show an evidence frame]**
   > It ends at low risk, with evidence frames and its decision trail.
3. **[Click "Face-swapped caller (recorded, daylight light)"]**
   > Now the same recording with my face replaced by an AI-generated face of no real person.
4. **[Verdict card: LOW RISK 44/100]**
   > And here is the honest part: it also says low risk. **Unmask does not catch this swap.** I'll show the numbers in a minute.
5. **[Quick cut: "Check a live caller" → "Share the call window" → the large challenge prompt with countdown]**
   > In live mode you share the call window and read the challenges out to the caller. Same pipeline, same verdict logic.

### 2:15 – 3:00 · The agent and its trace
**[AGENT_TRACE_EXAMPLE.json in an editor; highlight `signals_after` and `decision` for steps 1 and 3]**

> This is the agentic part, and it's real. After each clip OpenCV's measurements become per-signal scores with an
> uncertainty. In this session the seam signal was already clear after the passive scan, so the agent asked for a hand pass
> first. In four of our six sessions the seam was still uncertain and it asked for a head turn first. After the head turn the
> pose score was decisive, so it skipped the lighting test and stopped. The verdict never comes from the language model: a
> deterministic scorer produces it, the model only chooses what to test next, and a rule-based policy takes over if
> Bedrock fails. The user can skip any challenge or stop at any time, and everything is deleted within 24 hours.

### 3:00 – 3:30 · Architecture
**[docs/diagrams/architecture.png, then the CloudWatch dashboard]**

> One SAM template: S3 and CloudFront for the site, an HTTP API, a Lambda container with OpenCV 5 and the YuNet model
> baked in, clips in a private bucket that deletes everything after a day, sessions and traces in DynamoDB with a 24-hour
> TTL. A warm clip analyses in 0.8 to 2.3 seconds on Lambda and each Nova Lite decision takes about 0.7 seconds. The first
> clip after a fresh deploy took 8.5 seconds, so we added a keep-warm ping.

### 3:30 – 4:25 · Results, honestly
**[docs/results/ablation.png, then the per-session table from REPORT.md §6.1]**

> We evaluated on 24 clips: me, in three lighting sessions, real and swapped, with thresholds cross-validated
> leave-one-session-out because there is only one participant. Passive-only, a fixed challenge sequence, and the agent
> all flagged 2 of 3 swapped sessions **and** 2 of 3 real ones. In two of the three sessions the real clip actually scored a
> *higher* risk than the swapped one. The agent reached the same verdicts as the fixed sequence with fewer challenges,
> 2.67 instead of 3, so it saves effort but not accuracy. Digging into the features, the only consistent direction was that
> the swapped face is smoother, in all three sessions, but the differences between lighting sessions are bigger than the
> difference between real and swapped. Three sessions can't settle it either way.

### 4:25 – 4:55 · Limitations, responsible use, what's next
**[The three failure-case frames from REPORT.md §6.2; RESPONSIBLE_USE bullets]**

> So today Unmask is a working agentic framework, not a validated detector, and it can both miss a swap and raise a false
> alarm on a real caller. The advice it gives is built for that: verify through a channel you trust. Next steps are more
> people and skin tones, compressed through-a-call recordings, more swap tools, and features that compare within a session
> instead of using absolute thresholds. Unmask never identifies anyone, keeps no face database and deletes clips in 24
> hours. The link and the code are on our Devpost page. Thanks!

---

**Recording checklist**
- Only the samples shipped on the site (the owner's own recordings); no one else's footage.
- Do not show the face-swap tool's name or interface; say "an AI-generated face of no real person".
- If a verdict on your take differs from the script, say what the screen shows.
- Hide notifications and personal chats; check the final cut is ≤ 5:00.
