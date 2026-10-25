# Demo video script (≤ 5:00)

Target length 4:30, leaving 30 s of slack. Record at 1080p. Screen + voice-over; face cam only in
the intro. Times are cumulative. **[Bracketed]** = what's on screen. Fill `«…»` from
`docs/results/results.md` after the final eval run, and say only numbers that are in it.

---

### 0:00 – 0:20 · Team intro
**[Face cam, then title card "Unmask: is the face on your video call real?"]**

> Hi, I'm «name» «and this is «teammate»». We built Unmask for the OpenCV AI Competition: an agent
> that helps you check, during a live video call, whether the caller is a real-time face swap.

### 0:20 – 0:55 · Problem
**[B-roll: news headlines about deepfake video-call scams (your own screenshots, cite sources on
screen); then a phone ringing with a "family member" on video]**

> Real-time face swaps now run on a gaming laptop. Scammers use them on WhatsApp and Zoom to
> pose as family members, partners or bosses, then ask for money. Detecting a swap from a
> single frame is hard. But a live swap has a weakness: it struggles when the real person
> *moves*. Turn your head to profile, pass a hand in front of your face, or change the light,
> and the fake starts to break. Unmask turns that into a guided check anyone can run.

### 0:55 – 2:30 · Live demo catching a swap
**[Split screen: left a WhatsApp Web / Meet call with a consenting teammate using an AI-generated
face (state on screen: "swap uses an AI-generated face of no real person"); right the Unmask tab]**

1. **[Click "Share the call window", pick the call window, drag a box over the caller's tile]**
   > I open Unmask next to the call and share the call window. I box the caller's video.
2. **[Passive scan runs: 10 s countdown]**
   > First, a ten-second passive scan. OpenCV 5 finds the face with the YuNet detector, tracks five
   > landmarks, estimates head pose with solvePnP, and measures the seam around the face, its texture,
   > and how stable it is from frame to frame.
3. **[Prompt appears: "Say to the caller: Please turn your head slowly all the way to the left…" plus
   the "Why:" line]**
   > The seam score is borderline, so the agent asks for a head turn, and it says why. I read it out.
4. **[Caller turns; 5 s recording; result]**
   > At full profile, the swap breaks up: landmarks jump and detection flickers.
5. **[Next prompt: hand pass]**
   > Now the agent chooses a hand pass, because occlusion is the most uncertain signal left.
6. **[Verdict card: HIGH_RISK / SUSPICIOUS, evidence frames with face box, seam band, red motion overlay]**
   > Verdict: «HIGH RISK». The evidence frame shows the face being drawn *over* the hand. The advice is
   > plain: don't send money, call them back on a number you already know.
7. **[Quick cut: same flow with the teammate's real face → LOW_RISK]**
   > Same check on the real person: low risk, in «N» challenges.

### 2:30 – 3:15 · The agent and its trace
**[Scroll the "What the agent saw, chose and why" timeline; then open AGENT_TRACE_EXAMPLE.json in an
editor and highlight signals_after → decision across two steps]**

> This is the agentic part. After every clip, OpenCV's measurements become per-signal scores with an
> uncertainty. The agent, Claude on Amazon Bedrock with tool use, picks one action:
> issue a challenge, ask for better conditions, or conclude. Here you can see the vision output
> change the decision: after the head turn, seam and pose became decisive, so the next call switched to
> hand_pass. If Bedrock fails or answers something invalid, a rule-based policy with the same logic
> takes over, and that's logged too. The verdict itself never comes from the LLM. It comes from a
> deterministic, versioned scorer. And the user stays in control: skip any challenge, stop any time,
> delete the data.

### 3:15 – 3:45 · Architecture
**[docs/diagrams/architecture.svg, then the CloudWatch dashboard]**

> Everything runs on AWS from one SAM template: a static site on S3 and CloudFront; an HTTP API;
> a Lambda container with OpenCV 5 and the YuNet model baked in; clips in a private, encrypted S3
> bucket that deletes everything after a day; sessions and traces in DynamoDB with a 24-hour TTL.
> A five-second clip analyses in about «X» seconds on Lambda. The dashboard tracks latency, verdicts,
> the inconclusive rate and Bedrock fallbacks.

### 3:45 – 4:25 · Results
**[docs/results/ablation.png, then the per-challenge table]**

> We evaluated on «N» consenting participants, real and swapped, including clips recorded through real
> calls, with thresholds fitted leave-one-participant-out. Passive analysis alone caught «a»% of swaps.
> A fixed challenge sequence caught «b»%. Letting the agent choose caught «c»% with «d» challenges on
> average, at a «e»% false-positive rate on real callers.

### 4:25 – 4:55 · Limitations & responsible use
**[docs/results failure-case frame; RESPONSIBLE_USE bullets]**

> It isn't perfect. «Describe the main failure case, e.g. dim light on darker skin pushes clips to
> INCONCLUSIVE.» Our participant pool is small, so treat it as a risk assessment, not proof. Unmask never
> identifies anyone, keeps no face database, deletes clips in 24 hours, and says "inconclusive" when it
> can't tell. Try it yourself: the link and the sample clips are on our Devpost page. Thanks!

---

**Recording checklist**
- Use only consenting participants; the swap source face is AI-generated (show the on-screen note).
- Do not show the face-swap tool's UI or name.
- Hide notifications and personal chats; use a test WhatsApp/Meet account.
- Check the final cut is ≤ 5:00.
