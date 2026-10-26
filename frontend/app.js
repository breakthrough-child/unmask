/* Unmask frontend: vanilla JS, no build step. The API and scoring are unchanged; this file only presents them. */
(() => {
  "use strict";
  const CFG = window.UNMASK_CONFIG || {};
  const API = (new URLSearchParams(location.search).get("api") || CFG.apiUrl || "/api").replace(/\/$/, "");
  const $ = (id) => document.getElementById(id);
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const scrollTo = (node) => node.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  const CHALLENGE_NAMES = { passive: "Passive scan", head_turn: "Head turn", hand_pass: "Hand pass", light_change: "Light change" };
  const SIGNAL_NAMES = { seam: "Face-edge seam", texture: "Texture / sharpness", temporal: "Frame-to-frame stability",
    pose: "Head-turn consistency", occlusion: "Hand occlusion", lighting: "Lighting consistency", latency: "Response latency" };
  const SIGNAL_INFO = {
    seam: "Looks for a visible join where the face meets the hair and neck.",
    texture: "Compares how sharp the face is with the rest of the picture.",
    temporal: "Checks whether the face wobbles or flickers from frame to frame.",
    pose: "Checks how the face behaves while the head turns.",
    occlusion: "Checks what happens to the face when a hand passes in front of it.",
    lighting: "Checks whether the face's lighting follows changes in the room's light.",
    latency: "How quickly the person reacted to each request.",
  };
  const PLAIN = {   // plain-English reasons, no jargon and no numbers
    seam: { high: "The edge of the face looked unnatural in places.", low: "The face blended naturally into the hair and neck." },
    texture: { high: "The face looked smoother or blurrier than the rest of the picture.", low: "The face looked as sharp as the rest of the picture." },
    temporal: { high: "The face wobbled or flickered from moment to moment.", low: "The face stayed steady from moment to moment." },
    pose: { high: "The face broke up or jumped when the head turned.", low: "The face stayed consistent when the head turned." },
    occlusion: { high: "The face stayed visible where a hand should have covered it.", low: "A hand passing in front of the face hid it, as expected." },
    lighting: { high: "The lighting on the face didn't follow the room when the light changed.", low: "The lighting on the face followed the room's light." },
    latency: { high: "The person was slow to react to the requests.", low: "The person reacted to the requests promptly." },
  };
  const UNUSABLE = { too_dark: "The video was too dark.", no_face: "The caller's face wasn't visible enough.", face_too_small: "The caller's face was too small in the picture.",
    too_blurry: "The video was too blurry.", challenge_not_performed: "The requested action wasn't clear in the video.", too_few_frames: "The video was too short or choppy.", low_fps: "The video was too short or choppy." };
  const VERDICT = {
    LOW_RISK: { title: "No warning signs found", sub: "Unmask didn't spot signs of a face swap in this check. That doesn't prove the person is who they say they are.",
      icon: '<path d="M5 12.5l4.5 4.5L19 7.5" />' },
    SUSPICIOUS: { title: "Be careful", sub: "Some things looked off. This could be a face swap, or just a poor connection or camera.",
      icon: '<path d="M12 8v5M12 16.5v.5" /><path d="M10.3 4.3L2.6 18a2 2 0 001.7 3h15.4a2 2 0 001.7-3L13.7 4.3a2 2 0 00-3.4 0z" />' },
    HIGH_RISK: { title: "Warning: possible fake", sub: "Several signs pointed to a face swap. Treat this call as untrustworthy until you've verified the person another way.",
      icon: '<path d="M12 2l8 3v6c0 5-3.4 9-8 11-4.6-2-8-6-8-11V5l8-3z" /><path d="M9 9l6 6M15 9l-6 6" />' },
    INCONCLUSIVE: { title: "We couldn't tell", sub: "There wasn't enough clear video to judge. That isn't a sign either way.",
      icon: '<circle cx="12" cy="12" r="9" /><path d="M9.5 9.5a2.5 2.5 0 114 2c-.9.6-1.5 1.1-1.5 2.2M12 17.5v.5" />' },
  };
  const TODO = {
    common: ["Don't send money, codes or personal details because of this call.", "Hang up and call the person back on a number you already know.",
      "Ask a question only the real person could answer."],
    LOW_RISK: ["Still double-check anything unusual, especially requests for money or secrets."],
    SUSPICIOUS: ["Tell a family member or friend you trust what is going on."],
    HIGH_RISK: ["Tell a family member or friend you trust what is going on.", "If money or accounts are involved, contact your bank."],
    INCONCLUSIVE: ["Try again with a brighter, larger view of the caller, or verify another way."],
  };
  const PASSIVE_S = 10, CHALLENGE_S = 5;
  let session = null;

  // ------------------------------------------------------------------ small DOM helper
  function el(tag, attrs = {}, ...kids) {
    const e = document.createElement(tag);
    Object.entries(attrs).forEach(([k, v]) => (k === "class" ? (e.className = v) : e.setAttribute(k, v)));
    kids.flat().forEach((k) => e.append(k instanceof Node ? k : document.createTextNode(String(k))));
    return e;
  }

  // ------------------------------------------------------------------ progress panel
  const Progress = {
    root: $("progress"), list: $("progress-steps"), note: $("progress-slow"), title: $("progress-title"),
    start(title) { this.title.textContent = title; this.list.replaceChildren(); this.note.hidden = true; this.root.hidden = false; this.root.setAttribute("aria-busy", "true"); scrollTo(this.root); },
    step(label, state = "active") {
      const mark = el("span", { class: "mark", "aria-hidden": "true" });
      const txt = el("span", {}, label);
      const li = el("li", { class: state }, mark, txt);
      this.list.append(li); this.paint(li, state); return li;
    },
    set(li, state, label) { li.className = state; this.paint(li, state); if (label) li.lastChild.textContent = label; },
    paint(li, state) { li.firstChild.textContent = state === "done" ? "✓" : ""; li.setAttribute("aria-label", `${li.lastChild.textContent}, ${state === "done" ? "done" : state === "active" ? "in progress" : "waiting"}`); },
    slow(on) { this.note.hidden = !on; },
    hide() { this.note.hidden = true; this.root.setAttribute("aria-busy", "false"); this.root.hidden = true; },
  };
  /* If a request takes more than 5 s (cold start), say so. */
  async function withSlowNotice(fn) {
    const t = setTimeout(() => Progress.slow(true), 5000);
    try { return await fn(); } finally { clearTimeout(t); Progress.slow(false); }
  }
  function setBusy(b) {
    document.querySelectorAll("main button:not([role=tab]), main input").forEach((x) => { x.disabled = b; });
    document.body.classList.toggle("busy", b);
  }

  // ------------------------------------------------------------------ API
  async function api(method, path, body) {
    // Safe calls (reads, creating a session) are retried once on a cold start / gateway timeout.
    // Clip analysis is never retried, so a clip can't be counted twice.
    const retryable = method === "GET" || path === "/sessions";
    return withSlowNotice(async () => {
      let r;
      for (let attempt = 0; ; attempt++) {
        try {
          r = await fetch(API + path, { method, headers: body ? { "content-type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
          if (!retryable || attempt > 0 || ![502, 503, 504].includes(r.status)) break;
        } catch (e) { if (!retryable || attempt > 0) throw e; }
        await new Promise((res) => setTimeout(res, 1500));
      }
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
      return data;
    });
  }

  /* upload one clip, then analyse it (OpenCV + the agent's next choice happen in the same server call) */
  async function analyseClip(sid, challenge, blob, n, total) {
    const name = CHALLENGE_NAMES[challenge];
    const up = Progress.step(`Uploading clip ${n}${total ? ` of ${total}` : ""}: ${name}`);
    const ctype = (blob.type || "video/webm").split(";")[0];
    const u = await api("POST", `/sessions/${sid}/upload-url`, { challenge, content_type: ctype });
    const form = new FormData();
    Object.entries(u.upload.fields).forEach(([k, v]) => form.append(k, v));
    form.append("file", blob, `${challenge}.${ctype.split("/")[1]}`);
    const res = await withSlowNotice(() => fetch(u.upload.url, { method: "POST", body: form }));
    if (!res.ok) throw new Error(`upload failed (HTTP ${res.status})`);
    Progress.set(up, "done");
    const an = Progress.step("Analysing with OpenCV");
    const ag = Progress.step("Agent choosing next check", "pending");
    const r = await api("POST", `/sessions/${sid}/clips`, { key: u.key, challenge });
    Progress.set(an, "done");
    const c = r.step || {};
    Progress.set(ag, "done", c.chose === "conclude" ? "Agent decided it has enough" : c.chose === "request_better_conditions" ? "Agent asked for a better view"
      : `Agent chose the next check: ${CHALLENGE_NAMES[c.challenge] || "another check"}`);
    return r;
  }

  // ------------------------------------------------------------------ tabs
  function selectTab(which) {
    for (const t of ["live", "upload"]) {
      $(`tab-${t}`).setAttribute("aria-selected", String(t === which));
      $(`panel-${t}`).hidden = t !== which;
    }
  }
  $("tab-live").onclick = () => selectTab("live");
  $("tab-upload").onclick = () => selectTab("upload");
  $("btn-to-upload").onclick = () => selectTab("upload");
  document.querySelector(".modes").addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") { const next = $("tab-upload").getAttribute("aria-selected") === "true" ? "live" : "upload"; selectTab(next); $(`tab-${next}`).focus(); }
  });

  // ------------------------------------------------------------------ live mode (desktop Chrome / Edge only)
  const canShare = !!(navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia) && typeof MediaRecorder !== "undefined";
  const mobile = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent) || (matchMedia("(pointer: coarse)").matches && !matchMedia("(pointer: fine)").matches);
  const liveOK = canShare && !mobile;
  $("live-unavailable").hidden = liveOK;
  $("live-supported").hidden = !liveOK;
  if (!liveOK) {
    $("live-unavailable-text").textContent = mobile
      ? "Sharing a call window only works in desktop Chrome or Edge, not on phones or tablets. You can still try the samples or upload a recording."
      : "This browser can't share a window. Use desktop Chrome or Edge, or try the samples and upload a recording instead.";
  }
  let stream = null, crop = null, recording = false;
  const video = $("preview");
  $("share-support").textContent = "Works best in desktop Chrome or Edge. You can share a single tab, a window, or the whole screen.";

  function setLive(msg, err) { const n = $("live-status"); n.textContent = msg || ""; n.style.color = err ? "" : "var(--muted)"; }
  $("btn-share").onclick = async () => {
    try { stream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 15 }, audio: false }); }
    catch (e) { setLive(`Sharing was cancelled (${e.name}).`, true); return; }
    video.srcObject = stream; await video.play();
    stream.getVideoTracks()[0].addEventListener("ended", stopShare);
    $("live-area").hidden = false; $("btn-stop-share").hidden = false; $("btn-share").hidden = true;
    if (!session || session.status !== "active") session = await api("POST", "/sessions", { mode: "live" }).catch((e) => { setLive(e.message, true); return null; });
    if (session) renderPrompt(session.next_action, session);
    $("btn-record").focus();
  };
  function stopShare() { if (stream) stream.getTracks().forEach((t) => t.stop()); stream = null; video.srcObject = null; $("btn-stop-share").hidden = true; $("btn-share").hidden = false; }
  $("btn-stop-share").onclick = stopShare;

  (() => {   // optional crop to the caller's tile
    const box = $("preview-box"), rect = $("crop-rect");
    let start = null;
    box.addEventListener("pointerdown", (e) => { const b = box.getBoundingClientRect(); start = [e.clientX - b.left, e.clientY - b.top]; box.setPointerCapture(e.pointerId); });
    box.addEventListener("pointermove", (e) => {
      if (!start) return;
      const b = box.getBoundingClientRect(), x = e.clientX - b.left, y = e.clientY - b.top;
      Object.assign(rect.style, { left: Math.min(x, start[0]) + "px", top: Math.min(y, start[1]) + "px", width: Math.abs(x - start[0]) + "px", height: Math.abs(y - start[1]) + "px" });
      rect.hidden = false;
    });
    box.addEventListener("pointerup", () => {
      if (!start) return; start = null;
      const b = box.getBoundingClientRect(), r = rect.getBoundingClientRect();
      if (r.width < 40 || r.height < 40) { crop = null; rect.hidden = true; return; }
      crop = { x: (r.left - b.left) / b.width, y: (r.top - b.top) / b.height, w: r.width / b.width, h: r.height / b.height };
    });
    $("btn-clear-crop").onclick = () => { crop = null; rect.hidden = true; };
  })();

  function recordClip(seconds) {   // draw the (cropped) shared video onto a canvas and record that
    const vw = video.videoWidth, vh = video.videoHeight;
    const sx = crop ? crop.x * vw : 0, sy = crop ? crop.y * vh : 0, sw = crop ? crop.w * vw : vw, sh = crop ? crop.h * vh : vh;
    const scale = Math.min(1, 960 / sw);
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(sw * scale / 2) * 2; canvas.height = Math.round(sh * scale / 2) * 2;
    const ctx = canvas.getContext("2d");
    const mime = ["video/webm;codecs=vp8", "video/webm;codecs=vp9", "video/webm", "video/mp4"].find((m) => MediaRecorder.isTypeSupported(m));
    const rec = new MediaRecorder(canvas.captureStream(15), { mimeType: mime, videoBitsPerSecond: 2_500_000 });
    const chunks = []; rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    let timer;
    const draw = () => ctx.drawImage(video, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve) => {
      rec.onstop = () => { clearInterval(timer); resolve(new Blob(chunks, { type: mime.split(";")[0] })); };
      draw(); timer = setInterval(draw, 1000 / 15); rec.start(250);
      let left = seconds; $("countdown").textContent = `● ${left}s`;
      const tick = setInterval(() => { left -= 1; $("countdown").textContent = left > 0 ? `● ${left}s` : ""; if (left <= 0) { clearInterval(tick); rec.stop(); } }, 1000);
    });
  }
  async function countdown(n) { for (let i = n; i > 0; i--) { $("countdown").textContent = String(i); await new Promise((r) => setTimeout(r, 800)); } $("countdown").textContent = ""; }

  function renderPrompt(na, s) {
    const n = (s.clips || []).length + 1;
    if (na.action === "conclude") return;
    const ch = na.challenge || "passive";
    $("prompt-step").textContent = `Step ${n} · ${na.action === "request_better_conditions" ? "Fix the picture" : CHALLENGE_NAMES[ch]}`;
    $("prompt-text").textContent = na.action === "issue_challenge" ? `Say to the caller: “${na.prompt.replace(/^Ask them to /, "Please ").replace(/\.$/, "")}.”` : na.prompt;
    $("prompt-why").textContent = na.reason ? `Why: ${na.reason}` : "";
    $("btn-record").textContent = na.action === "issue_challenge" ? "I've asked them: start recording" : "Start recording";
    $("btn-skip").hidden = ch === "passive" && na.action === "record_passive";
  }
  $("btn-record").onclick = async () => {
    if (!session || recording || !stream) return;
    const na = session.next_action, ch = na.challenge || "passive";
    recording = true; setBusy(true); setLive("");
    try {
      await countdown(ch === "passive" ? 1 : 3);
      const blob = await recordClip(ch === "passive" ? PASSIVE_S : CHALLENGE_S);
      Progress.start("Checking the clip");
      const r = await analyseClip(session.session_id, ch, blob, (session.clips || []).length + 1, null);
      session = r;
      const c = r.clip;
      setLive(c.usable ? "" : `${CHALLENGE_NAMES[ch]}: couldn't use this clip (${c.reason.replaceAll("_", " ")}).`, !c.usable);
      if (r.status === "concluded") await finishAndShow(r); else { Progress.hide(); renderPrompt(r.next_action, r); }
    } catch (e) { Progress.hide(); setLive(`Something went wrong: ${e.message}. You can try again.`, true); }
    finally { recording = false; setBusy(false); $("countdown").textContent = ""; }
  };
  $("btn-skip").onclick = async () => {
    if (!session) return; setBusy(true);
    try { session = await api("POST", `/sessions/${session.session_id}/skip`, { challenge: session.next_action.challenge || "passive" });
      if (session.status === "concluded") showResults(session); else renderPrompt(session.next_action, session); }
    catch (e) { setLive(e.message, true); } finally { setBusy(false); }
  };
  $("btn-finish").onclick = async () => {
    if (!session) return; setBusy(true);
    try { session = await api("POST", `/sessions/${session.session_id}/finish`); showResults(session); }
    catch (e) { setLive(e.message, true); } finally { setBusy(false); }
  };
  async function finishAndShow(s) { const st = Progress.step("Preparing your result"); await new Promise((r) => setTimeout(r, 350)); Progress.set(st, "done"); Progress.hide(); showResults(s); }

  // ------------------------------------------------------------------ samples + upload
  function setUploadError(msg) { $("upload-error").textContent = msg || ""; }

  async function runSession(filesOrLoader, title) {
    hideResults(); setUploadError(""); setBusy(true);
    Progress.start(title);
    try {
      let files = filesOrLoader;
      if (typeof filesOrLoader === "function") {   // samples: fetch the clips first, as a visible step
        const ld = Progress.step("Loading the sample clips");
        files = await filesOrLoader();
        Progress.set(ld, "done");
      }
      const chs = Object.keys(files), total = chs.length;
      const st = Progress.step("Starting a private session");
      const avail = chs.filter((c) => c !== "passive");
      let s = await api("POST", "/sessions", { mode: "upload", available_challenges: avail.length ? avail : ["head_turn", "hand_pass", "light_change"] });
      Progress.set(st, "done");
      const used = new Set(); let n = 0;
      for (let guard = 0; guard < 12 && s.status === "active"; guard++) {
        const na = s.next_action, ch = na.challenge || "passive";
        if ((na.action === "record_passive" || na.action === "issue_challenge") && files[ch] && !used.has(ch)) {
          used.add(ch); n++;
          s = await analyseClip(s.session_id, ch, files[ch], n, total);
        } else {   // no (better) clip for what the agent wants: tell it, and let it pick something else
          const sk = Progress.step(`No ${CHALLENGE_NAMES[ch].toLowerCase()} clip provided: asking the agent for another check`);
          s = await api("POST", `/sessions/${s.session_id}/skip`, { challenge: ch });
          Progress.set(sk, "done");
        }
      }
      if (s.status === "active") s = await api("POST", `/sessions/${s.session_id}/finish`);
      session = s;
      await finishAndShow(s);
    } catch (err) { Progress.hide(); setUploadError(`Something went wrong: ${err.message}. Please try again.`); }
    finally { setBusy(false); }
  }

  $("upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const files = {};
    document.querySelectorAll("#upload-form input[type=file]").forEach((i) => { if (i.files[0]) files[i.dataset.ch] = i.files[0]; });
    if (!Object.keys(files).length) { setUploadError("Choose at least one video file."); return; }
    await runSession(files, "Checking your recording");
  });

  async function loadSamples() {
    const box = $("sample-buttons");
    try {
      const m = await (await fetch("samples/manifest.json")).json();
      box.replaceChildren();
      for (const sm of m.samples) {
        const isSwap = /swap/i.test(sm.id);
        const b = el("button", { class: "sample-btn", type: "button" },
          el("span", { class: "sample-ico", "aria-hidden": "true" }, isSwap ? "⇄" : "●"),
          el("span", {}, sm.title, el("small", {}, sm.description || "")));
        b.onclick = () => runSession(async () => {
          const files = {};
          for (const [ch, url] of Object.entries(sm.clips)) files[ch] = await (await fetch(url)).blob();
          return files;
        }, `Running: ${sm.title}`);
        box.append(b);
      }
      if (m.note) box.append(el("p", { class: "muted small" }, m.note));
    } catch { box.replaceChildren(el("span", { class: "muted" }, "No samples available.")); }
  }
  loadSamples();

  // ------------------------------------------------------------------ results
  function describeEvent(ev) {
    if (ev.type === "clip_analysed") {
      const f = ev.features || {}, bits = [];
      if (!ev.usable) bits.push(`unusable: ${String(ev.reason).replaceAll("_", " ")}`);
      if (f.seam_flicker != null) bits.push(`seam flicker ${f.seam_flicker.toFixed(2)}`);
      if (f.yaw_range != null) bits.push(`head turned ${Math.round(f.yaw_range)}°`);
      if (f.occ_persist != null) bits.push(`face visible through hand ${Math.round(f.occ_persist * 100)}%`);
      if (f.light_corr != null) bits.push(`light correlation ${f.light_corr.toFixed(2)}`);
      if (f.tex_lap_ratio != null) bits.push(`texture ratio ${f.tex_lap_ratio.toFixed(2)}`);
      return `Saw: ${CHALLENGE_NAMES[ev.challenge]} clip — ${bits.join(", ") || "measured"}`;
    }
    if (ev.type === "user_skipped") return `Saw: user skipped ${CHALLENGE_NAMES[ev.challenge] || ev.challenge}`;
    if (ev.type === "user_stopped") return "Saw: user stopped the check";
    return `Saw: ${ev.type}`;
  }

  function plainReasons(s) {
    const r = s.result || {}, v = r.verdict, sig = r.signals || {};
    const scored = Object.entries(sig).filter(([k, x]) => x.score != null && PLAIN[k]);
    let picks = [];
    if (v === "SUSPICIOUS" || v === "HIGH_RISK") picks = scored.filter(([, x]) => x.score >= 0.55).sort((a, b) => b[1].score - a[1].score).slice(0, 3).map(([k]) => PLAIN[k].high);
    else if (v === "LOW_RISK") picks = scored.filter(([, x]) => x.score <= 0.45).sort((a, b) => a[1].score - b[1].score).slice(0, 3).map(([k]) => PLAIN[k].low);
    else {
      const why = [...new Set((s.clips || []).filter((c) => !c.usable && UNUSABLE[c.reason]).map((c) => UNUSABLE[c.reason]))].slice(0, 3);
      picks = why;
    }
    if (!picks.length) picks = [v === "LOW_RISK" ? "None of the checks raised a strong warning." : v === "INCONCLUSIVE" ? "The checks gave mixed results." : "A few small things looked slightly off."];
    return picks;
  }

  function hideResults() { $("results").hidden = true; }

  function showResults(s) {
    const r = s.result || {}, v = r.verdict || "INCONCLUSIVE", vd = VERDICT[v];
    $("results").hidden = false;
    $("verdict").className = `verdict ${v}`;
    $("verdict-icon").innerHTML = `<svg viewBox="0 0 24 24" width="40" height="40" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">${vd.icon}</svg>`;
    $("verdict-title").textContent = vd.title;
    $("verdict-sub").textContent = vd.sub;
    const showMeter = v !== "INCONCLUSIVE" && r.risk != null;
    $("meter-wrap").hidden = !showMeter; $("meter-none").hidden = showMeter;
    if (showMeter) {
      $("meter-marker").style.left = `${Math.min(97, Math.max(3, r.risk * 100))}%`;
      $("meter").setAttribute("aria-label", `Risk level: ${r.risk < 0.4 ? "lower" : r.risk < 0.65 ? "medium" : "higher"}`);
    }
    $("reasons").replaceChildren(...plainReasons(s).map((t) => el("li", {}, t)));
    $("todo").style.setProperty("--vc", `var(--${{ LOW_RISK: "low", SUSPICIOUS: "sus", HIGH_RISK: "high", INCONCLUSIVE: "inc" }[v]})`);
    $("todo-list").replaceChildren(...[...TODO.common, ...TODO[v]].map((t) => el("li", {}, t)));

    // detailed view
    $("tech-summary").textContent = `Risk score ${r.risk == null ? "n/a" : Math.round(r.risk * 100) + " / 100"} · verdict ${v} · checks completed: ${(r.challenges_done || []).map((c) => CHALLENGE_NAMES[c]).join(", ") || "none"} · scoring config ${r.config_version || "?"}`;
    const ev = $("evidence"); ev.replaceChildren();
    (s.clips || []).forEach((c) => (c.evidence || []).forEach((e) => ev.append(el("figure", {}, el("img", { src: e.url, alt: `${CHALLENGE_NAMES[c.challenge]}: ${e.label}`, loading: "lazy" }), el("figcaption", {}, `${CHALLENGE_NAMES[c.challenge]} · ${e.label}`)))));
    if (!ev.children.length) ev.append(el("p", { class: "muted" }, "No evidence frames."));
    const tb = $("signals"); tb.replaceChildren();
    Object.entries(r.signals || {}).forEach(([k, sg]) => {
      const score = sg.score == null ? null : Math.round(sg.score * 100);
      tb.append(el("tr", {}, el("th", { scope: "row", title: SIGNAL_INFO[k] || "" }, SIGNAL_NAMES[k] || k, el("span", { class: "what" }, SIGNAL_INFO[k] || "")),
        el("td", {}, score == null ? "not measured" : [el("span", { class: "bar", style: `width:${score}px` }), `${score}`]),
        el("td", {}, sg.score == null ? "–" : `${Math.round((1 - sg.uncertainty) * 100)}%`)));
    });
    const tl = $("timeline"); tl.replaceChildren();
    (s.timeline || []).forEach((t) => {
      const chose = t.chose === "issue_challenge" ? `Chose: ${CHALLENGE_NAMES[t.challenge]} challenge` : t.chose === "request_better_conditions" ? "Chose: ask for better conditions" : "Chose: conclude";
      const pol = t.policy === "bedrock" ? `Bedrock · ${t.model_id || ""}` : t.policy === "rule_fallback" ? "Rule fallback" : t.policy;
      tl.append(el("li", {}, el("div", { class: "tl-saw" }, describeEvent(t.saw), t.risk != null ? ` · risk ${Math.round(t.risk * 100)}` : ""),
        el("div", { class: "tl-chose" }, chose, el("span", { class: `badge ${t.policy}` }, pol || "")), el("div", {}, `Why: ${t.why || ""}`)));
    });
    setDetails(false);
    scrollTo($("verdict")); $("verdict").focus({ preventScroll: true });
  }

  function setDetails(open) {
    $("detailed").hidden = !open;
    $("btn-details").setAttribute("aria-expanded", String(open));
    $("btn-details").textContent = open ? "Hide technical details" : "Show technical details";
  }
  $("btn-details").onclick = () => { const open = $("detailed").hidden; setDetails(open); if (open) scrollTo($("detailed")); };

  $("btn-trace").onclick = async () => {
    if (!session) return;
    const t = await api("GET", `/sessions/${session.session_id}/trace`);
    const a = el("a", { href: URL.createObjectURL(new Blob([JSON.stringify(t, null, 2)], { type: "application/json" })), download: `unmask-trace-${session.session_id}.json` });
    document.body.append(a); a.click(); a.remove();
  };
  $("btn-delete").onclick = async () => {
    if (!session || !confirm("Delete this session's clips, evidence frames and trace now?")) return;
    await api("DELETE", `/sessions/${session.session_id}`).catch(() => {});
    hideResults(); session = null; setUploadError("Deleted."); setLive("Deleted.");
  };
  $("btn-again").onclick = () => {
    session = null; hideResults(); window.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" });
    if (stream) api("POST", "/sessions", { mode: "live" }).then((s) => { session = s; renderPrompt(s.next_action, s); });
  };
  if (CFG.repoUrl) $("repo-link").href = CFG.repoUrl;
})();
