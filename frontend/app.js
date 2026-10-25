/* Unmask frontend: vanilla JS, no build step. */
(() => {
  "use strict";
  const CFG = window.UNMASK_CONFIG || {};
  const API = (new URLSearchParams(location.search).get("api") || CFG.apiUrl || "/api").replace(/\/$/, "");
  const $ = (id) => document.getElementById(id);
  const CHALLENGE_NAMES = { passive: "Passive scan", head_turn: "Head turn", hand_pass: "Hand pass", light_change: "Light change" };
  const SIGNAL_NAMES = { seam: "Face-edge seam", texture: "Texture / sharpness", temporal: "Frame-to-frame stability",
    pose: "Head-turn consistency", occlusion: "Hand occlusion", lighting: "Lighting consistency", latency: "Response latency" };
  const VERDICT_TEXT = { LOW_RISK: "Low risk", SUSPICIOUS: "Suspicious", HIGH_RISK: "High risk", INCONCLUSIVE: "Inconclusive" };
  const PASSIVE_S = 10, CHALLENGE_S = 5;
  let session = null;

  // ------------------------------------------------------------------ API
  async function api(method, path, body) {
    // Safe calls (reads, creating a session) are retried once on a cold start / gateway timeout.
    // Clip analysis is never retried, so a clip can't be counted twice.
    const retryable = method === "GET" || path === "/sessions";
    let r;
    for (let attempt = 0; ; attempt++) {
      try {
        r = await fetch(API + path, {
          method, headers: body ? { "content-type": "application/json" } : {},
          body: body ? JSON.stringify(body) : undefined,
        });
        if (!retryable || attempt > 0 || ![502, 503, 504].includes(r.status)) break;
      } catch (e) {
        if (!retryable || attempt > 0) throw e;
      }
      await new Promise((res) => setTimeout(res, 1500));
    }
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    return data;
  }

  async function uploadAndAnalyse(sid, challenge, blob, onStatus) {
    const ctype = (blob.type || "video/webm").split(";")[0];
    onStatus(`Uploading ${CHALLENGE_NAMES[challenge].toLowerCase()} clip (${(blob.size / 1e6).toFixed(1)} MB)…`);
    const u = await api("POST", `/sessions/${sid}/upload-url`, { challenge, content_type: ctype });
    const form = new FormData();
    Object.entries(u.upload.fields).forEach(([k, v]) => form.append(k, v));
    form.append("file", blob, `${challenge}.${ctype.split("/")[1]}`);
    const up = await fetch(u.upload.url, { method: "POST", body: form });
    if (!up.ok) throw new Error(`upload failed (HTTP ${up.status})`);
    onStatus("Analysing with OpenCV and choosing the next step…");
    return api("POST", `/sessions/${sid}/clips`, { key: u.key, challenge });
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
  document.querySelector(".modes").addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      const next = $("tab-live").getAttribute("aria-selected") === "true" ? "upload" : "live";
      selectTab(next); $(`tab-${next}`).focus();
    }
  });

  // ------------------------------------------------------------------ live mode
  let stream = null, crop = null, recording = false;
  const video = $("preview");
  const canShare = !!(navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia) && typeof MediaRecorder !== "undefined";
  $("share-support").textContent = canShare
    ? "Works best in desktop Chrome or Edge. In Chrome you can share a single tab, a window, or the whole screen."
    : "This browser can't share a window. Use desktop Chrome or Edge, or switch to 'Analyse a recording'.";
  $("btn-share").disabled = !canShare;

  $("btn-share").onclick = async () => {
    try {
      stream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 15 }, audio: false });
    } catch (e) { setLive(`Sharing was cancelled (${e.name}).`, true); return; }
    video.srcObject = stream; await video.play();
    stream.getVideoTracks()[0].addEventListener("ended", stopShare);
    $("live-area").hidden = false; $("btn-stop-share").hidden = false; $("btn-share").hidden = true;
    if (!session || session.status !== "active") {
      session = await api("POST", "/sessions", { mode: "live" }).catch((e) => { setLive(e.message, true); return null; });
    }
    if (session) renderPrompt(session.next_action, session);
    $("btn-record").focus();
  };
  function stopShare() {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null; video.srcObject = null;
    $("btn-stop-share").hidden = true; $("btn-share").hidden = false;
  }
  $("btn-stop-share").onclick = stopShare;

  // crop selection on the preview (caller's tile only)
  (() => {
    const box = $("preview-box"), rect = $("crop-rect");
    let start = null;
    box.addEventListener("pointerdown", (e) => { const b = box.getBoundingClientRect(); start = [e.clientX - b.left, e.clientY - b.top]; box.setPointerCapture(e.pointerId); });
    box.addEventListener("pointermove", (e) => {
      if (!start) return;
      const b = box.getBoundingClientRect(), x = e.clientX - b.left, y = e.clientY - b.top;
      Object.assign(rect.style, { left: Math.min(x, start[0]) + "px", top: Math.min(y, start[1]) + "px",
        width: Math.abs(x - start[0]) + "px", height: Math.abs(y - start[1]) + "px" });
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

  function recordClip(seconds) {
    // Draw the (cropped) shared video onto a canvas and record that, so only the caller's tile is uploaded.
    const vw = video.videoWidth, vh = video.videoHeight;
    const sx = crop ? crop.x * vw : 0, sy = crop ? crop.y * vh : 0, sw = crop ? crop.w * vw : vw, sh = crop ? crop.h * vh : vh;
    const scale = Math.min(1, 960 / sw);
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(sw * scale / 2) * 2; canvas.height = Math.round(sh * scale / 2) * 2;
    const ctx = canvas.getContext("2d");
    const mime = ["video/webm;codecs=vp8", "video/webm;codecs=vp9", "video/webm", "video/mp4"].find((m) => MediaRecorder.isTypeSupported(m));
    const rec = new MediaRecorder(canvas.captureStream(15), { mimeType: mime, videoBitsPerSecond: 2_500_000 });
    const chunks = [];
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    let timer;
    const draw = () => { ctx.drawImage(video, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height); };
    return new Promise((resolve) => {
      rec.onstop = () => { clearInterval(timer); resolve(new Blob(chunks, { type: mime.split(";")[0] })); };
      draw(); timer = setInterval(draw, 1000 / 15);
      rec.start(250);
      let left = seconds;
      $("countdown").textContent = `● ${left}s`;
      const tick = setInterval(() => {
        left -= 1;
        $("countdown").textContent = left > 0 ? `● ${left}s` : "";
        if (left <= 0) { clearInterval(tick); rec.stop(); }
      }, 1000);
    });
  }

  async function countdown(n) {
    for (let i = n; i > 0; i--) { $("countdown").textContent = String(i); await new Promise((r) => setTimeout(r, 800)); }
    $("countdown").textContent = "";
  }

  function setLive(msg, err) { $("live-status").textContent = msg || ""; $("live-status").classList.toggle("error", !!err); }

  function renderPrompt(na, s) {
    const n = (s.clips || []).length + 1;
    if (na.action === "conclude") { return; }
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
    recording = true; setBusy(true);
    try {
      await countdown(ch === "passive" ? 1 : 3);
      setLive("Recording…");
      const blob = await recordClip(ch === "passive" ? PASSIVE_S : CHALLENGE_S);
      const r = await uploadAndAnalyse(session.session_id, ch, blob, setLive);
      session = r;
      const c = r.clip;
      setLive(c.usable ? `${CHALLENGE_NAMES[ch]}: analysed (${Math.round(c.timing_ms.total)} ms).` : `${CHALLENGE_NAMES[ch]}: couldn't use this clip (${c.reason.replaceAll("_", " ")}).`, !c.usable);
      if (r.status === "concluded") showResults(r); else renderPrompt(r.next_action, r);
    } catch (e) { setLive(`Something went wrong: ${e.message}. You can try again.`, true); }
    finally { recording = false; setBusy(false); $("countdown").textContent = ""; }
  };
  $("btn-skip").onclick = async () => {
    if (!session) return;
    setBusy(true);
    try {
      session = await api("POST", `/sessions/${session.session_id}/skip`, { challenge: session.next_action.challenge || "passive" });
      if (session.status === "concluded") showResults(session); else renderPrompt(session.next_action, session);
    } catch (e) { setLive(e.message, true); } finally { setBusy(false); }
  };
  $("btn-finish").onclick = async () => {
    if (!session) return;
    setBusy(true);
    try { session = await api("POST", `/sessions/${session.session_id}/finish`); showResults(session); }
    catch (e) { setLive(e.message, true); } finally { setBusy(false); }
  };
  function setBusy(b) { for (const id of ["btn-record", "btn-skip", "btn-finish"]) $(id).disabled = b; }

  // ------------------------------------------------------------------ upload mode
  function setUp(msg, err) { $("upload-status").textContent = msg || ""; $("upload-status").classList.toggle("error", !!err); }

  async function runUploadSession(files) {
    // files: {challenge: Blob}
    $("results").hidden = true;
    const avail = Object.keys(files).filter((c) => c !== "passive");
    let s = await api("POST", "/sessions", { mode: "upload", available_challenges: avail.length ? avail : ["head_turn", "hand_pass", "light_change"] });
    const used = new Set();
    for (let guard = 0; guard < 12 && s.status === "active"; guard++) {
      const na = s.next_action, ch = na.challenge || "passive";
      if ((na.action === "record_passive" || na.action === "issue_challenge") && files[ch] && !used.has(ch)) {
        used.add(ch);
        if (na.action === "issue_challenge") setUp(`Agent asked for: ${CHALLENGE_NAMES[ch]}. ${na.reason || ""}`);
        s = await uploadAndAnalyse(s.session_id, ch, files[ch], setUp);
      } else {
        // No (better) clip for what the agent wants: tell it so, and let it pick something else.
        setUp(`No ${CHALLENGE_NAMES[ch].toLowerCase()} clip available; letting the agent choose again…`);
        s = await api("POST", `/sessions/${s.session_id}/skip`, { challenge: ch });
      }
    }
    if (s.status === "active") s = await api("POST", `/sessions/${s.session_id}/finish`);
    session = s; setUp("Done."); showResults(s);
  }

  $("upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const files = {};
    document.querySelectorAll("#upload-form input[type=file]").forEach((i) => { if (i.files[0]) files[i.dataset.ch] = i.files[0]; });
    if (!Object.keys(files).length) { setUp("Choose at least one video file.", true); return; }
    try { await runUploadSession(files); } catch (err) { setUp(`Something went wrong: ${err.message}`, true); }
  });

  async function loadSamples() {
    const box = $("sample-buttons");
    try {
      const m = await (await fetch("samples/manifest.json")).json();
      box.innerHTML = "";
      for (const sm of m.samples) {
        const b = document.createElement("button");
        b.className = "primary"; b.textContent = sm.title; b.title = sm.description || "";
        b.onclick = async () => {
          box.querySelectorAll("button").forEach((x) => (x.disabled = true));
          try {
            setUp(`Loading sample “${sm.title}”…`);
            const files = {};
            for (const [ch, url] of Object.entries(sm.clips)) files[ch] = await (await fetch(url)).blob();
            await runUploadSession(files);
          } catch (err) { setUp(`Sample failed: ${err.message}`, true); }
          finally { box.querySelectorAll("button").forEach((x) => (x.disabled = false)); }
        };
        box.appendChild(b);
      }
      if (m.note) { const n = document.createElement("span"); n.className = "muted"; n.textContent = m.note; box.appendChild(n); }
    } catch { box.innerHTML = '<span class="muted">No samples available.</span>'; }
  }
  loadSamples();

  // ------------------------------------------------------------------ results
  function el(tag, attrs = {}, ...kids) {
    const e = document.createElement(tag);
    Object.entries(attrs).forEach(([k, v]) => (k === "class" ? (e.className = v) : e.setAttribute(k, v)));
    kids.flat().forEach((k) => e.append(k instanceof Node ? k : document.createTextNode(String(k))));
    return e;
  }

  function describeEvent(ev) {
    if (ev.type === "clip_analysed") {
      const f = ev.features || {};
      const bits = [];
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

  function showResults(s) {
    const r = s.result || {};
    const v = r.verdict || "INCONCLUSIVE";
    $("results").hidden = false;
    $("verdict").className = `verdict ${v}`;
    $("verdict-label").textContent = VERDICT_TEXT[v];
    $("meter-fill").style.width = `${Math.round((r.risk ?? 0) * 100)}%`;
    $("verdict-risk").textContent = r.risk == null ? "Not enough usable video to score." :
      `Risk score ${Math.round(r.risk * 100)} / 100 · challenges completed: ${(r.challenges_done || []).map((c) => CHALLENGE_NAMES[c]).join(", ") || "none"}`;
    $("verdict-advice").textContent = r.advice || "";
    $("verdict-reasons").replaceChildren(...(r.top_reasons || []).map((t) => el("li", {}, t)));

    const ev = $("evidence"); ev.replaceChildren();
    (s.clips || []).forEach((c) => (c.evidence || []).forEach((e) =>
      ev.append(el("figure", {}, el("img", { src: e.url, alt: `${CHALLENGE_NAMES[c.challenge]}: ${e.label}`, loading: "lazy" }),
        el("figcaption", {}, `${CHALLENGE_NAMES[c.challenge]} · ${e.label}`)))));
    if (!ev.children.length) ev.append(el("p", { class: "muted" }, "No evidence frames."));

    const tb = $("signals"); tb.replaceChildren();
    Object.entries(r.signals || {}).forEach(([k, sg]) => {
      const score = sg.score == null ? null : Math.round(sg.score * 100);
      tb.append(el("tr", {}, el("th", { scope: "row" }, SIGNAL_NAMES[k] || k),
        el("td", {}, score == null ? "not measured" : [el("span", { class: "bar", style: `width:${score}px` }), `${score}`]),
        el("td", {}, sg.score == null ? "–" : `${Math.round((1 - sg.uncertainty) * 100)}%`)));
    });

    const tl = $("timeline"); tl.replaceChildren();
    (s.timeline || []).forEach((t) => {
      const chose = t.chose === "issue_challenge" ? `Chose: ${CHALLENGE_NAMES[t.challenge]} challenge` :
        t.chose === "request_better_conditions" ? "Chose: ask for better conditions" : "Chose: conclude";
      const pol = t.policy === "bedrock" ? `Bedrock · ${t.model_id || ""}` : t.policy === "rule_fallback" ? "Rule fallback" : t.policy;
      tl.append(el("li", {}, el("div", { class: "tl-saw" }, describeEvent(t.saw), t.risk != null ? ` · risk ${Math.round(t.risk * 100)}` : ""),
        el("div", { class: "tl-chose" }, chose, el("span", { class: `badge ${t.policy}` }, pol || "")),
        el("div", {}, `Why: ${t.why || ""}`)));
    });
    $("results").scrollIntoView({ behavior: "smooth" });
    $("verdict-label").setAttribute("tabindex", "-1"); $("verdict-label").focus({ preventScroll: true });
  }

  $("btn-trace").onclick = async () => {
    if (!session) return;
    const t = await api("GET", `/sessions/${session.session_id}/trace`);
    const a = el("a", { href: URL.createObjectURL(new Blob([JSON.stringify(t, null, 2)], { type: "application/json" })), download: `unmask-trace-${session.session_id}.json` });
    document.body.append(a); a.click(); a.remove();
  };
  $("btn-delete").onclick = async () => {
    if (!session || !confirm("Delete this session's clips, evidence frames and trace now?")) return;
    await api("DELETE", `/sessions/${session.session_id}`).catch(() => {});
    $("results").hidden = true; session = null;
    setUp("Deleted."); setLive("Deleted.");
  };
  $("btn-again").onclick = () => { session = null; $("results").hidden = true; window.scrollTo({ top: 0, behavior: "smooth" });
    if (stream) api("POST", "/sessions", { mode: "live" }).then((s) => { session = s; renderPrompt(s.next_action, s); }); };
  if (CFG.repoUrl) $("repo-link").href = CFG.repoUrl;
})();
