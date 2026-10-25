"""AWS Lambda handler (API Gateway HTTP API, payload v2.0).

Routes
  GET    /health
  POST   /sessions                       {mode, available_challenges?}
  GET    /sessions/{id}
  DELETE /sessions/{id}                  delete clips, evidence and session now
  GET    /sessions/{id}/trace
  POST   /sessions/{id}/upload-url       {challenge, content_type}
  POST   /sessions/{id}/clips            {key, challenge}
  POST   /sessions/{id}/skip             {challenge}
  POST   /sessions/{id}/finish
"""
from __future__ import annotations

import json
import os
import re
import time
import traceback
import uuid

from unmask.agent.policy import CHALLENGE_TYPES
from unmask.agent.session import _run_decision, add_clip, finish, new_session, skip
from unmask.scoring import load_config
from unmask.vision import analyze_clip

from . import obs
from .storage import Conflict, from_env

USE_BEDROCK = os.environ.get("USE_BEDROCK", "true").lower() == "true"
MAX_UPLOAD = int(os.environ.get("MAX_UPLOAD_BYTES", str(60 * 1024 * 1024)))
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")

store = from_env()
CFG = load_config()

SID_RE = re.compile(r"^[0-9a-f]{32}$")
EXT = {"video/webm": "webm", "video/mp4": "mp4", "video/quicktime": "mov", "video/x-matroska": "mkv"}


class HttpError(Exception):
    def __init__(self, status, msg):
        super().__init__(msg)
        self.status, self.msg = status, msg


def resp(status, body):
    return {"statusCode": status, "headers": {"content-type": "application/json",
                                              "access-control-allow-origin": ALLOWED_ORIGIN,
                                              "cache-control": "no-store"},
            "body": json.dumps(body, default=str)}


# ---------------------------------------------------------------- storage
def load(sid: str) -> dict:
    if not SID_RE.match(sid or ""):
        raise HttpError(400, "bad session id")
    s = store.get_session(sid)
    if not s or int(s.get("expires_at", 0)) < time.time():
        raise HttpError(404, "session not found or expired")
    return s


def save(s: dict) -> None:
    try:
        store.put_session(s)
    except Conflict:
        raise HttpError(409, "session was updated concurrently; retry")


def presign_get(key: str) -> str:
    return store.presign_get(key)


def public_view(s: dict, include_trace: bool = False) -> dict:
    out = {k: s.get(k) for k in ("session_id", "mode", "status", "available_challenges", "next_action",
                                 "result", "provisional", "created_at", "expires_at")}
    out["clips"] = []
    for c in s["clips"]:
        cc = {k: c.get(k) for k in ("index", "challenge", "usable", "reason", "instruction", "features",
                                    "quality", "timing_ms")}
        cc["evidence"] = [{"label": e["label"], "url": presign_get(e["key"])} for e in c.get("evidence", [])]
        out["clips"].append(cc)
    out["timeline"] = [timeline_entry(st) for st in s["trace"]]
    if include_trace:
        out["trace"] = s["trace"]
    return out


def timeline_entry(st: dict) -> dict:
    ev, d = st["event"], st["decision"]
    return {"step": st["step"], "saw": ev, "risk": st.get("risk"), "provisional_verdict": st.get("provisional_verdict"),
            "chose": d["action"], "challenge": d.get("challenge"), "why": d.get("reason"),
            "policy": d.get("policy"), "model_id": d.get("model_id"), "latency_ms": d.get("latency_ms"),
            "fallback_reason": d.get("fallback_reason")}


# ---------------------------------------------------------------- routes
def create_session(body):
    mode = body.get("mode", "live")
    if mode not in ("live", "upload"):
        raise HttpError(400, "mode must be live or upload")
    avail = body.get("available_challenges") or list(CHALLENGE_TYPES)
    if not all(c in CHALLENGE_TYPES for c in avail):
        raise HttpError(400, f"available_challenges must be a subset of {CHALLENGE_TYPES}")
    s = new_session(mode=mode, available=avail)
    save(s)
    obs.log("session_created", session_id=s["session_id"], mode=mode)
    obs.metric("SessionsCreated", 1, "Count", Mode=mode)
    return 201, public_view(s)


def upload_url(s, body):
    ch = body.get("challenge")
    if ch not in ("passive",) + CHALLENGE_TYPES:
        raise HttpError(400, "unknown challenge")
    ctype = (body.get("content_type") or "video/webm").split(";")[0].strip()
    if ctype not in EXT:
        raise HttpError(415, f"content_type must be one of {sorted(EXT)}")
    key = f"clips/{s['session_id']}/{len(s['clips']):02d}_{ch}_{uuid.uuid4().hex[:8]}.{EXT[ctype]}"
    post = store.presign_post(key, ctype, MAX_UPLOAD)
    return 200, {"key": key, "challenge": ch, "upload": post, "max_bytes": MAX_UPLOAD}


def analyse(s, body):
    key, ch = body.get("key", ""), body.get("challenge")
    if not key.startswith(f"clips/{s['session_id']}/") or ".." in key:
        raise HttpError(400, "key does not belong to this session")
    if ch not in ("passive",) + CHALLENGE_TYPES:
        raise HttpError(400, "unknown challenge")
    if s["status"] != "active":
        raise HttpError(409, "session already concluded")
    t0 = time.perf_counter()
    local = f"/tmp/{uuid.uuid4().hex}{os.path.splitext(key)[1]}"
    try:
        try:
            store.download(key, local)
        except Exception:
            raise HttpError(404, "clip not uploaded")
        t_dl = time.perf_counter()
        res = analyze_clip(local, ch)
    finally:
        if os.path.exists(local):
            os.remove(local)
    refs = []
    for i, (label, jpg) in enumerate(res.pop("evidence", [])):
        ek = f"evidence/{s['session_id']}/{len(s['clips']):02d}_{ch}_{i}.jpg"
        store.put_bytes(ek, jpg, "image/jpeg")
        refs.append({"label": label, "key": ek})
    res.pop("series", None)
    t_an = time.perf_counter()
    step = add_clip(s, res, CFG, use_model=USE_BEDROCK, evidence_refs=refs, clip_key=key)
    t_ag = time.perf_counter()
    save(s)
    _observe_step(s, step, res, {"download_ms": (t_dl - t0) * 1000, "analysis_ms": (t_an - t_dl) * 1000,
                                 "agent_ms": (t_ag - t_an) * 1000, "total_ms": (time.perf_counter() - t0) * 1000})
    view = public_view(s)
    return 200, {"clip": view["clips"][-1], "step": timeline_entry(step), **view}


def _observe_step(s, step, res, timing):
    d = step["decision"]
    obs.log("clip_analysed", session_id=s["session_id"], challenge=res["challenge"], usable=res["usable"],
            reason=res["reason"], frames=res.get("timing_ms", {}).get("frames"), action=d["action"],
            next_challenge=d.get("challenge"), policy=d.get("policy"), fallback_reason=d.get("fallback_reason"),
            risk=step.get("risk"), **{k: round(v, 1) for k, v in timing.items()})
    obs.metric("ClipLatencyMs", timing["total_ms"], "Milliseconds", Challenge=res["challenge"])
    obs.metric("AnalysisLatencyMs", timing["analysis_ms"], "Milliseconds")
    obs.metric("AgentLatencyMs", timing["agent_ms"], "Milliseconds", Policy=d.get("policy", "?"))
    obs.metric("ClipsAnalysed", 1, "Count", Usable=str(bool(res["usable"])))
    if d.get("policy") == "rule_fallback":
        obs.metric("BedrockFallbacks", 1, "Count")
    if s["status"] == "concluded":
        _observe_verdict(s)


def _observe_verdict(s):
    v = s["result"]["verdict"]
    obs.metric("Verdicts", 1, "Count", Verdict=v)
    obs.metric("InconclusiveRate", 100.0 if v == "INCONCLUSIVE" else 0.0, "Percent")
    obs.metric("ChallengesToVerdict", len(s["result"]["challenges_done"]), "Count")
    obs.log("session_concluded", session_id=s["session_id"], verdict=v, risk=s["result"]["risk"])


def do_skip(s, body):
    ch = body.get("challenge")
    if ch not in ("passive",) + CHALLENGE_TYPES:
        raise HttpError(400, "unknown challenge")
    step = skip(s, ch, CFG, use_model=USE_BEDROCK) if ch != "passive" else _skip_passive(s)
    save(s)
    if s["status"] == "concluded":
        _observe_verdict(s)
    return 200, {"step": timeline_entry(step), **public_view(s)}


def _skip_passive(s):
    for c in s["clips"]:
        if c["challenge"] == "passive" and not c.get("usable"):
            c["superseded"] = True
    return _run_decision(s, CFG, USE_BEDROCK, {"type": "user_skipped", "challenge": "passive", "note": "user skipped"})


def do_finish(s, _body):
    if s["status"] == "active":
        finish(s, CFG)
        save(s)
        _observe_verdict(s)
    return 200, public_view(s)


def delete_session(sid):
    s = load(sid)
    for prefix in (f"clips/{sid}/", f"evidence/{sid}/"):
        store.delete_prefix(prefix)
    store.delete_session(s["session_id"])
    obs.log("session_deleted", session_id=sid)
    return 200, {"deleted": sid}


ROUTES = {
    ("POST", "/sessions/{id}/upload-url"): upload_url,
    ("POST", "/sessions/{id}/clips"): analyse,
    ("POST", "/sessions/{id}/skip"): do_skip,
    ("POST", "/sessions/{id}/finish"): do_finish,
}


def handler(event, context):
    t0 = time.perf_counter()
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    route = event.get("routeKey", "").split(" ", 1)[-1]
    sid = (event.get("pathParameters") or {}).get("id")
    obs.set_context(request_id=getattr(context, "aws_request_id", None), route=route, session_id=sid)
    try:
        body = json.loads(event.get("body") or "{}") if method in ("POST", "PUT") else {}
        if not isinstance(body, dict):
            raise HttpError(400, "body must be a JSON object")
        if route == "/health":
            import cv2
            status, out = 200, {"ok": True, "opencv": cv2.__version__, "config": CFG.get("version"),
                                "bedrock": USE_BEDROCK, "model": os.environ.get("BEDROCK_MODEL_ID")}
        elif route == "/sessions" and method == "POST":
            status, out = create_session(body)
        elif route == "/sessions/{id}" and method == "GET":
            status, out = 200, public_view(load(sid))
        elif route == "/sessions/{id}" and method == "DELETE":
            status, out = delete_session(sid)
        elif route == "/sessions/{id}/trace":
            s = load(sid)
            status, out = 200, {"session_id": sid, "config_version": CFG.get("version"), "status": s["status"],
                                "result": s["result"], "clips": public_view(s)["clips"], "trace": s["trace"]}
        elif (method, route) in ROUTES:
            status, out = ROUTES[(method, route)](load(sid), body)
        else:
            raise HttpError(404, "no such route")
    except HttpError as e:
        status, out = e.status, {"error": e.msg}
    except Exception as e:  # noqa: BLE001
        obs.log("error", level="ERROR", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-2000:])
        obs.metric("Errors", 1, "Count", Route=route or "?")
        status, out = 500, {"error": "internal error"}
    obs.log("request", method=method, status=status, ms=round((time.perf_counter() - t0) * 1000, 1))
    return resp(status, out)
