"""Session state machine: clip result -> scores -> decision -> trace step.

Storage-agnostic: callers persist the returned dict (DynamoDB in Lambda, JSON
files locally). The verdict always comes from unmask.scoring, never the LLM."""
from __future__ import annotations

import time
import uuid

from ..scoring import is_confident, load_config, session_risk
from .bedrock import decide
from .policy import CHALLENGE_TYPES, PROMPTS

# Clip fields kept in the session (series/evidence bytes are too big / stored elsewhere)
_KEEP = ("challenge", "usable", "reason", "instruction", "features", "quality", "timing_ms", "opencv_version")


def new_session(mode: str = "live", available: list[str] | None = None, session_id: str | None = None) -> dict:
    now = time.time()
    return {
        "session_id": session_id or uuid.uuid4().hex,
        "created_at": now,
        "expires_at": int(now + 24 * 3600),
        "mode": mode,
        "available_challenges": [c for c in (available or list(CHALLENGE_TYPES)) if c in CHALLENGE_TYPES],
        "status": "active",
        "clips": [],
        "trace": [],
        "next_action": {"action": "record_passive", "challenge": "passive",
                        "prompt": "Keep the caller's video visible. Recording a 10-second passive scan."},
        "result": None,
    }


def _signal_view(risk: dict) -> dict:
    return {k: {"score": v["score"], "uncertainty": v["uncertainty"]} for k, v in risk["signals"].items()}


def _next_action(decision: dict, state_last: dict | None) -> dict:
    a = decision["action"]
    if a == "issue_challenge":
        ch = decision["challenge"]
        return {"action": a, "challenge": ch, "prompt": PROMPTS[ch], "reason": decision["reason"]}
    if a == "request_better_conditions":
        ch = (decision.get("challenge") or (state_last or {}).get("challenge") or "passive")
        instr = decision.get("instruction") or (state_last or {}).get("instruction") or decision["reason"]
        return {"action": a, "challenge": ch, "prompt": instr, "reason": decision["reason"],
                "reason_code": decision.get("reason_code") or (state_last or {}).get("reason")}
    return {"action": "conclude", "reason": decision["reason"]}


def _run_decision(session: dict, cfg: dict, use_model: bool, event: dict) -> dict:
    from .policy import build_state
    clips = [c for c in session["clips"] if not c.get("superseded")]
    risk = session_risk(session["clips"], cfg)
    state = build_state(clips, risk, cfg, session["available_challenges"])
    # Budgets and failure counters count every clip, including retried/skipped ones;
    # only `last_clip` (what needs fixing now) ignores superseded clips.
    full = build_state(session["clips"], risk, cfg, session["available_challenges"])
    for k in ("total_unusable", "consecutive_unusable", "challenge_attempts"):
        state[k] = full[k]
    confident = is_confident(risk, cfg)
    decision = decide(state, cfg, confident, use_model=use_model)
    prev = session["trace"][-1]["signals_after"] if session["trace"] else None
    step = {
        "step": len(session["trace"]) + 1,
        "ts": round(time.time(), 3),
        "event": event,
        "signals_before": prev,
        "signals_after": _signal_view(risk),
        "risk": risk["risk"],
        "provisional_verdict": risk["verdict"],
        "confident": confident,
        "agent_input": state,
        "decision": decision,
    }
    session["trace"].append(step)
    nxt = _next_action(decision, state["last_clip"])
    session["next_action"] = nxt
    if nxt["action"] == "conclude":
        session["status"] = "concluded"
        session["result"] = {k: risk[k] for k in ("risk", "verdict", "signals", "challenges_done",
                                                  "usable_clips", "top_reasons", "advice", "config_version")}
    else:
        session["result"] = None
    session["provisional"] = {"risk": risk["risk"], "verdict": risk["verdict"]}
    return step


def add_clip(session: dict, clip_result: dict, cfg: dict | None = None, use_model: bool = True,
             evidence_refs: list | None = None, clip_key: str | None = None) -> dict:
    """Record an analysed clip and run one agent step. Returns the trace step."""
    if session["status"] != "active":
        raise ValueError("session already concluded")
    cfg = cfg or load_config()
    rec = {k: clip_result.get(k) for k in _KEEP}
    rec["index"] = len(session["clips"])
    rec["evidence"] = evidence_refs or []
    rec["clip_key"] = clip_key
    # A new clip for the same request supersedes the earlier unusable one for policy purposes.
    for c in session["clips"]:
        if not c.get("usable") and c["challenge"] == rec["challenge"]:
            c["superseded"] = True
    session["clips"].append(rec)
    event = {"type": "clip_analysed", "challenge": rec["challenge"], "usable": rec["usable"],
             "reason": rec["reason"], "features": rec["features"], "timing_ms": rec["timing_ms"],
             "evidence": [e.get("label") if isinstance(e, dict) else e for e in rec["evidence"]]}
    return _run_decision(session, cfg, use_model, event)


def skip(session: dict, challenge: str, cfg: dict | None = None, use_model: bool = True,
         note: str = "user could not provide this clip") -> dict:
    """User declines / cannot provide the requested clip (e.g. upload mode without that file)."""
    if session["status"] != "active":
        raise ValueError("session already concluded")
    cfg = cfg or load_config()
    if challenge in session["available_challenges"]:
        session["available_challenges"].remove(challenge)
    for c in session["clips"]:
        if c["challenge"] == challenge and not c.get("usable"):
            c["superseded"] = True
    return _run_decision(session, cfg, use_model, {"type": "user_skipped", "challenge": challenge, "note": note})


def finish(session: dict, cfg: dict | None = None, note: str = "user ended the check") -> dict:
    """Human control: the user can stop at any time and get the verdict so far."""
    cfg = cfg or load_config()
    risk = session_risk(session["clips"], cfg)
    session["trace"].append({"step": len(session["trace"]) + 1, "ts": round(time.time(), 3),
                             "event": {"type": "user_stopped", "note": note},
                             "signals_after": _signal_view(risk), "risk": risk["risk"],
                             "provisional_verdict": risk["verdict"],
                             "decision": {"action": "conclude", "policy": "human", "reason": note}})
    session["status"] = "concluded"
    session["next_action"] = {"action": "conclude", "reason": note}
    session["result"] = {k: risk[k] for k in ("risk", "verdict", "signals", "challenges_done",
                                              "usable_clips", "top_reasons", "advice", "config_version")}
    return session["trace"][-1]


def replay(clips_by_challenge: dict, analyze, cfg: dict | None = None, use_model: bool = False,
           fixed_order: list[str] | None = None, max_steps: int = 12) -> dict:
    """Offline loop over pre-recorded clips. `analyze(challenge) -> clip_result` (or None if absent).

    fixed_order=None -> agent chooses; fixed_order=[...] -> ablation baseline that plays
    the challenges in that order regardless of the scores; [] -> passive only."""
    cfg = cfg or load_config()
    avail = [c for c in CHALLENGE_TYPES if c in clips_by_challenge]
    s = new_session(mode="replay", available=avail)
    cache: dict = {}

    def get(ch):
        if ch not in cache:
            cache[ch] = analyze(ch) if ch in clips_by_challenge else None
        return cache[ch]

    passive = get("passive")
    if passive is None:
        skip(s, "passive", cfg, use_model)
    else:
        add_clip(s, passive, cfg, use_model)
    if fixed_order is not None:
        s["trace"][-1]["decision"]["note"] = "fixed-order baseline: decision ignored"
        for ch in fixed_order:
            r = get(ch)
            if r is not None:
                s["status"] = "active"  # the baseline ignores the agent's stop decisions
                add_clip(s, r, cfg, use_model)
        finish(s, cfg, note="fixed sequence finished")
        return s
    for _ in range(max_steps):
        if s["status"] != "active":
            break
        na = s["next_action"]
        ch = na.get("challenge")
        if na["action"] == "issue_challenge":
            r = get(ch)
            if r is None:
                skip(s, ch, cfg, use_model)
            else:
                add_clip(s, r, cfg, use_model)
        elif na["action"] == "request_better_conditions":
            # No better recording exists offline: the user "skips" it.
            if ch == "passive":
                s["clips"][-1]["superseded"] = True
                _run_decision(s, cfg, use_model, {"type": "user_skipped", "challenge": "passive",
                                                  "note": "no better passive clip available"})
            else:
                skip(s, ch, cfg, use_model, note="no better recording available offline")
    if s["status"] == "active":
        finish(s, cfg, note="replay step budget exhausted")
    return s
