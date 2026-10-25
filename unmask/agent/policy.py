"""Deterministic rule-based policy. Used as the Bedrock fallback and as the
guardrail that every model decision is validated against."""
from __future__ import annotations

CHALLENGE_TYPES = ("head_turn", "hand_pass", "light_change")

ACTIONS = ("issue_challenge", "request_better_conditions", "conclude")

PROMPTS = {
    "head_turn": "Ask them to turn their head slowly all the way to the left, then all the way to the right.",
    "hand_pass": "Ask them to pass one hand slowly across their face, from one side to the other.",
    "light_change": "Ask them to move closer to a lamp or window, or shine their phone torch on their face.",
}


def build_state(clips: list[dict], risk: dict, cfg: dict, available: list[str]) -> dict:
    """Compact, JSON-able view of the session used by both policies (and logged in the trace)."""
    pol = cfg["policy"]
    attempts = [c for c in clips if c["challenge"] != "passive"]
    consecutive_unusable = 0
    for c in reversed(clips):
        if c.get("usable"):
            break
        consecutive_unusable += 1
    done_ok = sorted({c["challenge"] for c in attempts if c.get("usable")})
    last = clips[-1] if clips else None
    return {
        "signals": {k: {"score": v["score"], "uncertainty": v["uncertainty"]} for k, v in risk["signals"].items()},
        "risk": risk["risk"],
        "provisional_verdict": risk["verdict"],
        "challenges_completed": done_ok,
        "challenge_attempts": len(attempts),
        "max_challenges": pol["max_challenges"],
        "available_challenges": [c for c in available if c in CHALLENGE_TYPES],
        "last_clip": None if last is None else {
            "challenge": last["challenge"], "usable": last.get("usable"),
            "reason": last.get("reason"), "instruction": last.get("instruction")},
        "consecutive_unusable": consecutive_unusable,
        "total_unusable": sum(1 for c in clips if not c.get("usable")),
        "resolves": pol["resolves"],
    }


def challenge_values(state: dict, cfg: dict) -> dict[str, float]:
    """Expected uncertainty resolved by each still-useful challenge:
    sum over the signals it measures of weight * current uncertainty."""
    out = {}
    for ch in state["available_challenges"]:
        if ch in state["challenges_completed"]:
            continue
        v = 0.0
        for sig in state["resolves"].get(ch, []):
            v += cfg["signals"][sig]["weight"] * state["signals"][sig]["uncertainty"]
        out[ch] = round(v, 3)
    return out


def forced_action(state: dict, cfg: dict, confident: bool) -> dict | None:
    """Hard rules both policies must obey. Returns an action or None."""
    pol = cfg["policy"]
    last = state["last_clip"]
    if state["consecutive_unusable"] >= pol["max_consecutive_unusable"] or \
            state["total_unusable"] >= pol["max_total_unusable"]:
        return {"action": "conclude", "reason": "Repeated unusable clips: stopping with INCONCLUSIVE rather than guessing.",
                "forced": "repeated_failure"}
    if last is not None and not last["usable"]:
        return {"action": "request_better_conditions", "reason_code": last["reason"],
                "challenge": last["challenge"],
                "reason": f"The last clip was unusable ({last['reason']}); fixing that first.",
                "forced": "unusable_clip"}
    if state["challenge_attempts"] >= pol["max_challenges"]:
        return {"action": "conclude", "reason": "Reached the maximum of 4 challenges.", "forced": "max_challenges"}
    if not challenge_values(state, cfg):
        return {"action": "conclude", "reason": "No further challenges available.", "forced": "no_challenges_left"}
    if confident:
        return {"action": "conclude", "reason": "Scores are already decisive; no further challenge needed.",
                "forced": "confident"}
    return None


def rule_policy(state: dict, cfg: dict, confident: bool) -> dict:
    forced = forced_action(state, cfg, confident)
    if forced:
        return forced
    values = challenge_values(state, cfg)
    order = {c: i for i, c in enumerate(CHALLENGE_TYPES)}
    best = max(values, key=lambda c: (values[c], -order[c]))
    sigs = state["resolves"][best]
    worst = max(sigs, key=lambda s: state["signals"][s]["uncertainty"])
    sc = state["signals"][worst]["score"]
    seen = "unmeasured" if sc is None else f"borderline at {sc:.2f}" if 0.3 < sc < 0.7 else f"at {sc:.2f}"
    return {"action": "issue_challenge", "challenge": best,
            "reason": f"The {worst} signal is {seen}; a {best.replace('_', ' ')} tests it directly "
                      f"(expected uncertainty resolved {values[best]:.2f}).",
            "values": values}
