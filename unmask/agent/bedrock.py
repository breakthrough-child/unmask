"""Bedrock Converse tool-use policy. The model picks the next action; it never
produces the verdict. Every call is validated against the rule guardrails and
falls back to the deterministic policy on error, timeout or invalid output."""
from __future__ import annotations

import json
import os
import time

from .policy import CHALLENGE_TYPES, forced_action, rule_policy

DEFAULT_MODEL_ID = "anthropic.claude-opus-5-5"

SYSTEM_PROMPT = """You are the decision step of Unmask, a tool that helps a person check whether the caller on a video call is a real-time face swap.
OpenCV has measured the caller's video. You receive per-signal suspicion scores in [0,1] (0.5 = undecided) and an uncertainty for each (1 = unknown / borderline).
Choose exactly ONE next action by calling one of the tools. Policy:
- Pick the challenge that best resolves the most uncertain, most important signals. Which signals each challenge measures is given in `resolves`:
  borderline seam or pose -> head_turn; odd lighting -> light_change; borderline texture or occlusion -> hand_pass.
- Only choose from `available_challenges`, never repeat one in `challenges_completed`, never exceed `max_challenges`.
- Call conclude when the scores are already decisive or nothing useful is left to test.
- If the last clip was unusable, call request_better_conditions with a short, specific instruction.
You only advise. You never end calls, contact anyone, or try to identify the caller, and you do not decide the verdict (a deterministic scorer does).
Give a one-sentence reason that cites the signal values you relied on."""

TOOLS = [
    {"toolSpec": {
        "name": "issue_challenge",
        "description": "Ask the user to relay a physical challenge to the caller and record their response.",
        "inputSchema": {"json": {
            "type": "object", "additionalProperties": False, "required": ["challenge", "reason"],
            "properties": {"challenge": {"type": "string", "enum": list(CHALLENGE_TYPES)},
                           "reason": {"type": "string", "description": "One sentence citing the signals."}}}}}},
    {"toolSpec": {
        "name": "request_better_conditions",
        "description": "The last clip could not be analysed. Ask the user to fix the recording conditions.",
        "inputSchema": {"json": {
            "type": "object", "additionalProperties": False, "required": ["instruction", "reason"],
            "properties": {"instruction": {"type": "string"}, "reason": {"type": "string"}}}}}},
    {"toolSpec": {
        "name": "conclude",
        "description": "Stop issuing challenges; the deterministic scorer will produce the verdict.",
        "inputSchema": {"json": {
            "type": "object", "additionalProperties": False, "required": ["reason"],
            "properties": {"reason": {"type": "string"}}}}}},
]


class PolicyError(Exception):
    pass


_client = None


def _bedrock():
    global _client
    if _client is None:
        import boto3
        from botocore.config import Config
        _client = boto3.client(
            "bedrock-runtime", region_name=os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION", "us-east-1"),
            config=Config(read_timeout=float(os.environ.get("BEDROCK_TIMEOUT_S", "12")), connect_timeout=3,
                          retries={"max_attempts": 1, "mode": "standard"}))
    return _client


def _validate(call: dict, state: dict) -> None:
    name, args = call["name"], call["input"]
    if name == "issue_challenge":
        ch = args.get("challenge")
        if ch not in state["available_challenges"]:
            raise PolicyError(f"challenge {ch!r} not available")
        if ch in state["challenges_completed"]:
            raise PolicyError(f"challenge {ch!r} already completed")
        if state["challenge_attempts"] >= state["max_challenges"]:
            raise PolicyError("challenge budget exhausted")
    if not str(args.get("reason", "")).strip():
        raise PolicyError("missing reason")


def bedrock_policy(state: dict, cfg: dict, confident: bool) -> dict:
    """Ask the model for the next action. Raises PolicyError on any problem."""
    model_id = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
    msg = {"role": "user", "content": [{"text": "Current session state (JSON):\n" + json.dumps(state, sort_keys=True)
                                        + f"\nScorer early-stop rule says confident={str(confident).lower()}."
                                        + "\nCall exactly one tool now."}]}
    req = dict(modelId=model_id, system=[{"text": SYSTEM_PROMPT}], messages=[msg],
               toolConfig={"tools": TOOLS, "toolChoice": {"auto": {}}},
               inferenceConfig={"maxTokens": 400})
    effort = os.environ.get("BEDROCK_EFFORT", "low")
    t0 = time.perf_counter()
    try:
        client = _bedrock()
        try:
            resp = client.converse(**req, **({"additionalModelRequestFields": {"output_config": {"effort": effort}}}
                                              if effort else {}))
        except client.exceptions.ValidationException:
            resp = client.converse(**req)  # model/region rejects the effort field
    except PolicyError:
        raise
    except Exception as e:  # botocore errors, timeouts, missing access, no credentials
        raise PolicyError(f"{type(e).__name__}: {e}") from e
    latency = (time.perf_counter() - t0) * 1000
    calls = [b["toolUse"] for b in resp.get("output", {}).get("message", {}).get("content", []) if "toolUse" in b]
    if len(calls) != 1:
        raise PolicyError(f"expected one tool call, got {len(calls)} (stopReason={resp.get('stopReason')})")
    call = calls[0]
    _validate(call, state)
    args = call["input"]
    out = {"action": call["name"], "reason": args["reason"].strip(), "model_id": model_id,
           "model_latency_ms": round(latency, 1), "usage": resp.get("usage")}
    if call["name"] == "issue_challenge":
        out["challenge"] = args["challenge"]
    if call["name"] == "request_better_conditions":
        out["instruction"] = args.get("instruction")
    return out


def decide(state: dict, cfg: dict, confident: bool, use_model: bool = True) -> dict:
    """Guardrails first, then the model, then the deterministic fallback."""
    t0 = time.perf_counter()
    forced = forced_action(state, cfg, confident)
    if forced is not None and forced.get("forced") in ("repeated_failure", "unusable_clip", "max_challenges",
                                                         "no_challenges_left"):
        forced.update(policy="guardrail", latency_ms=round((time.perf_counter() - t0) * 1000, 1))
        return forced
    if use_model:
        try:
            out = bedrock_policy(state, cfg, confident)
            out.update(policy="bedrock", latency_ms=round((time.perf_counter() - t0) * 1000, 1))
            return out
        except PolicyError as e:
            fb = rule_policy(state, cfg, confident)
            fb.update(policy="rule_fallback", fallback_reason=str(e)[:300],
                      latency_ms=round((time.perf_counter() - t0) * 1000, 1))
            return fb
    out = rule_policy(state, cfg, confident)
    out.update(policy="rule", latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    return out
