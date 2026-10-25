"""Bedrock Converse tool-use policy. The model picks the next action; it never
produces the verdict. Every call is validated against the rule guardrails and
falls back to the deterministic policy on error, timeout or invalid output."""
from __future__ import annotations

import copy
import json
import os
import time

from .policy import CHALLENGE_TYPES, forced_action, rule_policy

# Fast, cheap default for a one-tool-call decision: Amazon Nova Lite through the EU
# cross-region inference profile (the project Region is eu-north-1). Override with
# BEDROCK_MODEL_ID (e.g. eu.amazon.nova-pro-v1:0, anthropic.claude-haiku-5-5, an ARN).
DEFAULT_MODEL_ID = "eu.amazon.nova-lite-v1:0"
# Cross-region inference-profile prefixes Bedrock uses in front of model IDs.
PROFILE_PREFIXES = ("us", "eu", "apac", "global", "us-gov", "jp", "au", "ca")
_MODEL_ID_ERRORS = ("inference profile", "on-demand throughput", "model identifier", "model id", "provided model")
_resolved: dict[str, str] = {}

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


def model_family(model_id: str) -> str:
    m = model_id.lower()
    if "amazon.nova" in m:
        return "nova"
    if "anthropic." in m:
        return "anthropic"
    return "other"


def tools_for(family: str) -> list[dict]:
    """Nova's tool schemas accept a JSON-Schema subset; drop additionalProperties for it."""
    if family != "nova":
        return TOOLS
    tools = copy.deepcopy(TOOLS)
    for t in tools:
        t["toolSpec"]["inputSchema"]["json"].pop("additionalProperties", None)
    return tools


def request_for(family: str, model_id: str, base: dict) -> tuple[dict, dict]:
    """(request, optional extras). Extras are dropped on retry if the model rejects them."""
    max_tokens = int(os.environ.get("BEDROCK_MAX_TOKENS", "1000" if family == "nova" else "2000"))
    req = dict(base, modelId=model_id, toolConfig={"tools": tools_for(family), "toolChoice": {"auto": {}}})
    if family == "nova":
        # AWS recommends greedy decoding for Nova tool calling.
        req["inferenceConfig"] = {"maxTokens": max_tokens, "temperature": 0.0}
        extras = {"additionalModelRequestFields": {"inferenceConfig": {"topK": 1}}}
    else:
        req["inferenceConfig"] = {"maxTokens": max_tokens}
        effort = os.environ.get("BEDROCK_EFFORT", "low")
        extras = {"additionalModelRequestFields": {"output_config": {"effort": effort}}} \
            if family == "anthropic" and effort else {}
    return req, extras


class PolicyError(Exception):
    pass


_client = None


def _bedrock():
    global _client
    if _client is None:
        import boto3
        from botocore.config import Config
        _client = boto3.client(
            "bedrock-runtime", region_name=os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION", "eu-north-1"),
            config=Config(read_timeout=float(os.environ.get("BEDROCK_TIMEOUT_S", "12")), connect_timeout=3,
                          retries={"max_attempts": 1, "mode": "standard"}))
    return _client


def region_prefix(region: str) -> str | None:
    """Geographic inference-profile prefix for a region (us-east-1 -> 'us', eu-west-1 -> 'eu', ...)."""
    if region.startswith("us-gov-"):
        return "us-gov"
    for start, prefix in (("us-", "us"), ("ca-", "us"), ("eu-", "eu"), ("ap-", "apac")):
        if region.startswith(start):
            return prefix
    return None


def candidate_model_ids(model_id: str, region: str, mode: str = "auto") -> list[str]:
    """Model IDs to try, in order.

    BEDROCK_PROFILE=auto (default): the ID as given, then the regional profile
    (us./eu./apac.), then global. =none: only as given. =us|eu|apac|global|...:
    force that prefix. IDs that already carry a prefix, or ARNs, are used as is."""
    if model_id.startswith("arn:") or model_id.split(".", 1)[0] in PROFILE_PREFIXES:
        return [model_id]
    if mode == "none":
        return [model_id]
    if mode in PROFILE_PREFIXES:
        return [f"{mode}.{model_id}"]
    out = [model_id]
    rp = region_prefix(region)
    if rp:
        out.append(f"{rp}.{model_id}")
    out.append(f"global.{model_id}")
    return out


def _is_model_id_error(e: Exception) -> bool:
    return any(k in str(e).lower() for k in _MODEL_ID_ERRORS)


def _converse(client, req: dict, extras: dict):
    """Converse call; retries without the model-specific extras if they are rejected."""
    if extras:
        try:
            return client.converse(**req, **extras)
        except client.exceptions.ValidationException as e:
            if _is_model_id_error(e):
                raise
    return client.converse(**req)


def _normalise(call: dict) -> dict:
    """Tolerate model quirks: input as a JSON string, enum values in another case."""
    args = call.get("input")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError as e:
            raise PolicyError(f"tool input is not JSON: {e}") from e
    if not isinstance(args, dict):
        raise PolicyError(f"tool input must be an object, got {type(args).__name__}")
    if isinstance(args.get("challenge"), str):
        args["challenge"] = args["challenge"].strip().lower()
    name = str(call.get("name", "")).strip()
    if name not in ("issue_challenge", "request_better_conditions", "conclude"):
        raise PolicyError(f"unknown tool {name!r}")
    return {**call, "name": name, "input": args}


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
    configured = os.environ.get("BEDROCK_MODEL_ID") or DEFAULT_MODEL_ID
    region = os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION", "eu-north-1")
    msg = {"role": "user", "content": [{"text": "Current session state (JSON):\n" + json.dumps(state, sort_keys=True)
                                        + f"\nScorer early-stop rule says confident={str(confident).lower()}."
                                        + "\nCall exactly one tool now."}]}
    # toolChoice "auto": Nova and current Claude models both support it (Claude rejects
    # forced choice); the prompt asks for exactly one call and _validate enforces it.
    base = dict(system=[{"text": SYSTEM_PROMPT}], messages=[msg])
    candidates = [_resolved[configured]] if configured in _resolved else \
        candidate_model_ids(configured, region, os.environ.get("BEDROCK_PROFILE", "auto"))
    t0 = time.perf_counter()
    resp, model_id, last_err = None, None, None
    try:
        client = _bedrock()
        for mid in candidates:
            try:
                req, extras = request_for(model_family(mid), mid, base)
                resp = _converse(client, req, extras)
                model_id = _resolved[configured] = mid
                break
            except client.exceptions.ValidationException as e:
                last_err = e
                if not _is_model_id_error(e):
                    raise
    except Exception as e:  # botocore errors, timeouts, missing access, no credentials
        raise PolicyError(f"{type(e).__name__}: {e}") from e
    if resp is None:
        raise PolicyError(f"no usable model ID among {candidates}: {last_err}")
    latency = (time.perf_counter() - t0) * 1000
    calls = [b["toolUse"] for b in resp.get("output", {}).get("message", {}).get("content", []) if "toolUse" in b]
    if len(calls) != 1:
        raise PolicyError(f"expected one tool call, got {len(calls)} (stopReason={resp.get('stopReason')})")
    call = _normalise(calls[0])
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
