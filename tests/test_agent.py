import pytest

from unmask.agent import bedrock
from unmask.agent.bedrock import decide
from unmask.agent.policy import build_state, rule_policy
from unmask.agent.session import add_clip, finish, new_session, skip
from unmask.scoring import load_config, session_risk

CFG = load_config()


def feats(signal, offset):
    return {f: s["center"] + offset * s["scale"] * s.get("direction", 1)
            for f, s in CFG["signals"][signal]["features"].items()}


def clip(ch, usable=True, reason=None, **f):
    return {"challenge": ch, "usable": usable, "reason": reason, "instruction": "fix it" if reason else None,
            "features": f, "quality": {}, "timing_ms": {"total": 1}}


def state_for(clips, available=("head_turn", "hand_pass", "light_change")):
    return build_state(clips, session_risk(clips, CFG), CFG, list(available))


def test_borderline_seam_picks_head_turn():
    f = {**feats("seam", 0.1), **feats("texture", -5), **feats("temporal", -5)}
    d = rule_policy(state_for([clip("passive", **f)]), CFG, False)
    assert d["action"] == "issue_challenge" and d["challenge"] == "head_turn"
    assert "seam" in d["reason"] or "pose" in d["reason"]


def test_vision_output_changes_choice():
    """Same session, different OpenCV results -> different next challenge."""
    seam_ok = {**feats("seam", -6), **feats("texture", 0.1), **feats("temporal", -6)}
    ht = clip("head_turn", **{**seam_ok, **feats("pose", -6)})
    d1 = rule_policy(state_for([clip("passive", **seam_ok), ht]), CFG, False)
    assert d1["challenge"] == "hand_pass"
    d2 = rule_policy(state_for([clip("passive", **seam_ok), ht], available=("head_turn", "light_change")), CFG, False)
    assert d2["challenge"] == "light_change"


def test_unusable_clip_requests_better_conditions_then_inconclusive():
    s = new_session()
    add_clip(s, clip("passive", usable=False, reason="too_dark"), CFG, use_model=False)
    assert s["next_action"]["action"] == "request_better_conditions"
    assert s["next_action"]["reason_code"] == "too_dark"
    add_clip(s, clip("passive", usable=False, reason="too_dark"), CFG, use_model=False)
    assert s["status"] == "concluded"
    assert s["result"]["verdict"] == "INCONCLUSIVE"


def test_max_four_challenges():
    s = new_session()
    add_clip(s, clip("passive", **feats("seam", 0)), CFG, use_model=False)
    n = 0
    while s["status"] == "active" and n < 10:
        na = s["next_action"]
        if na["action"] == "issue_challenge":
            add_clip(s, clip(na["challenge"], usable=False, reason="challenge_not_performed"), CFG, use_model=False)
        else:
            add_clip(s, clip(na["challenge"], **feats("seam", 0)), CFG, use_model=False)
        n += 1
    assert s["status"] == "concluded"
    assert sum(1 for c in s["clips"] if c["challenge"] != "passive") <= 4


def test_skip_removes_challenge_and_continues():
    s = new_session()
    add_clip(s, clip("passive", **feats("seam", 0)), CFG, use_model=False)
    first = s["next_action"]["challenge"]
    skip(s, first, CFG, use_model=False)
    assert first not in s["available_challenges"]
    assert s["next_action"].get("challenge") != first


def test_human_can_stop_any_time():
    s = new_session()
    add_clip(s, clip("passive", **feats("seam", 0)), CFG, use_model=False)
    finish(s, CFG)
    assert s["status"] == "concluded" and s["trace"][-1]["decision"]["policy"] == "human"


class FakeClient:
    class exceptions:
        class ValidationException(Exception):
            pass

    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.calls = response, exc, []

    def converse(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return self.response


def _tool_response(name, args):
    return {"output": {"message": {"content": [{"text": "ok"}, {"toolUse": {"toolUseId": "1", "name": name, "input": args}}]}},
            "stopReason": "tool_use", "usage": {"inputTokens": 10, "outputTokens": 5}}


@pytest.fixture
def fake(monkeypatch):
    def install(**kw):
        c = FakeClient(**kw)
        monkeypatch.setattr(bedrock, "_client", c)
        return c
    return install


def test_bedrock_decision_used_when_valid(fake):
    c = fake(response=_tool_response("issue_challenge", {"challenge": "light_change", "reason": "lighting unmeasured"}))
    d = decide(state_for([clip("passive", **feats("seam", 0))]), CFG, False)
    assert d["policy"] == "bedrock" and d["challenge"] == "light_change"
    assert c.calls[0]["toolConfig"]["toolChoice"] == {"auto": {}}


def test_bedrock_invalid_choice_falls_back(fake):
    fake(response=_tool_response("issue_challenge", {"challenge": "dance", "reason": "x"}))
    d = decide(state_for([clip("passive", **feats("seam", 0))]), CFG, False)
    assert d["policy"] == "rule_fallback" and "not available" in d["fallback_reason"]


def test_bedrock_error_falls_back(fake):
    fake(exc=TimeoutError("read timeout"))
    d = decide(state_for([clip("passive", **feats("seam", 0))]), CFG, False)
    assert d["policy"] == "rule_fallback" and d["action"] == "issue_challenge"


def test_guardrail_overrides_model_on_unusable_clip(fake):
    c = fake(response=_tool_response("conclude", {"reason": "done"}))
    d = decide(state_for([clip("passive", usable=False, reason="no_face")]), CFG, False)
    assert d["policy"] == "guardrail" and d["action"] == "request_better_conditions"
    assert not c.calls
