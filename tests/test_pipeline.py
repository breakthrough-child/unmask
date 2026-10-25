"""End-to-end: synthetic video files -> decode -> analyse -> agent replay -> verdict."""
import json

from unmask import synth
from unmask.agent.session import replay
from unmask.scoring import load_config
from unmask.vision.analyze import analyze_clip
from unmask.vision.video import load_clip

from .conftest import TruthDetector


def test_video_roundtrip(tmp_path):
    p = synth.person(0)
    frames, fps, _ = synth.render_clip(p, "passive", False, seconds=3)
    path = str(tmp_path / "c.mp4")
    synth.write_video(path, frames, fps)
    clip = load_clip(path)
    assert 25 <= len(clip.frames) <= 31
    assert clip.frames[0].shape[1] <= 640
    assert 8 < clip.fps < 12


def _session(pid, swapped, fixed_order=None):
    p = synth.person(pid)

    def analyze(ch):
        frames, fps, truth = synth.render_clip(p, ch, swapped)
        from unmask.vision.video import clip_from_frames
        return analyze_clip(clip_from_frames(frames, fps), ch, detector=TruthDetector(truth, swapped), with_evidence=False)

    clips = {c: True for c in ("passive", "head_turn", "hand_pass", "light_change")}
    return replay(clips, analyze, load_config(), use_model=False, fixed_order=fixed_order)


def test_agent_replay_reaches_verdict_and_trace_is_json():
    s = _session(0, True)
    assert s["status"] == "concluded"
    assert s["result"]["verdict"] in ("SUSPICIOUS", "HIGH_RISK", "LOW_RISK", "INCONCLUSIVE")
    json.dumps(s)  # trace must be serialisable for DynamoDB / export
    assert all("decision" in st and "reason" in st["decision"] for st in s["trace"])


def test_fixed_order_plays_every_challenge():
    s = _session(1, False, fixed_order=["head_turn", "hand_pass", "light_change"])
    assert {c["challenge"] for c in s["clips"]} == {"passive", "head_turn", "hand_pass", "light_change"}
