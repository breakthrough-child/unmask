import math

import numpy as np

from unmask.vision import signals as S
from unmask.vision.analyze import analyze_clip
from unmask.vision.pose import MODEL_POINTS, head_pose, yaw_proxy

PIDS = (0, 1, 2)


def _project(yaw_deg):
    import cv2
    rvec = np.array([0.0, math.radians(yaw_deg), 0.0])
    tvec = np.array([0.0, 0.0, 600.0])
    cam = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], float)
    pts, _ = cv2.projectPoints(MODEL_POINTS, rvec, tvec, cam, None)
    return pts.reshape(5, 2)


def test_head_pose_recovers_yaw_sign_and_magnitude():
    for yaw in (-40, -20, 0, 20, 40):
        est, _pitch = head_pose(_project(yaw), (640, 480))
        assert abs(est - yaw) < 6, (yaw, est)
        assert np.sign(yaw_proxy(_project(yaw))) == np.sign(yaw) or yaw == 0


def test_head_turn_yaw_range(synth_clip):
    clip, det = synth_clip(0, "head_turn", False)
    r = analyze_clip(clip, "head_turn", detector=det)
    assert r["features"]["yaw_range"] > 50
    assert r["usable"], r["reason"]
    assert 0.3 < r["features"]["response_latency_s"] < 2.5


def _feat(synth_clip, pid, ch, swapped, key):
    clip, det = synth_clip(pid, ch, swapped)
    return analyze_clip(clip, ch, detector=det, with_evidence=False)["features"][key]


def test_seam_flicker_higher_on_swapped(synth_clip):
    for pid in PIDS:
        assert _feat(synth_clip, pid, "passive", True, "seam_flicker") > _feat(synth_clip, pid, "passive", False, "seam_flicker")


def test_texture_smoother_on_swapped(synth_clip):
    for pid in PIDS:
        assert _feat(synth_clip, pid, "passive", True, "tex_lap_ratio") < _feat(synth_clip, pid, "passive", False, "tex_lap_ratio")


def test_landmark_jitter_higher_on_swapped(synth_clip):
    for pid in PIDS:
        assert _feat(synth_clip, pid, "passive", True, "landmark_jitter") > _feat(synth_clip, pid, "passive", False, "landmark_jitter")


def test_occlusion_detects_hand_and_face_leak(synth_clip):
    for pid in PIDS:
        clip, det = synth_clip(pid, "hand_pass", False)
        real = analyze_clip(clip, "hand_pass", detector=det, with_evidence=False)["features"]
        clip, det = synth_clip(pid, "hand_pass", True)
        fake = analyze_clip(clip, "hand_pass", detector=det, with_evidence=False)["features"]
        assert real["occ_max_frac"] > 0.25 and fake["occ_max_frac"] > 0.25
        assert fake["occ_persist"] > real["occ_persist"]


def test_lighting_follows_scene_only_when_real(synth_clip):
    for pid in PIDS:
        clip, det = synth_clip(pid, "light_change", False)
        real = analyze_clip(clip, "light_change", detector=det, with_evidence=False)["features"]
        clip, det = synth_clip(pid, "light_change", True)
        fake = analyze_clip(clip, "light_change", detector=det, with_evidence=False)["features"]
        assert real["light_ref_range"] > 6
        assert fake["light_gain_dev"] > real["light_gain_dev"]


def test_motion_masks_find_moving_square():
    frames = []
    for i in range(6):
        f = np.full((240, 320, 3), 90, np.uint8)
        f[100:160, 20 + i * 30:80 + i * 30] = 220
        frames.append(f)
    masks = S.motion_masks(frames)
    assert len(masks) == 6
    assert masks[3][100:160, 100:170].mean() > 0.2
    assert masks[3][0:60, 0:320].mean() < 0.05


def test_evidence_frames_are_jpegs(synth_clip):
    clip, det = synth_clip(1, "hand_pass", True)
    r = analyze_clip(clip, "hand_pass", detector=det)
    assert r["evidence"], "expected evidence frames"
    for label, jpg in r["evidence"]:
        assert jpg[:2] == b"\xff\xd8" and label
