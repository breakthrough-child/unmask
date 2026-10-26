"""Known-answer tests for the passive signals (resolution, noise, skin, dropout)."""
import cv2
import numpy as np

from unmask.vision import passive as P
from unmask.vision.analyze import analyze_clip
from unmask.vision.face import Face

RNG = np.random.default_rng(0)


def textured(s=48, sigma=1.0):
    t = RNG.normal(128, 40, (s, s)).astype(np.float32)
    return np.clip(cv2.GaussianBlur(t, (0, 0), sigma), 0, 255)


def test_upsampled_patch_has_less_fine_detail():
    sharp = textured(48, 0.6)
    low = cv2.resize(cv2.resize(sharp, (16, 16), interpolation=cv2.INTER_AREA), (48, 48), interpolation=cv2.INTER_LINEAR)
    assert P.pyr_ratio(low) < 0.5 * P.pyr_ratio(sharp)
    assert P.hf_power(low) < 0.5 * P.hf_power(sharp)


def test_noise_level_tracks_injected_noise():
    base = np.full((48, 48), 120.0, np.float32)
    lo = np.clip(base + RNG.normal(0, 2, base.shape), 0, 255).astype(np.uint8)
    hi = np.clip(base + RNG.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
    nl, nh = P.noise_level(P.residual(lo)), P.noise_level(P.residual(hi))
    assert 2.0 < nh / nl < 4.5          # ~3x noise -> ~3x level (median filter shrinks both alike)


def test_noise_level_ignores_texture_edges():
    """Flat blocks set the level, so a strong edge pattern must not inflate it much."""
    flat = np.clip(np.full((48, 48), 120.0) + RNG.normal(0, 3, (48, 48)), 0, 255).astype(np.uint8)
    edged = flat.copy()
    edged[:, 24:] = np.clip(edged[:, 24:].astype(int) + 80, 0, 255)
    assert P.noise_level(P.residual(edged)) < 1.5 * P.noise_level(P.residual(flat))


def test_residual_correlation_helper():
    a = RNG.normal(0, 1, (32, 32))
    assert P._corr(a, a) > 0.999 and abs(P._corr(a, RNG.normal(0, 1, (32, 32)))) < 0.2
    assert P._corr(np.zeros((8, 8)), a[:8, :8]) is None


def test_dropout_separates_jumpy_from_smooth_descriptors():
    n = 40
    smooth = np.cumsum(RNG.normal(0, 0.02, (n, 5)), axis=0).astype(np.float32)
    jumpy = smooth.copy()
    for k in (9, 10, 25, 26):                      # two brief appearance jumps
        jumpy[k] += 3.0
    s_sep, s_minor, s_sw = P.kmeans_dropout(smooth, 4.0)
    j_sep, j_minor, j_sw = P.kmeans_dropout(jumpy, 4.0)
    assert j_minor < 0.2 and j_sw >= 1.0           # a small cluster that flips in and out
    assert j_sep > 1.3 * s_sep or j_minor < s_minor
    assert P.kmeans_dropout(smooth[:5], 1.0) is None


def test_skin_region_colour_difference():
    img = np.zeros((240, 320, 3), np.uint8)
    img[:] = (120, 150, 190)                       # skin-like BGR
    box = np.array([110.0, 40.0, 100.0, 125.0])
    centre, forehead, neck = P._regions_masks(img.shape, box)
    assert centre.sum() > 500 and forehead.sum() > 100 and neck.sum() > 100
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    assert np.linalg.norm(P._med_lab(lab, centre) - P._med_lab(lab, neck)) < 1e-3     # same colour -> 0
    img2 = img.copy()
    img2[neck] = (160, 140, 120)                                                       # different neck colour
    lab2 = cv2.cvtColor(img2.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    assert np.linalg.norm(P._med_lab(lab2, centre)[1:] - P._med_lab(lab2, neck)[1:]) > 10


def test_too_few_eligible_frames_gives_no_features():
    f = np.zeros((480, 640, 3), np.uint8)
    face = Face(np.array([260.0, 160.0, 120.0, 150.0]), np.zeros((5, 2)), 0.95)
    out = P.passive_features([f] * 5, [face] * 5, [0.0] * 5, [i / 10 for i in range(5)])
    assert out["p_n_eligible"] == 5 and all(out[k] is None for k in P.FEATURES)


def test_passive_features_are_off_by_default(synth_clip, monkeypatch):
    monkeypatch.delenv("UNMASK_PASSIVE_FEATURES", raising=False)
    clip, det = synth_clip(0, "passive", False)
    assert "p_n_eligible" not in analyze_clip(clip, "passive", detector=det, with_evidence=False)["features"]


def test_neck_patch_geometry():
    assert P.neck_cy(100, 150, 40, 480, fit=False) == 100 + 1.18 * 150        # pre-registered: no adjustment
    assert P.neck_cy(100, 150, 40, 480, fit=True) == 100 + 1.18 * 150          # already fits -> unchanged
    assert abs(P.neck_cy(300, 150, 40, 480, fit=True) - 460) < 1e-9            # slid up to fit the frame
    assert P.neck_cy(330, 150, 40, 480, fit=True) is None                      # would start above the chin line


def test_ineligible_frames_are_excluded(synth_clip, monkeypatch):
    monkeypatch.setenv("UNMASK_PASSIVE_FEATURES", "1")
    clip, det = synth_clip(0, "head_turn", False)
    r = analyze_clip(clip, "head_turn", detector=det, with_evidence=False)
    assert r["features"]["p_n_eligible"] < len(clip.frames)     # large yaw frames dropped
    clip2, det2 = synth_clip(0, "passive", False)
    r2 = analyze_clip(clip2, "passive", detector=det2, with_evidence=False)
    f = r2["features"]
    assert f["p_n_eligible"] >= 0.8 * len(clip2.frames)         # a frontal clip keeps most frames
    for k in ("p_res_pyr_neck", "p_noise_face_vs_bg", "p_skin_de_neck", "p_ctl_bg_noise"):
        assert f[k] is not None and np.isfinite(f[k]), k
