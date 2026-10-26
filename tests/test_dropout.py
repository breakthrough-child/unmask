"""Known-answer tests for the within-clip dropout features (Experiment 2)."""
import numpy as np

from unmask.vision import dropout as D
from unmask.vision.face import Face

RNG = np.random.default_rng(1)
BOX = np.array([250.0, 140.0, 140.0, 180.0])


def make(n=40, fps=10, swap_pattern=None):
    """Noisy background; face region smooth (swapped look) or noisy (sharp original) per frame."""
    frames, faces, times = [], [], []
    bg = RNG.normal(120, 25, (480, 640, 3)).astype(np.float32)
    for i in range(n):
        f = bg.copy() + RNG.normal(0, 3, bg.shape)
        sharp = swap_pattern(i) if swap_pattern else False
        x, y, w, h = (int(v) for v in BOX)
        yy, xx = np.mgrid[y:y + h, x:x + w]
        face = np.full((h, w, 3), 170.0, np.float32) + 10 * np.sin(xx / 20.0)[..., None]
        if sharp:
            face = face + RNG.normal(0, 25, face.shape)
        f[y:y + h, x:x + w] = face
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
        faces.append(Face(BOX.copy(), np.zeros((5, 2)), 0.95))
        times.append(i / fps)
    return frames, faces, times


def test_stable_clip_has_no_dropout():
    fr, fc, t = make(swap_pattern=None)
    out = D.within_clip_dropout(fr, fc, t)
    assert out["n_pre"] == 10 and out["n_challenge"] == 30
    assert abs(out["d_tex_change"]) < 0.3 and out["d_tex_maxjump"] < 0.5 and out["d_out_frac"] < 0.1


def test_flickering_clip_is_detected_on_all_three_features():
    stable = D.within_clip_dropout(*make(swap_pattern=None))
    fr, fc, t = make(swap_pattern=lambda i: i >= 10 and i % 4 in (1, 2))      # smooth <-> sharp every few frames
    flick = D.within_clip_dropout(fr, fc, t)
    assert flick["d_tex_maxjump"] > stable["d_tex_maxjump"] + 1.0
    assert flick["d_out_frac"] > stable["d_out_frac"] + 0.2
    assert flick["d_tex_change"] > stable["d_tex_change"] + 0.3                 # sharper on average after the pre window


def test_missing_features_when_too_few_frames():
    fr, fc, t = make(n=12)
    out = D.within_clip_dropout(fr, fc, t)
    assert out["n_challenge"] == 2 and all(out[k] is None for k in D.FEATURES)


def test_undetected_frames_are_skipped_and_jumps_need_adjacent_frames():
    fr, fc, t = make(swap_pattern=lambda i: i >= 10 and i % 2 == 0)
    fc = [None if (i >= 10 and i % 2 == 1) else f for i, f in enumerate(fc)]   # every other challenge frame undetected
    out = D.within_clip_dropout(fr, fc, t)
    assert out["n_challenge"] == 15 and out["d_tex_maxjump"] is None            # no adjacent detected pairs
