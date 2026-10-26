"""Within-clip swap dropout (docs/PASSIVE_EXPERIMENT.md, Experiment 2).

Hypothesis: during a hand pass / head turn a swapper drops out and the original face shows through, so within a swapped clip the face texture
jumps between a smooth and a sharp look. Every quantity is measured against the SAME clip's pre-challenge frames, so between-session
offsets cancel. Uses ALL detected frames (no yaw / occlusion filter)."""
from __future__ import annotations

import cv2
import numpy as np

from . import passive as PS
from . import regions as R
from .signals import present

PRE_END_S = 1.0          # pre-challenge window: t < 1.0 s
MIN_PRE = 5
MIN_CHALLENGE = 10
OUT_Z = 3.0              # outside-appearance threshold on the RMS z-score
STD_FLOOR = 0.25         # floor on the pre-window std, as a fraction of the clip-wide std
FEATURES = ("d_tex_change", "d_tex_maxjump", "d_out_frac")


def frame_tex(gray_f32_lap: np.ndarray, masks: dict) -> float | None:
    if masks["bg"].sum() < 100 or masks["face"].sum() < 30:
        return None
    return float(np.log((gray_f32_lap[masks["face"]].var() + 1.0) / (gray_f32_lap[masks["bg"]].var() + 1.0)))


def descriptor(frame, gray, box, tex: float) -> np.ndarray | None:
    """[L, a, b of the central face, tex, log HF of the face patch]."""
    x, y, w, h = box
    s = PS.patch_size(w)
    patch = PS.crop(gray, x + w / 2, y + h / 2 + 0.10 * h, s)
    if patch is None:
        return None
    lab = cv2.cvtColor(frame.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    centre, _, _ = PS._regions_masks(frame.shape, box)
    c = PS._med_lab(lab, centre)
    if c is None:
        return None
    return np.array([*c, tex, PS._log(PS.hf_power(patch))], np.float32)


def within_clip_dropout(frames, faces, times) -> dict:
    out: dict = {k: None for k in FEATURES}
    rows = []   # (index, t, tex, descriptor|None) for every detected frame
    for i, (f, fc) in enumerate(zip(frames, faces)):
        if not present(fc):
            continue
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        tex = frame_tex(lap, R.frame_masks(gray.shape, fc.box))
        if tex is None:
            continue
        rows.append((i, float(times[i]), tex, descriptor(f, gray, fc.box, tex)))
    pre = [r for r in rows if r[1] < PRE_END_S]
    ch = [r for r in rows if r[1] >= PRE_END_S]
    out["n_pre"], out["n_challenge"] = len(pre), len(ch)
    if len(pre) < MIN_PRE or len(ch) < MIN_CHALLENGE:
        return out
    out["d_tex_change"] = float(np.median([r[2] for r in ch]) - np.median([r[2] for r in pre]))
    jumps = [abs(b[2] - a[2]) for a, b in zip(ch, ch[1:]) if b[0] == a[0] + 1]
    out["d_tex_maxjump"] = float(max(jumps)) if jumps else None
    dpre = np.array([r[3] for r in pre if r[3] is not None])
    dch = np.array([r[3] for r in ch if r[3] is not None])
    dall = np.array([r[3] for r in rows if r[3] is not None])
    if len(dpre) >= MIN_PRE and len(dch) >= MIN_CHALLENGE:
        sd = np.maximum(np.maximum(dpre.std(axis=0), STD_FLOOR * dall.std(axis=0)), 1e-6)
        z = (dch - dpre.mean(axis=0)) / sd
        out["d_out_frac"] = float(np.mean(np.sqrt(np.mean(z ** 2, axis=1)) > OUT_Z))
    return out
