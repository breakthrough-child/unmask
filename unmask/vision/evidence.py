"""Annotated evidence frames: face box, landmarks, seam band, flagged motion regions."""
from __future__ import annotations

import cv2
import numpy as np

from . import regions as R
from .signals import present

GREEN, AMBER, RED, CYAN = (80, 200, 80), (0, 180, 255), (60, 60, 230), (230, 200, 0)


def _seam_contour(box) -> list[np.ndarray]:
    """Seam band edges (r = BAND[0], BAND[1]) as full-frame ellipses."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    out = []
    for r in R.BAND:
        out.append(cv2.ellipse2Poly((int(cx), int(cy)), (int(w / 2 * r), int(h / 2 * r)), 0, 0, 360, 10))
    return out


def annotate(frame: np.ndarray, fc, label: str, motion: np.ndarray | None = None,
             color=AMBER) -> np.ndarray:
    img = frame.copy()
    if motion is not None and motion.any():
        overlay = img.copy()
        overlay[motion] = (0, 0, 255)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    if fc is not None:
        x, y, w, h = (int(v) for v in fc.box)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        for poly in _seam_contour(fc.box):
            cv2.polylines(img, [poly], True, CYAN, 1, cv2.LINE_AA)
        for (lx, ly) in fc.landmarks:
            cv2.circle(img, (int(lx), int(ly)), 3, GREEN, -1, cv2.LINE_AA)
        cv2.putText(img, f"{fc.score:.2f}", (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (20, 20, 20), -1)
    cv2.putText(img, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def pick_evidence(clip, faces, challenge: str, seam_series, yaws, occ_frac, masks, max_frames: int = 3):
    """Choose a few informative frames and return [(label, jpeg_bytes)]."""
    chosen: list[tuple[int, str, np.ndarray | None]] = []
    seam_idx = [i for i, s in enumerate(seam_series) if s is not None]
    if seam_idx:
        i = max(seam_idx, key=lambda k: seam_series[k])
        chosen.append((i, f"t={clip.times[i]:.1f}s strongest face-edge seam ({seam_series[i]:.2f})", None))
    if challenge == "head_turn":
        idx = [i for i, y in enumerate(yaws) if y is not None]
        if idx:
            i = max(idx, key=lambda k: abs(yaws[k]))
            chosen.append((i, f"t={clip.times[i]:.1f}s max head turn (yaw {yaws[i]:+.0f} deg)", None))
    if challenge == "hand_pass" and occ_frac:
        i = int(np.argmax(occ_frac))
        lab = "face still detected" if present(faces[i]) else "face hidden"
        chosen.append((i, f"t={clip.times[i]:.1f}s hand over face ({occ_frac[i]:.0%}), {lab}", masks[i]))
    if challenge == "light_change" and clip.frames:
        chosen.append((len(clip.frames) - 1, f"t={clip.times[-1]:.1f}s after lighting change", None))
    if not chosen and clip.frames:
        chosen.append((len(clip.frames) // 2, f"t={clip.times[len(clip.frames) // 2]:.1f}s", None))
    out, seen = [], set()
    for i, label, mask in chosen:
        if i in seen:
            continue
        seen.add(i)
        img = annotate(clip.frames[i], faces[i], label, mask)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            out.append((label, buf.tobytes()))
        if len(out) >= max_frames:
            break
    return out
