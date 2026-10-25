"""Quality gates: decide whether a clip is usable and, if not, say exactly why."""
from __future__ import annotations

import cv2
import numpy as np

from .signals import present

MIN_FRAMES = 15
MIN_FPS = 5.0
MIN_FACE_PRESENT = 0.5
MIN_FACE_WIDTH = 60      # px at <=640 wide
MIN_BRIGHTNESS = 45      # mean Lab L (0-255) on the face / frame
MIN_FACE_SHARPNESS = 8.0  # Laplacian variance on the face crop

# challenge compliance: the requested action must actually be visible
MIN_YAW_RANGE = 25.0
MIN_OCCLUSION = 0.25
MIN_LIGHT_RANGE = 6.0

INSTRUCTIONS = {
    "too_few_frames": "The clip was too short or choppy. Keep the call window shared and steady for the whole recording.",
    "low_fps": "The video is updating too slowly. Close other heavy tabs or ask the caller to improve their connection.",
    "no_face": "No face was visible in most of the clip. Ask the caller to face the camera, and make sure their video tile is visible in the shared window.",
    "face_too_small": "The caller's face is too small. Enlarge their video tile (full screen or speaker view) before the next check.",
    "too_dark": "The caller's video is too dark. Ask them to switch on a light or face a window.",
    "too_blurry": "The caller's video is too blurry. Ask them to hold the camera still and wait for the connection to sharpen.",
    "challenge_not_performed": "The requested action wasn't visible. Repeat the instruction clearly and record again once they start.",
}


def gate(clip, faces, challenge: str, feats: dict) -> tuple[bool, str | None, dict]:
    """Return (usable, reason, metrics)."""
    n = len(clip.frames)
    pres = [present(fc) for fc in faces]
    metrics: dict = {"frames": n, "fps": round(clip.fps, 2)}
    if n < MIN_FRAMES:
        return False, "too_few_frames", metrics
    if clip.fps < MIN_FPS:
        return False, "low_fps", metrics
    frac = float(np.mean(pres)) if pres else 0.0
    metrics["face_present_frac"] = round(frac, 3)
    # hand-pass and head-turn legitimately hide the face for part of the clip
    need = MIN_FACE_PRESENT * (0.6 if challenge in ("hand_pass", "head_turn") else 1.0)
    if frac < need:
        return False, "no_face", metrics
    widths = [fc.width for fc, p in zip(faces, pres) if p]
    metrics["face_width"] = float(np.median(widths))
    if metrics["face_width"] < MIN_FACE_WIDTH:
        return False, "face_too_small", metrics
    bright, sharp = [], []
    for f, fc, p in list(zip(clip.frames, faces, pres))[::3]:
        if not p:
            continue
        x, y, w, h = (int(v) for v in fc.box)
        roi = f[max(0, y):y + h, max(0, x):x + w]
        if roi.size == 0:
            continue
        bright.append(float(cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)[..., 0].mean()))
        g = cv2.cvtColor(cv2.resize(roi, (128, 128)), cv2.COLOR_BGR2GRAY)
        sharp.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
    metrics["face_brightness"] = float(np.median(bright)) if bright else 0.0
    metrics["face_sharpness"] = float(np.median(sharp)) if sharp else 0.0
    if metrics["face_brightness"] < MIN_BRIGHTNESS:
        return False, "too_dark", metrics
    if metrics["face_sharpness"] < MIN_FACE_SHARPNESS:
        return False, "too_blurry", metrics
    if challenge == "head_turn" and (feats.get("yaw_range") or 0) < MIN_YAW_RANGE:
        return False, "challenge_not_performed", metrics
    if challenge == "hand_pass" and (feats.get("occ_max_frac") or 0) < MIN_OCCLUSION:
        return False, "challenge_not_performed", metrics
    if challenge == "light_change" and (feats.get("light_ref_range") or 0) < MIN_LIGHT_RANGE:
        return False, "challenge_not_performed", metrics
    return True, None, metrics
