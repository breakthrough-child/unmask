"""Synthetic test fixtures: procedurally drawn 'participants' (no real person) in
real and swap-like variants for every challenge. They exist so the unit tests,
CLI, eval harness and API run end to end before real footage is recorded.

The swap-like variant imitates artefacts of real-time face swaps — a blended
boundary that flickers, a smoother face texture, landmarks that wobble, a face
that renders over an occluding hand, lighting that only partly follows the
scene, and a face that breaks up at large head turns. It is NOT a face-swap
tool: it only draws cartoons. Results on these fixtures say nothing about real
deepfake detection performance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

W, H, FPS = 640, 480, 10.0

# skin tones given as RGB (light -> dark), converted to OpenCV's BGR order below
_SKIN_RGB = [(224, 182, 158), (198, 145, 110), (160, 112, 80), (110, 74, 52), (236, 205, 184), (141, 95, 66)]
SKIN = [tuple(reversed(c)) for c in _SKIN_RGB]
BG = [(70, 90, 120), (120, 110, 90), (60, 60, 60), (150, 150, 140), (90, 120, 80), (110, 80, 100)]


@dataclass
class Person:
    pid: str
    skin: tuple
    bg: tuple
    face_w: int
    cx: int
    cy: int
    seed: int


def person(i: int) -> Person:
    rng = np.random.default_rng(1000 + i)
    return Person(pid=f"synth{i:02d}", skin=SKIN[i % len(SKIN)], bg=BG[(i * 2 + 1) % len(BG)],
                  face_w=int(rng.integers(130, 180)), cx=int(W / 2 + rng.integers(-40, 40)),
                  cy=int(H / 2 - 20 + rng.integers(-20, 20)), seed=i)


def _background(p: Person, rng) -> np.ndarray:
    img = np.zeros((H, W, 3), np.float32)
    img[:] = p.bg
    # a sharp, textured background: shelves / posters / a door frame
    r = np.random.default_rng(p.seed)
    for _ in range(14):
        x0, y0 = int(r.integers(0, W)), int(r.integers(0, H))
        col = tuple(float(c) for c in r.integers(30, 220, 3))
        cv2.rectangle(img, (x0, y0), (x0 + int(r.integers(20, 160)), y0 + int(r.integers(10, 120))), col, -1)
    tex = r.normal(0, 9, (H, W, 1)).astype(np.float32)
    return img + tex


_NOISE_BANK: list[np.ndarray] = []
_YY, _XX = np.mgrid[0:H, 0:W].astype(np.float32)


def _skin_texture(shape, seed, amp=7.0):
    """Unit-variance fine-grain noise (cached bank, so rendering stays fast), scaled by amp."""
    if not _NOISE_BANK:
        r = np.random.default_rng(12345)
        for _ in range(16):
            t = cv2.GaussianBlur(r.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.7)
            _NOISE_BANK.append((t / t.std())[..., None])
    return _NOISE_BANK[seed % len(_NOISE_BANK)] * amp


def draw_face(img, p: Person, cx, cy, fw, yaw_deg, skin, jitter=(0, 0), texture_amp=7.0, seed=0):
    """Draw a face (in place) and return its 5 landmarks."""
    fh = int(fw * 1.28)
    s = math.sin(math.radians(yaw_deg))
    ew = max(8, int(fw / 2 * (1 - 0.35 * abs(s))))
    # shade + texture only inside the face's bounding box (fast)
    x0, x1 = max(0, cx - ew - 2), min(W, cx + ew + 3)
    y0, y1 = max(0, cy - fh // 2 - 2), min(H, cy + fh // 2 + 3)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.ellipse(mask, (cx - x0, cy - y0), (ew, fh // 2), 0, 0, 360, 255, -1)
    yy, xx = _YY[y0:y1, x0:x1], _XX[y0:y1, x0:x1]
    shade = 1.0 - 0.25 * (((xx - cx) / max(ew, 1)) ** 2 + ((yy - cy) / (fh / 2)) ** 2) - 0.12 * s * (xx - cx) / max(ew, 1)
    face = np.clip(shade, 0.5, 1.1)[..., None] * np.array(skin, np.float32)
    face += _skin_texture(img.shape, seed, texture_amp)[y0:y1, x0:x1]
    np.copyto(img[y0:y1, x0:x1], face, where=(mask > 0)[..., None])
    off = s * fw * 0.33
    jx, jy = jitter
    ex = fw * 0.2 * (1 - 0.3 * abs(s))
    eye_y = cy - fh * 0.12 + jy
    reye = (cx + off - ex + jx, eye_y)
    leye = (cx + off + ex + jx, eye_y)
    nose = (cx + off * 1.35 + jx, cy + fh * 0.07 + jy)
    rm = (cx + off - fw * 0.15 + jx, cy + fh * 0.25 + jy)
    lm = (cx + off + fw * 0.15 + jx, cy + fh * 0.25 + jy)
    dark = tuple(c * 0.35 for c in skin)
    for (x, y) in (reye, leye):
        cv2.ellipse(img, (int(x), int(y)), (int(fw * 0.085), int(fw * 0.045)), 0, 0, 360, (235, 235, 235), -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), int(fw * 0.035), (40, 30, 25), -1, cv2.LINE_AA)
        cv2.ellipse(img, (int(x), int(y - fw * 0.1)), (int(fw * 0.1), int(fw * 0.025)), 0, 180, 360, dark, 3, cv2.LINE_AA)
    cv2.line(img, (int(nose[0]), int(eye_y + fw * 0.05)), (int(nose[0]), int(nose[1])), tuple(c * 0.7 for c in skin), 3, cv2.LINE_AA)
    cv2.ellipse(img, (int(nose[0]), int(nose[1])), (int(fw * 0.07), int(fw * 0.035)), 0, 0, 180, tuple(c * 0.6 for c in skin), -1, cv2.LINE_AA)
    cv2.ellipse(img, (int((rm[0] + lm[0]) / 2), int(rm[1])), (int(fw * 0.16), int(fw * 0.05)), 0, 0, 180, (60, 50, 150), -1, cv2.LINE_AA)
    # hair
    cv2.ellipse(img, (cx, int(cy - fh * 0.38)), (int(ew * 1.05), int(fh * 0.2)), 0, 180, 360, (30, 30, 35), -1, cv2.LINE_AA)
    return np.array([reye, leye, nose, rm, lm], np.float32), (cx, cy, ew, fh // 2)


def _hand(img, x, y, size, skin):
    cv2.ellipse(img, (int(x), int(y)), (int(size * 0.45), int(size * 0.7)), 10, 0, 360, tuple(c * 0.95 for c in skin), -1, cv2.LINE_AA)
    for k in range(4):
        cv2.ellipse(img, (int(x - size * 0.3 + k * size * 0.2), int(y - size * 0.75)), (int(size * 0.08), int(size * 0.3)),
                    0, 0, 360, tuple(c * 0.92 for c in skin), -1, cv2.LINE_AA)
    return _skin_texture(img.shape, int(x) % 97 + 3, 6.0)


def render_clip(p: Person, challenge: str, swapped: bool, seconds: float | None = None):
    """Return (frames BGR uint8, fps, truth dict)."""
    seconds = seconds or (8.0 if challenge == "passive" else 5.0)
    n = int(seconds * FPS)
    rng = np.random.default_rng(p.seed * 31 + sum(map(ord, challenge)) + (7 if swapped else 0))
    base_bg = _background(p, rng)
    frames, truth = [], []
    for i in range(n):
        t = i / FPS
        img = base_bg.copy()
        cx = p.cx + int(4 * math.sin(t * 1.3))
        cy = p.cy + int(3 * math.sin(t * 0.9))
        yaw = 6 * math.sin(t * 0.8)
        gain = 1.0
        if challenge == "head_turn":
            # turn to one side then the other after ~0.8 s reaction time
            u = max(0.0, t - 0.8) / (seconds - 0.8)
            yaw = 70 * math.sin(2 * math.pi * u)
        if challenge == "light_change":
            u = min(1.0, max(0.0, (t - 1.0) / 2.5))
            gain = 0.65 + 0.7 * u
        # neck / shoulders
        cv2.rectangle(img, (cx - p.face_w // 4, cy + int(p.face_w * 0.55)), (cx + p.face_w // 4, cy + int(p.face_w * 1.1)), p.skin, -1)
        cv2.ellipse(img, (cx, H + 40), (int(p.face_w * 1.4), int(p.face_w * 0.9)), 0, 180, 360, (80, 60, 140), -1)
        img += _skin_texture(img.shape, i, 4.0) * 0.5
        jitter = (0.0, 0.0)
        tex = 7.0
        if swapped:  # landmark wobble + smoother texture of the rendered face
            jitter = tuple(rng.normal(0, 2.2, 2))
            tex = 1.5
        lms, (ecx, ecy, eax, eay) = draw_face(img, p, cx, cy, p.face_w, yaw, p.skin, texture_amp=7.0, seed=i)
        if swapped:
            # the "swapped" face: smoother, slightly different tone, blended with
            # a hard-ish, frame-to-frame jittering elliptical boundary
            sw = np.zeros_like(img)
            swap_skin = tuple(c * 0.93 + 8 for c in p.skin)
            follow = 1.0 if abs(yaw) < 45 else 0.4  # breaks up at large yaw
            lms, _ = draw_face(sw, p, cx, cy, p.face_w, yaw * follow, swap_skin, jitter=jitter, texture_amp=tex, seed=i + 500)
            sw = cv2.GaussianBlur(sw, (0, 0), 1.6)
            fh = int(p.face_w * 1.28)
            s = math.sin(math.radians(yaw))
            ew = max(8, int(p.face_w / 2 * (1 - 0.35 * abs(s))))
            k = 0.86 + rng.normal(0, 0.025)
            mask = np.zeros((H, W), np.float32)
            cv2.ellipse(mask, (cx + int(rng.normal(0, 2)), cy + int(rng.normal(0, 2))), (int(ew * k), int(fh / 2 * k)), 0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), 1.2)[..., None]
            if abs(yaw) > 50 and rng.random() < 0.35:
                mask *= 0.0  # swap dropout: the real face shows through
            if challenge == "light_change":
                # face lighting only partly follows the scene
                fgain = 1.0 + 0.3 * (gain - 1.0)
                sw *= fgain / max(gain, 1e-3)
            img = img * (1 - mask) + sw * mask
        if challenge == "hand_pass":
            u = (t - 1.0) / 3.0
            if 0 <= u <= 1:
                hx = p.cx - p.face_w * 1.3 + u * p.face_w * 2.6
                hy = cy + p.face_w * 0.1
                hand_img = img.copy()
                noise = _hand(hand_img, hx, hy, p.face_w * 0.9, p.skin)
                hm = np.zeros((H, W), np.float32)
                cv2.ellipse(hm, (int(hx), int(hy)), (int(p.face_w * 0.9 * 0.45), int(p.face_w * 0.9 * 0.7)), 10, 0, 360, 1.0, -1)
                hm = hm[..., None]
                hand_img = hand_img + noise * hm
                if swapped:
                    # the swap keeps rendering the face *over* the hand
                    fm = np.zeros((H, W), np.float32)
                    fh = int(p.face_w * 1.28)
                    cv2.ellipse(fm, (cx, cy), (int(p.face_w / 2 * 0.8), int(fh / 2 * 0.8)), 0, 0, 360, 1.0, -1)
                    leak = (hm[..., 0] * fm * 0.75)[..., None]
                    img = hand_img * (1 - leak) + img * leak
                else:
                    img = hand_img
        img = img * gain
        img += _skin_texture(img.shape, i + 7, 2.0)  # sensor noise
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
        hand = None
        if challenge == "hand_pass" and 0 <= (t - 1.0) / 3.0 <= 1:
            hand = float(p.cx - p.face_w * 1.3 + (t - 1.0) / 3.0 * p.face_w * 2.6)
        truth.append({"box": (ecx - eax, ecy - eay, 2 * eax, 2 * eay), "landmarks": lms.tolist(),
                      "yaw": yaw, "hand_x": hand})
    return frames, FPS, {"faces": truth, "swapped": swapped, "participant": p.pid}


def write_video(path: str, frames, fps=FPS):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames[0].shape[1], frames[0].shape[0]))
    for f in frames:
        vw.write(f)
    vw.release()


def make_dataset(root: str, n_participants: int = 6, ext: str = "mp4") -> list[dict]:
    """Write data-layout fixtures: root/{real,swapped}/<pid>/<challenge>.mp4 + labels.csv."""
    import csv
    import os
    rows = []
    for i in range(n_participants):
        p = person(i)
        for label in ("real", "swapped"):
            d = os.path.join(root, label, p.pid)
            os.makedirs(d, exist_ok=True)
            for ch in ("passive", "head_turn", "hand_pass", "light_change"):
                path = os.path.join(d, f"{ch}.{ext}")
                if not os.path.exists(path):
                    frames, fps, _ = render_clip(p, ch, label == "swapped")
                    write_video(path, frames, fps)
                rows.append({"participant": p.pid, "label": label, "challenge": ch,
                             "path": os.path.relpath(path, root), "condition": "synthetic", "via_call": "no"})
    with open(os.path.join(root, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows
