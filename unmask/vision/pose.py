"""Head pose (yaw / pitch) from YuNet's 5 landmarks with cv2.solvePnP."""
from __future__ import annotations

import math

import cv2
import numpy as np

# Generic 3D face model (mm), x right in the image, y down, z toward camera.
# Order matches YuNet: right eye, left eye, nose tip, right mouth, left mouth.
MODEL_POINTS = np.array([
    [-32.0, -35.0, -28.0],
    [32.0, -35.0, -28.0],
    [0.0, 0.0, 0.0],
    [-25.0, 35.0, -25.0],
    [25.0, 35.0, -25.0],
], dtype=np.float64)


def head_pose(landmarks: np.ndarray, frame_size: tuple[int, int]) -> tuple[float, float] | None:
    """Return (yaw, pitch) in degrees; positive yaw = face turned to image right."""
    w, h = frame_size
    f = float(w)
    cam = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    img = landmarks.astype(np.float64).reshape(5, 1, 2)
    ok, rvec, _tvec = cv2.solvePnP(MODEL_POINTS, img, cam, None, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None
    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, img, cam, None, rvec, _tvec,
                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rot, _ = cv2.Rodrigues(rvec)
    # Euler angles (y-axis yaw, x-axis pitch) from the rotation matrix.
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -rot[2, 0]))))
    pitch = math.degrees(math.atan2(rot[2, 1], rot[2, 2]))
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return yaw, pitch


def yaw_proxy(landmarks: np.ndarray) -> float:
    """Geometric yaw estimate (degrees) from nose offset relative to the eyes.

    Robust when PnP is ill-conditioned (5 points, near-profile). Used to
    sanity-check / blend with the PnP yaw."""
    reye, leye, nose = landmarks[0], landmarks[1], landmarks[2]
    mid = (reye + leye) / 2
    eye_dist = np.linalg.norm(leye - reye)
    if eye_dist < 1e-3:
        return 0.0
    ratio = float((nose[0] - mid[0]) / eye_dist)
    return math.degrees(math.asin(max(-1.0, min(1.0, ratio * 1.6))))

