"""Geometry helpers: face-aligned crops and the ring/band masks used by the signals."""
from __future__ import annotations

import cv2
import numpy as np

CROP = 128          # side of the face-aligned crop
CROP_MARGIN = 1.5   # crop side = CROP_MARGIN * max(box w, h)

# Elliptical radii (as a fraction of the face ellipse) for the rings.
INNER = (0.0, 0.70)        # face interior used for texture / lighting
RING_IN = (0.62, 0.82)     # skin just inside the seam band
BAND = (0.82, 1.02)        # where a swap's blend boundary usually sits
RING_OUT = (1.02, 1.22)    # just outside the face (hair / ears / background)


def crop_face(frame: np.ndarray, box: np.ndarray, size: int = CROP) -> np.ndarray:
    """Square crop centred on the face box, resized to size x size (border replicated)."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    side = CROP_MARGIN * max(w, h)
    scale = size / side
    m = np.array([[scale, 0, size / 2 - cx * scale], [0, scale, size / 2 - cy * scale]], dtype=np.float32)
    return cv2.warpAffine(frame, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


_ring_cache: dict = {}


def ellipse_radius_map(box_aspect: float, size: int = CROP) -> np.ndarray:
    """Normalised elliptical radius for each crop pixel (1.0 = face ellipse edge)."""
    key = (round(box_aspect, 2), size)
    if key not in _ring_cache:
        side_units = CROP_MARGIN  # crop side in units of max(w, h)
        a = 0.5 / side_units * size            # semi-axis along the larger box dim
        ax, ay = (a / box_aspect, a) if box_aspect >= 1 else (a, a * box_aspect)
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        r = np.sqrt(((xx - size / 2) / ax) ** 2 + ((yy - size / 2) / ay) ** 2)
        _ring_cache[key] = r
    return _ring_cache[key]


def ring_mask(box: np.ndarray, rng: tuple[float, float], size: int = CROP) -> np.ndarray:
    aspect = float(box[3] / max(box[2], 1e-3))  # h / w
    r = ellipse_radius_map(aspect, size)
    return (r >= rng[0]) & (r < rng[1])


def frame_masks(shape: tuple[int, int], box: np.ndarray) -> dict[str, np.ndarray]:
    """Full-frame boolean masks: face interior, neck, background."""
    h, w = shape
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2
    face = np.zeros((h, w), np.uint8)
    cv2.ellipse(face, (int(cx), int(cy)), (max(1, int(bw * 0.35)), max(1, int(bh * 0.35))), 0, 0, 360, 1, -1)
    excl = np.zeros((h, w), np.uint8)
    cv2.ellipse(excl, (int(cx), int(cy)), (max(1, int(bw * 0.9)), max(1, int(bh * 1.0))), 0, 0, 360, 1, -1)
    neck = np.zeros((h, w), np.uint8)
    nx0, nx1 = int(max(0, cx - 0.3 * bw)), int(min(w, cx + 0.3 * bw))
    ny0, ny1 = int(min(h, y + 1.08 * bh)), int(min(h, y + 1.5 * bh))
    neck[ny0:ny1, nx0:nx1] = 1
    bg = (1 - excl).astype(bool) & ~neck.astype(bool)
    # Ignore the region directly below the neck (body) for the background reference.
    bg[ny0:, nx0:nx1] = False
    return {"face": face.astype(bool), "neck": neck.astype(bool) & ~face.astype(bool), "bg": bg}
