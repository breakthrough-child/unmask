"""YuNet face detection (cv2.FaceDetectorYN) and simple single-face tracking."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_FILE = "face_detection_yunet_2023mar.onnx"
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
# Low threshold so we can observe confidence dropping (a signal in itself);
# frames below FACE_MIN_SCORE are treated as "no face" downstream.
DETECT_THRESHOLD = 0.3
FACE_MIN_SCORE = 0.6

# YuNet landmark order (image coordinates): right eye, left eye, nose tip,
# right mouth corner, left mouth corner ("right" = the subject's right).
LM_REYE, LM_LEYE, LM_NOSE, LM_RMOUTH, LM_LMOUTH = range(5)


def model_path() -> str:
    env = os.environ.get("UNMASK_YUNET_MODEL")
    if env:
        return env
    here = Path(__file__).resolve()
    for base in (here.parents[2], Path.cwd(), Path("/opt")):
        p = base / "models" / MODEL_FILE
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"YuNet model {MODEL_FILE} not found; run `make setup`")


def verify_model(path: str | None = None) -> bool:
    data = Path(path or model_path()).read_bytes()
    return hashlib.sha256(data).hexdigest() == MODEL_SHA256


@dataclass
class Face:
    box: np.ndarray        # x, y, w, h (float)
    landmarks: np.ndarray  # 5 x 2
    score: float

    @property
    def center(self) -> np.ndarray:
        x, y, w, h = self.box
        return np.array([x + w / 2, y + h / 2])

    @property
    def width(self) -> float:
        return float(self.box[2])


class FaceDetector:
    """Thin wrapper around cv2.FaceDetectorYN that resizes its input size per frame."""

    def __init__(self, path: str | None = None, score_threshold: float = DETECT_THRESHOLD):
        self._det = cv2.FaceDetectorYN.create(path or model_path(), "", (320, 320),
                                              score_threshold, 0.3, 50)
        self._size = (320, 320)

    def detect(self, frame: np.ndarray) -> list[Face]:
        h, w = frame.shape[:2]
        if (w, h) != self._size:
            self._det.setInputSize((w, h))
            self._size = (w, h)
        _, out = self._det.detect(frame)
        if out is None:
            return []
        faces = []
        for row in out:
            faces.append(Face(box=row[0:4].astype(float),
                              landmarks=row[4:14].reshape(5, 2).astype(float),
                              score=float(row[14])))
        return faces


def track_faces(detector: FaceDetector, frames: list[np.ndarray]) -> list[Face | None]:
    """Detect per frame and keep one face: the one nearest the previous track
    (or the largest when there is no track yet)."""
    track: list[Face | None] = []
    prev: Face | None = None
    for f in frames:
        faces = detector.detect(f)
        if not faces:
            track.append(None)
            continue
        if prev is None:
            best = max(faces, key=lambda fc: fc.box[2] * fc.box[3])
        else:
            best = min(faces, key=lambda fc: np.linalg.norm(fc.center - prev.center) / max(prev.width, 1))
            if np.linalg.norm(best.center - prev.center) > 1.5 * prev.width:
                best = max(faces, key=lambda fc: fc.box[2] * fc.box[3])
        track.append(best)
        prev = best
    return track
