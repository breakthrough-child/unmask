"""Clip loading: decode, downscale to <=640px wide and resample to ~10 fps."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MAX_WIDTH = 640
TARGET_FPS = 10.0
MAX_SECONDS = 15.0


@dataclass
class Clip:
    frames: list[np.ndarray]          # BGR uint8, downscaled
    times: list[float]                # seconds from clip start
    source_fps: float
    source_size: tuple[int, int]      # (w, h) before downscale
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.times[-1] - self.times[0] if len(self.times) > 1 else 0.0

    @property
    def fps(self) -> float:
        return (len(self.times) - 1) / self.duration if self.duration > 0 else 0.0


def _downscale(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def load_clip(path: str, max_width: int = MAX_WIDTH, target_fps: float = TARGET_FPS,
              max_seconds: float = MAX_SECONDS) -> Clip:
    """Decode a video file and keep frames spaced ~1/target_fps apart.

    Browser MediaRecorder WebM files often report a bogus or zero FPS, so frame
    timing uses CAP_PROP_POS_MSEC when it is monotonic and falls back to the
    container FPS (or 30) otherwise.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not (1.0 <= src_fps <= 240.0):
        src_fps = 30.0
    frames, times = [], []
    size = (0, 0)
    next_t = 0.0
    prev_t = -1.0 / src_fps
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        ms = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
        # Trust the decoder timestamp only while it advances; else step by FPS.
        t = ms / 1000.0 if ms / 1000.0 > prev_t + 1e-4 else prev_t + 1.0 / src_fps
        prev_t = t
        if t > max_seconds:
            break
        if t + 1e-6 >= next_t:
            size = (frame.shape[1], frame.shape[0])
            frames.append(_downscale(frame, max_width))
            times.append(t)
            next_t = t + 1.0 / target_fps
    cap.release()
    if times:
        t0 = times[0]
        times = [t - t0 for t in times]
    return Clip(frames=frames, times=times, source_fps=src_fps, source_size=size)


def clip_from_frames(frames: list[np.ndarray], fps: float, max_width: int = MAX_WIDTH,
                     target_fps: float = TARGET_FPS) -> Clip:
    """Build a Clip from in-memory frames (tests, synthetic fixtures)."""
    step = max(1, int(round(fps / target_fps)))
    kept = frames[::step]
    times = [i * step / fps for i in range(len(kept))]
    size = (frames[0].shape[1], frames[0].shape[0]) if frames else (0, 0)
    return Clip(frames=[_downscale(f, max_width) for f in kept], times=times,
                source_fps=fps, source_size=size)
