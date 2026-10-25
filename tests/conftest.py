import numpy as np
import pytest

from unmask import synth
from unmask.vision.face import Face
from unmask.vision.video import clip_from_frames


class TruthDetector:
    """Returns the ground-truth face drawn by unmask.synth, one frame at a time.

    For hand_pass it imitates what a detector does: a real face hidden by a hand
    is lost, while a swapped face that renders over the hand stays 'visible'."""

    def __init__(self, truth, swapped=False):
        self.faces = truth["faces"]
        self.swapped = swapped
        self.i = 0

    def detect(self, frame):
        t = self.faces[min(self.i, len(self.faces) - 1)]
        self.i += 1
        x, y, w, h = t["box"]
        if t["hand_x"] is not None and not self.swapped and abs(t["hand_x"] - (x + w / 2)) < w * 0.45:
            return []
        return [Face(box=np.array([x, y, w, h], float), landmarks=np.array(t["landmarks"], float), score=0.9)]


def make(pid: int, challenge: str, swapped: bool):
    p = synth.person(pid)
    frames, fps, truth = synth.render_clip(p, challenge, swapped)
    return clip_from_frames(frames, fps), TruthDetector(truth, swapped)


@pytest.fixture(scope="session")
def synth_clip():
    cache = {}

    def get(pid, challenge, swapped):
        key = (pid, challenge, swapped)
        if key not in cache:
            p = synth.person(pid)
            cache[key] = synth.render_clip(p, challenge, swapped)
        frames, fps, truth = cache[key]
        return clip_from_frames(frames, fps), TruthDetector(truth, swapped)
    return get
