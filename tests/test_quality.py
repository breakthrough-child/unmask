import numpy as np

from unmask.vision.analyze import analyze_clip
from unmask.vision.face import Face
from unmask.vision.video import clip_from_frames


class FixedDetector:
    def __init__(self, box=(260, 160, 120, 150), score=0.9, every=1):
        self.box, self.score, self.every, self.i = box, score, every, 0

    def detect(self, frame):
        self.i += 1
        if self.box is None or (self.i % self.every):
            return []
        x, y, w, h = self.box
        lms = np.array([[x + .3 * w, y + .4 * h], [x + .7 * w, y + .4 * h], [x + .5 * w, y + .6 * h],
                        [x + .35 * w, y + .8 * h], [x + .65 * w, y + .8 * h]])
        return [Face(np.array(self.box, float), lms, self.score)]


def _frames(n=30, value=None, blur=False):
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        f = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) if value is None else np.full((480, 640, 3), value, np.uint8)
        if blur:
            import cv2
            f = cv2.GaussianBlur(np.full((480, 640, 3), 128, np.uint8), (31, 31), 10)
        out.append(f)
    return out


def _reason(frames, det, ch="passive", fps=10):
    return analyze_clip(clip_from_frames(frames, fps), ch, detector=det, with_evidence=False)["reason"]


def test_too_few_frames():
    assert _reason(_frames(5), FixedDetector()) == "too_few_frames"


def test_no_face():
    assert _reason(_frames(30), FixedDetector(box=None)) == "no_face"


def test_face_too_small():
    assert _reason(_frames(30), FixedDetector(box=(300, 200, 30, 36))) == "face_too_small"


def test_too_dark():
    assert _reason(_frames(30, value=12), FixedDetector()) == "too_dark"


def test_too_blurry():
    assert _reason(_frames(30, blur=True), FixedDetector()) == "too_blurry"


def test_challenge_not_performed():
    # a static face cannot satisfy a head-turn request
    assert _reason(_frames(30), FixedDetector(), "head_turn") == "challenge_not_performed"


def test_usable_clip_has_no_reason():
    r = analyze_clip(clip_from_frames(_frames(30), 10), "passive", detector=FixedDetector(), with_evidence=False)
    assert r["usable"] and r["reason"] is None
