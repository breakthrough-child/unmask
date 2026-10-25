"""analyze_clip(): run every OpenCV signal on one clip and return a JSON-able result."""
from __future__ import annotations

import time

import cv2

from . import signals as S
from .evidence import pick_evidence
from .face import FaceDetector, track_faces
from .quality import INSTRUCTIONS, gate
from .video import Clip, load_clip

CHALLENGES = ("passive", "head_turn", "hand_pass", "light_change")

_detector: FaceDetector | None = None


def get_detector() -> FaceDetector:
    global _detector
    if _detector is None:
        _detector = FaceDetector()
    return _detector


def _round(d: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if not k.startswith("_")}


def analyze_clip(source: str | Clip, challenge: str = "passive", with_evidence: bool = True,
                 detector: FaceDetector | None = None) -> dict:
    """Analyse a clip recorded in response to `challenge`.

    Returns {challenge, usable, reason, instruction, features, series, quality,
    timing_ms, evidence: [(label, jpeg_bytes)]}.
    """
    if challenge not in CHALLENGES:
        raise ValueError(f"unknown challenge {challenge!r}")
    t0 = time.perf_counter()
    clip = load_clip(source) if isinstance(source, str) else source
    t_load = time.perf_counter()
    result: dict = {"challenge": challenge, "usable": False, "reason": None, "instruction": None,
                    "features": {}, "series": {}, "quality": {}, "evidence": [],
                    "opencv_version": cv2.__version__}
    if len(clip.frames) < 2:
        result.update(reason="too_few_frames", instruction=INSTRUCTIONS["too_few_frames"],
                      quality={"frames": len(clip.frames)})
        result["timing_ms"] = {"total": round((time.perf_counter() - t0) * 1000, 1)}
        return result
    size = (clip.frames[0].shape[1], clip.frames[0].shape[0])

    faces = track_faces(detector or get_detector(), clip.frames)
    t_det = time.perf_counter()

    feats: dict = {}
    det = S.detection_stats(faces)
    det["det_flips_per_s"] = det["_flips"] / clip.duration if clip.duration > 0 else None
    feats.update(det)
    feats["landmark_jitter"] = S.landmark_jitter(faces)

    yaws, errs = S.pose_series(faces, size)
    feats["reproj_err"] = S._nanstat(lambda x: sorted(x)[len(x) // 2], errs)
    seam, seam_series = S.seam_features(clip.frames, faces)
    feats.update(seam)
    feats.update(S.texture_features(clip.frames, faces))

    masks = None
    occ_frac: list[float] = []
    light: dict = {}
    if challenge == "head_turn":
        feats.update(S.head_turn_features(faces, yaws, errs, seam_series))
    if challenge == "hand_pass":
        masks = S.motion_masks(clip.frames)
        occ, occ_frac = S.occlusion_features(clip.frames, faces, masks)
        feats.update(occ)
    if challenge == "light_change":
        light = S.lighting_series(clip.frames, faces)
        feats.update(S.lighting_features(light))
    if challenge != "passive":
        feats["response_latency_s"] = S.response_onset(challenge, clip.times, yaws, occ_frac, light)
    t_sig = time.perf_counter()

    usable, reason, quality = gate(clip, faces, challenge, feats)
    result.update(usable=usable, reason=reason, instruction=INSTRUCTIONS.get(reason) if reason else None,
                  quality=quality, features=_round(feats))
    result["series"] = {
        "t": [round(t, 3) for t in clip.times],
        "det_score": [round(fc.score, 3) if fc is not None else 0.0 for fc in faces],
        "yaw": [round(y, 1) if y is not None else None for y in yaws],
        "seam": [round(s, 3) if s is not None else None for s in seam_series],
    }
    if occ_frac:
        result["series"]["occlusion"] = [round(o, 3) for o in occ_frac]
    if light:
        result["series"]["light_face"] = [round(v, 1) if v is not None else None for v in light["face"]]
        result["series"]["light_neck"] = [round(v, 1) if v is not None else None for v in light["neck"]]
    if with_evidence:
        result["evidence"] = pick_evidence(clip, faces, challenge, seam_series, yaws, occ_frac,
                                           masks or [None] * len(faces))
    t_end = time.perf_counter()
    result["timing_ms"] = {
        "decode": round((t_load - t0) * 1000, 1),
        "detect": round((t_det - t_load) * 1000, 1),
        "signals": round((t_sig - t_det) * 1000, 1),
        "evidence": round((t_end - t_sig) * 1000, 1),
        "total": round((t_end - t0) * 1000, 1),
        "frames": len(clip.frames),
    }
    return result
