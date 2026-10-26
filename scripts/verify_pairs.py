"""Verify data/swapped/* clips against their data/real/* originals (no screenshots, numbers only).

  python scripts/verify_pairs.py [--data data] [--every 5] [--json out.json]

Per pair checks:
  1. both files open in OpenCV and decode to the same number of frames
  2. same size and fps; swapped is H.264 yuv420p (ffprobe)
  3. the FACE region clearly differs: on frames where YuNet finds a face in the original,
     the face crop's mean absolute difference (MAD) is compared with
       - the BACKGROUND MAD (outside an enlarged face box), which should stay small, and
       - a re-encode baseline: the original re-encoded through the same H.264 settings,
         which tells us how much difference plain compression noise explains.
     A frame "clearly differs" when face MAD >= RATIO x max(background MAD, re-encode face MAD, 1.0)
     and >= ABS_FLOOR gray levels. The bar is relative to the clip's own noise floor, so it does not
     depend on how bright the footage is (a fixed absolute MAD wrongly failed darker sessions).
     A clip PASSES when at least MIN_FRAC of its face frames clearly differ (default: the median frame).
     The fraction is always reported: frames where a hand hides the face legitimately differ less.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from unmask.vision.face import FaceDetector  # noqa: E402

ABS_FLOOR = 3.0        # gray levels (0-255): guards the ratio when the noise floor is ~0
RATIO = 3.0
MIN_FRAC = 0.5
CHALLENGES = ("passive", "head_turn", "hand_pass", "light_change")


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"codec": s["codec_name"], "pix_fmt": s["pix_fmt"], "w": s["width"], "h": s["height"],
            "fps": float(num) / float(den), "frames": int(s["nb_read_frames"])}


def read_all(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {path}")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def reencode(src: Path, dst: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p", str(dst)], check=True)


def gray(f: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)


def region_mads(a: np.ndarray, b: np.ndarray, box: np.ndarray) -> tuple[float, float]:
    h, w = a.shape[:2]
    x, y, bw, bh = box
    x0, y0, x1, y1 = (int(max(0, x)), int(max(0, y)), int(min(w, x + bw)), int(min(h, y + bh)))
    d = np.abs(gray(a) - gray(b))
    face = float(d[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else float("nan")
    mask = np.ones((h, w), bool)
    pad_x, pad_y = int(0.6 * bw), int(0.6 * bh)   # exclude hair/ears/neck around the face too
    mask[max(0, y0 - pad_y):min(h, y1 + pad_y), max(0, x0 - pad_x):min(w, x1 + pad_x)] = False
    bg = float(d[mask].mean()) if mask.any() else float("nan")
    return face, bg


def verify_pair(real: Path, swapped: Path, det: FaceDetector, every: int) -> dict:
    res: dict = {"clip": f"{real.parent.name}/{real.name}", "ok": True, "problems": []}
    pr, ps = probe(real), probe(swapped)
    res.update(real=pr, swapped=ps)
    fr, fs = read_all(real), read_all(swapped)
    res["decoded"] = {"real": len(fr), "swapped": len(fs)}
    if len(fr) != len(fs) or pr["frames"] != ps["frames"]:
        res["problems"].append(f"frame count differs: real {len(fr)} vs swapped {len(fs)}")
    if (ps["w"], ps["h"]) != (pr["w"], pr["h"]):
        res["problems"].append(f"size differs: {pr['w']}x{pr['h']} vs {ps['w']}x{ps['h']}")
    if abs(ps["fps"] - pr["fps"]) > 1e-3:
        res["problems"].append(f"fps differs: {pr['fps']} vs {ps['fps']}")
    if ps["codec"] != "h264" or ps["pix_fmt"] != "yuv420p":
        res["problems"].append(f"swapped is {ps['codec']}/{ps['pix_fmt']}, expected h264/yuv420p")

    with tempfile.TemporaryDirectory() as t:
        base = Path(t) / "re.mp4"
        reencode(real, base)
        fb = read_all(base)
    n = min(len(fr), len(fs), len(fb))
    face_d, bg_d, face_re, bg_re, used = [], [], [], [], 0
    for i in range(0, n, every):
        faces = det.detect(fr[i])
        faces = [f for f in faces if f.score >= 0.6]
        if not faces:
            continue
        box = max(faces, key=lambda f: f.box[2] * f.box[3]).box
        fd, bd = region_mads(fr[i], fs[i], box)
        fre, bre = region_mads(fr[i], fb[i], box)
        if np.isnan(fd):
            continue
        used += 1
        face_d.append(fd), bg_d.append(bd), face_re.append(fre), bg_re.append(bre)
    res["face_frames_checked"] = used
    res["face_frames_sampled"] = len(range(0, n, every))
    if used == 0:
        res["problems"].append("no face found in the original: cannot compare face regions")
    else:
        face_d, bg_d, face_re, bg_re = map(np.array, (face_d, bg_d, face_re, bg_re))
        noise = np.maximum(np.maximum(bg_d, face_re), 1.0)
        passes = (face_d >= ABS_FLOOR) & (face_d >= RATIO * noise)
        res["median_face_to_noise_ratio"] = round(float(np.median(face_d / noise)), 2)
        res["face_mad_swapped"] = round(float(np.median(face_d)), 2)
        res["background_mad_swapped"] = round(float(np.median(bg_d)), 2)
        res["face_mad_reencode_baseline"] = round(float(np.median(face_re)), 2)
        res["frac_frames_face_clearly_differs"] = round(float(passes.mean()), 3)
        if passes.mean() < MIN_FRAC:
            res["problems"].append(f"face clearly differs on only {passes.mean():.0%} of checked frames (need {MIN_FRAC:.0%})")
        elif passes.mean() < 0.9:
            res["note"] = f"face differs on {passes.mean():.0%} of frames (frames with a hand over the face change less)"
    res["ok"] = not res["problems"]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--every", type=int, default=5, help="check every Nth frame")
    ap.add_argument("--json")
    a = ap.parse_args()
    data = Path(a.data)
    det = FaceDetector()
    results, missing = [], []
    for real in sorted((data / "real").glob("*/*.mp4")):
        swapped = data / "swapped" / real.parent.name / real.name
        if not swapped.exists():
            missing.append(str(swapped.relative_to(data)))
            continue
        try:
            results.append(verify_pair(real, swapped, det, a.every))
        except Exception as e:  # noqa: BLE001
            results.append({"clip": f"{real.parent.name}/{real.name}", "ok": False, "problems": [f"{type(e).__name__}: {e}"]})
    print(f"{'clip':<32}{'frames':>14}{'fmt':>16}{'face MAD':>10}{'bg MAD':>8}{'re-enc':>8}{'x noise':>8}{'%differs':>10}  result")
    for r in results:
        if "swapped" not in r:
            print(f"{r['clip']:<32} ERROR {r['problems']}")
            continue
        print(f"{r['clip']:<32}{r['decoded']['real']:>6}/{r['decoded']['swapped']:<6}"
              f"{r['swapped']['codec'] + '/' + r['swapped']['pix_fmt']:>16}"
              f"{r.get('face_mad_swapped', float('nan')):>10}{r.get('background_mad_swapped', float('nan')):>8}"
              f"{r.get('face_mad_reencode_baseline', float('nan')):>8}"
              f"{r.get('median_face_to_noise_ratio', float('nan')):>8}"
              f"{r.get('frac_frames_face_clearly_differs', float('nan')):>10}  "
              f"{('PASS' + (' (' + r['note'] + ')' if r.get('note') else '')) if r['ok'] else 'FAIL ' + '; '.join(r['problems'])}")
    for m in missing:
        print(f"MISSING {m}")
    ok = bool(results) and not missing and all(r["ok"] for r in results)
    print(f"\n{sum(r['ok'] for r in results)}/{len(results)} pairs pass" + (f", {len(missing)} missing" if missing else ""))
    if a.json:
        Path(a.json).write_text(json.dumps({"results": results, "missing": missing}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
