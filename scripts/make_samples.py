"""Build frontend/samples/ (the judges' one-click demo sets) and manifest.json.

  python scripts/make_samples.py                         # synthetic samples (placeholder)
  python scripts/make_samples.py --from data/real/p01 --title "Real caller (p01)" \
                                 --from data/swapped/p01 --title "Face swap (p01)"

Only use recordings of participants who consented to public use (docs/DATA_COLLECTION.md).
Clips are re-encoded to <=640 px, <=10 s so the demo stays small.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "frontend" / "samples"
CHS = ("passive", "head_turn", "hand_pass", "light_change")


def reencode(src: Path, dst: Path, max_w=640, max_s=10.0):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    fps = fps if 1 <= fps <= 60 else 15
    vw = None
    n = 0
    while n < max_s * fps:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        if w > max_w:
            f = cv2.resize(f, (max_w, int(h * max_w / w) // 2 * 2), interpolation=cv2.INTER_AREA)
        if vw is None:
            vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (f.shape[1], f.shape[0]))
        vw.write(f)
        n += 1
    cap.release()
    if vw:
        vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", action="append", default=[])
    ap.add_argument("--title", action="append", default=[])
    a = ap.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    samples = []
    if a.src:
        for i, src in enumerate(a.src):
            sid = f"sample{i + 1}"
            title = a.title[i] if i < len(a.title) else sid
            clips = {}
            for ch in CHS:
                for ext in (".webm", ".mp4", ".mov", ".mkv"):
                    p = Path(src) / f"{ch}{ext}"
                    if p.exists():
                        dst = OUT / f"{sid}_{ch}.mp4"
                        reencode(p, dst)
                        clips[ch] = f"samples/{dst.name}"
                        break
            samples.append({"id": sid, "title": title, "clips": clips,
                            "description": "Recorded with consent for the Unmask evaluation."})
    else:
        from unmask import synth
        for sid, title, swapped, pid in (("synthetic-real", "Synthetic caller: real", False, 4),
                                         ("synthetic-swap", "Synthetic caller: swap-like", True, 4)):
            p = synth.person(pid)
            clips = {}
            for ch in CHS:
                frames, fps, _ = synth.render_clip(p, ch, swapped)
                dst = OUT / f"{sid}_{ch}.mp4"
                synth.write_video(str(dst), frames, fps)
                clips[ch] = f"samples/{dst.name}"
            samples.append({"id": sid, "title": title, "clips": clips,
                            "description": "Procedurally drawn cartoon (no real person). Pipeline demo only."})
    note = None if a.src else "Synthetic cartoons until consented real samples are added."
    (OUT / "manifest.json").write_text(json.dumps({"samples": samples, "note": note}, indent=2))
    print(f"wrote {len(samples)} sample sets to {OUT}")


if __name__ == "__main__":
    main()
