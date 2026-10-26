"""Build frontend/samples/ (the judges' one-click demo sets) and manifest.json.

  python scripts/make_samples.py                        # default: the owner's daylight session, real + swapped
  python scripts/make_samples.py --session divine_dim

Privacy rule: only sessions from `divine_*` folders (the project owner's own consenting footage) may be
published as samples; anything else is refused. Clips are copied as they are when they are already
<= 640 px wide and <= 10.5 s (no extra re-encoding generation), otherwise re-encoded to H.264.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "frontend" / "samples"
CHS = ("passive", "head_turn", "hand_pass", "light_change")
PUBLIC_PREFIX = "divine_"


def probe(path: Path) -> tuple[int, float]:
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width:format=duration",
                        "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    j = json.loads(o)
    return int(j["streams"][0]["width"]), float(j["format"]["duration"])


def add(path: Path, dst: Path) -> None:
    w, dur = probe(path)
    if w <= 640 and dur <= 10.5 and path.suffix.lower() == ".mp4":
        shutil.copyfile(path, dst)
    else:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-t", "10", "-vf", "scale='min(640,iw)':-2",
                        "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="divine_daylight")
    a = ap.parse_args()
    if not a.session.startswith(PUBLIC_PREFIX):
        sys.exit(f"refusing: only `{PUBLIC_PREFIX}*` sessions may be published as samples (got {a.session!r})")
    cond = a.session.split("_", 1)[1]
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    samples = []
    for label, title, sid in (("real", f"Real caller (recorded, {cond} light)", "real"),
                              ("swapped", f"Face-swapped caller (recorded, {cond} light)", "swap")):
        src = ROOT / "data" / label / a.session
        clips = {}
        for ch in CHS:
            p = next((q for q in src.glob(f"{ch}.*")), None)
            if p is None:
                sys.exit(f"missing {src}/{ch}.*")
            dst = OUT / f"{sid}_{ch}.mp4"
            add(p, dst)
            clips[ch] = f"samples/{dst.name}"
        desc = ("Recorded with the project owner's consent." if label == "real" else
                "The same recording with the face replaced by an AI-generated face of no real person.")
        samples.append({"id": f"{sid}-{cond}", "title": title, "clips": clips, "description": desc})
    note = "Recorded sessions of the project owner; the swap uses an AI-generated face of no real person."
    (OUT / "manifest.json").write_text(json.dumps({"samples": samples, "note": note}, indent=2))
    size = sum(f.stat().st_size for f in OUT.iterdir()) / 1e6
    print(f"wrote {len(samples)} sample sets ({size:.1f} MB) to {OUT}")


if __name__ == "__main__":
    main()
