"""Local CLI.

  python -m unmask.cli analyze clip.webm --challenge head_turn
  python -m unmask.cli session data/real/p01            # agent replay over a folder of clips
  python -m unmask.cli synth fixtures/generated          # write synthetic fixtures
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .agent.policy import CHALLENGE_TYPES
from .agent.session import replay
from .scoring import load_config
from .vision import analyze_clip

EXTS = (".webm", ".mp4", ".mov", ".mkv", ".avi")


def find_clips(folder: str) -> dict[str, str]:
    out = {}
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() in EXTS and p.stem in ("passive",) + CHALLENGE_TYPES:
            out[p.stem] = str(p)
    return out


def _strip(res: dict, outdir: str | None, prefix: str) -> dict:
    ev = res.pop("evidence", [])
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        res["evidence"] = []
        for i, (label, jpg) in enumerate(ev):
            path = os.path.join(outdir, f"{prefix}_{i}.jpg")
            Path(path).write_bytes(jpg)
            res["evidence"].append({"label": label, "path": path})
    return res


def cmd_analyze(a):
    res = analyze_clip(a.path, a.challenge)
    res = _strip(res, a.evidence_dir, Path(a.path).stem)
    if not a.series:
        res.pop("series", None)
    print(json.dumps(res, indent=2))


def cmd_session(a):
    clips = find_clips(a.folder)
    if not clips:
        sys.exit(f"no clips named passive/head_turn/hand_pass/light_change in {a.folder}")
    cfg = load_config(a.config)
    evid: dict[str, list] = {}

    def analyze(ch):
        r = analyze_clip(clips[ch], ch)
        r = _strip(r, a.evidence_dir, ch)
        evid[ch] = r.get("evidence", [])
        return r

    s = replay(clips, analyze, cfg, use_model=a.bedrock)
    if a.trace:
        Path(a.trace).write_text(json.dumps(s, indent=2, default=str))
    for st in s["trace"]:
        ev, d = st["event"], st["decision"]
        what = f"{ev['type']}:{ev.get('challenge')}" + ("" if ev.get("usable", True) else f" UNUSABLE({ev.get('reason')})")
        act = d["action"] + (f"({d['challenge']})" if d.get("challenge") else "")
        print(f"[{st['step']}] saw {what:<34} risk={st['risk']}  -> {act:<28} [{d.get('policy')}] {d['reason']}")
    r = s["result"]
    print(f"\nVERDICT: {r['verdict']}  (risk {r['risk']}, challenges {r['challenges_done']})")
    for t in r["top_reasons"]:
        print(f"  - {t}")
    print(f"ADVICE: {r['advice']}")


def cmd_synth(a):
    from .synth import make_dataset
    rows = make_dataset(a.root, a.n)
    print(f"wrote {len(rows)} clips to {a.root}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="unmask")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("analyze", help="analyse one clip")
    p.add_argument("path")
    p.add_argument("--challenge", default="passive", choices=("passive",) + CHALLENGE_TYPES)
    p.add_argument("--evidence-dir")
    p.add_argument("--series", action="store_true")
    p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("session", help="run the agent over a folder of challenge clips")
    p.add_argument("folder")
    p.add_argument("--bedrock", action="store_true", help="use Bedrock for decisions (falls back to rules)")
    p.add_argument("--config")
    p.add_argument("--trace", help="write the full session + trace JSON here")
    p.add_argument("--evidence-dir")
    p.set_defaults(fn=cmd_session)
    p = sub.add_parser("synth", help="generate synthetic fixtures")
    p.add_argument("root")
    p.add_argument("-n", type=int, default=6)
    p.set_defaults(fn=cmd_synth)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
