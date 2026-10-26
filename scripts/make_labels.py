"""Write data/labels.csv exactly as docs/DATA_COLLECTION.md expects, for real AND swapped clips.

  python scripts/make_labels.py [--data data] [--owner divine]

Session folders are named <owner>_<condition> (e.g. divine_dim). Columns:
  participant,label,challenge,path,condition,via_call
`participant` is a pseudonym (p01 for --owner, then p02, p03 ... for everyone else in sorted order);
the pseudonym -> owner mapping is written to data/participants.json (git-ignored, never published).
`condition` is the folder name after the owner (daylight, dim, roomlight, call_whatsapp, ...);
via_call is "yes" when the condition starts with call_.
A swapped clip only gets a row if its real counterpart exists, and vice versa is reported.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHALLENGES = ("passive", "head_turn", "hand_pass", "light_change")
EXTS = (".webm", ".mp4", ".mov", ".mkv")


def build_rows(data: Path, owner: str) -> tuple[list[dict], dict[str, str], list[str]]:
    sessions = sorted({p.parent.name for label in ("real", "swapped") for p in (data / label).glob("*/*")
                       if p.suffix.lower() in EXTS and p.stem in CHALLENGES})
    owners = sorted({s.split("_", 1)[0] for s in sessions}, key=lambda o: (o != owner, o))
    pseudo = {o: f"p{i + 1:02d}" for i, o in enumerate(owners)}
    rows, warnings = [], []
    for label in ("real", "swapped"):
        for s in sessions:
            for ch in CHALLENGES:
                files = [p for p in (data / label / s).glob(f"{ch}.*") if p.suffix.lower() in EXTS]
                other = [p for p in (data / ("swapped" if label == "real" else "real") / s).glob(f"{ch}.*")]
                if not files:
                    if other:
                        warnings.append(f"{label}/{s}/{ch}: missing (the other side exists)")
                    continue
                o, _, cond = s.partition("_")
                rows.append({"participant": pseudo[o], "label": label, "challenge": ch,
                             "path": str(files[0].relative_to(data)), "condition": cond or "default",
                             "via_call": "yes" if cond.startswith("call_") else "no"})
    return rows, {v: k for k, v in pseudo.items()}, warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--owner", default="divine", help="session-folder prefix of the project owner (becomes p01)")
    a = ap.parse_args()
    data = Path(a.data)
    rows, mapping, warnings = build_rows(data, a.owner)
    if not rows:
        print("no clips found", file=sys.stderr)
        return 1
    with open(data / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["participant", "label", "challenge", "path", "condition", "via_call"])
        w.writeheader()
        w.writerows(rows)
    (data / "participants.json").write_text(json.dumps(mapping, indent=2))
    n_real = sum(r["label"] == "real" for r in rows)
    print(f"wrote {data / 'labels.csv'}: {len(rows)} rows ({n_real} real, {len(rows) - n_real} swapped), "
          f"{len(mapping)} participant(s): {', '.join(sorted(mapping))}")
    for w_ in warnings:
        print("WARNING", w_)
    return 0


if __name__ == "__main__":
    sys.exit(main())
