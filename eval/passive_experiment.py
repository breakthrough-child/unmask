"""Passive-detection experiment (protocol: docs/PASSIVE_EXPERIMENT.md, fixed before computing anything).

  python eval/passive_experiment.py [--data data]

Computes the passive features on all 24 clips, evaluates feature sets T / B / N / C leave-one-session-out at clip level,
reports per-feature paired separation, face-independent controls, and the pre-registered success verdict.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from unmask.scoring.scorer import feature_score  # noqa: E402
from unmask.scoring.tune import fit_config  # noqa: E402
from unmask.vision import passive as P  # noqa: E402

spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
RE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RE)

BASELINE = ["seam_edge", "seam_flicker", "seam_edge_cv", "tex_lap_ratio", "tex_hf_ratio", "landmark_jitter", "det_score_std"]
NEW = list(P.FEATURES)
SETS = {"T": ["tex_lap_ratio"], "B": BASELINE, "N": NEW, "C": BASELINE + NEW}
HYP = {"tex_lap_ratio": -1, "p_res_pyr_neck": -1, "p_res_pyr_surround": -1, "p_res_fft_neck": -1,
       "p_noise_face_vs_neck": -1, "p_noise_face_vs_bg": -1, "p_skin_de_neck": 1, "p_skin_de_neck_ab": 1,
       "p_skin_de_forehead_ab": 1, "p_drop_sep": 1, "p_drop_switch_rate": 1}   # pre-registered expected sign of (swapped - real)
CONTROLS = list(P.CONTROLS)
THRESHOLD = 0.5
CHALLENGES = ("passive", "head_turn", "hand_pass", "light_change")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fit_fold(train: list[dict], feats: list[str]) -> dict:
    """Existing fitting rule (tune.fit_config); constant features in the training data are dropped."""
    usable = []
    for f in feats:
        vals = [c["features"].get(f) for c in train]
        vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
        if len(vals) >= 4 and np.std(vals) > 0:
            usable.append(f)
    skeleton = {"version": "x", "signals": {"s": {"weight": 1.0, "features": {
        f: {"center": 0.0, "scale": 1.0, "direction": 1, "weight": 1.0} for f in usable}}}}
    cfg = fit_config(train, skeleton, version="fold")
    spec = {f: v for f, v in cfg["signals"]["s"]["features"].items() if cfg["fit_report"][f]["status"] == "fitted"}
    return spec


def clip_score(spec: dict, features: dict) -> float | None:
    num = den = 0.0
    for f, sp in spec.items():
        x = features.get(f)
        if isinstance(x, (int, float)) and math.isfinite(x):
            w = sp.get("weight", 1.0)
            num += w * feature_score(x, sp)
            den += w
    return None if den == 0 else num / den


def evaluate(clips: list[dict], feats: list[str]) -> dict:
    sessions = sorted({c["condition"] for c in clips})
    rows = []
    for held in sessions:
        train = [c for c in clips if c["condition"] != held]
        spec = fit_fold(train, feats)
        for c in clips:
            if c["condition"] != held:
                continue
            s = clip_score(spec, c["features"])
            rows.append({"session": held, "label": c["label"], "challenge": c["challenge"], "score": s,
                         "flag": None if s is None else s >= THRESHOLD, "n_feats": len(spec)})
    return summarise(rows, sessions)


def summarise(rows: list[dict], sessions: list[str]) -> dict:
    ok = [r for r in rows if r["flag"] is not None]
    sw = [r for r in ok if r["label"] == "swapped"]
    rl = [r for r in ok if r["label"] == "real"]
    tp, fp = sum(r["flag"] for r in sw), sum(r["flag"] for r in rl)
    det, far = (tp / len(sw) if sw else float("nan")), (fp / len(rl) if rl else float("nan"))
    per = {}
    for s in sessions:
        a = [r for r in ok if r["session"] == s and r["label"] == "swapped"]
        b = [r for r in ok if r["session"] == s and r["label"] == "real"]
        d_ = sum(r["flag"] for r in a) / len(a) if a else float("nan")
        f_ = sum(r["flag"] for r in b) / len(b) if b else float("nan")
        per[s] = {"detection": d_, "false_alarm": f_, "balanced_accuracy": (d_ + 1 - f_) / 2}
    return {"detection": det, "detection_ci": wilson(tp, len(sw)), "n_swapped": len(sw), "tp": tp,
            "false_alarm": far, "false_alarm_ci": wilson(fp, len(rl)), "n_real": len(rl), "fp": fp,
            "balanced_accuracy": (det + 1 - far) / 2, "per_session": per,
            "n_missing": len(rows) - len(ok)}


def auc(real: list[float], swapped: list[float]) -> float:
    if not real or not swapped:
        return float("nan")
    wins = sum((s > r) + 0.5 * (s == r) for s in swapped for r in real)
    return wins / (len(real) * len(swapped))


def paired(clips: list[dict], feat: str) -> dict:
    by = {(c["condition"], c["challenge"], c["label"]): c["features"].get(feat) for c in clips}
    diffs, per_sess = [], {}
    for (cond, ch, lab), v in by.items():
        if lab != "real":
            continue
        s = by.get((cond, ch, "swapped"))
        if isinstance(v, (int, float)) and isinstance(s, (int, float)) and math.isfinite(v) and math.isfinite(s):
            diffs.append(s - v)
            per_sess.setdefault(cond, []).append(s - v)
    present = [c for c in clips if isinstance(c["features"].get(feat), (int, float))]
    real = [c["features"][feat] for c in present if c["label"] == "real"]
    swp = [c["features"][feat] for c in present if c["label"] == "swapped"]
    n = len(diffs)
    pos = sum(d > 0 for d in diffs)
    neg = sum(d < 0 for d in diffs)
    maj = 1 if pos >= neg else -1
    sess_signs = [np.sign(np.median(v)) for v in per_sess.values()]
    return {"coverage": len(present), "n_pairs": n, "median_d": float(np.median(diffs)) if diffs else float("nan"),
            "pos": pos, "neg": neg, "majority_sign": maj, "consistent_pairs": max(pos, neg),
            "sessions_with_majority_sign": int(sum(s == maj for s in sess_signs)), "n_sessions": len(sess_signs),
            "median_real": float(np.median(real)) if real else float("nan"),
            "median_swapped": float(np.median(swp)) if swp else float("nan"),
            "auc_pooled": auc(real, swp), "hypothesised_sign": HYP.get(feat)}


def fmt_ci(v: float, ci) -> str:
    return f"{100 * v:.0f}% [{100 * ci[0]:.0f}–{100 * ci[1]:.0f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "results"))
    a = ap.parse_args()
    data = Path(a.data)
    work = ROOT / "eval" / "out" / "passive"
    clips = RE.analyse_all(RE.read_labels(data), work / "cache", work / "evidence")
    assert len(clips) == 24 and all(c["usable"] for c in clips), "expected 24 usable clips"

    feats_all = BASELINE + NEW + CONTROLS
    per_feature = {f: paired(clips, f) for f in feats_all}
    single = {f: evaluate(clips, [f]) for f in BASELINE + NEW}
    sets = {k: evaluate(clips, v) for k, v in SETS.items()}

    c, b = sets["C"], sets["B"]
    sess_ok = sum(v["balanced_accuracy"] >= 0.75 for v in c["per_session"].values())
    crit = {"bal_acc_C": c["balanced_accuracy"], "c1_bal_acc_ge_0.75": c["balanced_accuracy"] >= 0.75,
            "sessions_ge_0.75": sess_ok, "c2_two_of_three_sessions": sess_ok >= 2,
            "bal_acc_B": b["balanced_accuracy"], "c3_beats_baseline_by_0.10": c["balanced_accuracy"] - b["balanced_accuracy"] >= 0.10}
    crit["improved"] = bool(crit["c1_bal_acc_ge_0.75"] and crit["c2_two_of_three_sessions"] and crit["c3_beats_baseline_by_0.10"])

    result = {"per_feature": per_feature, "single_feature_loso": single, "sets": sets, "criterion": crit,
              "coverage_note": "clips with the feature (of 24)"}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "passive_experiment.json").write_text(json.dumps(result, indent=2, default=float))

    L = ["# Passive-detection experiment: results", "",
         "Protocol: [docs/PASSIVE_EXPERIMENT.md](../PASSIVE_EXPERIMENT.md) (pre-registered). **One participant, 3 lighting sessions, 24 clips**; "
         "leave-one-session-out; clip-level 95% Wilson intervals are optimistic (clips of a session are correlated).", "",
         "## Clip-level passive detection (leave-one-session-out, flag if score ≥ 0.5)", "",
         "| Set | Features | Detection (swapped clips flagged) | False alarms (real clips flagged) | Balanced accuracy | Per-session balanced accuracy (daylight / dim / roomlight) |",
         "|---|---|---|---|---|---|"]
    names = {"T": "texture only", "B": "baseline passive", "N": "new only", "C": "baseline + new (primary)"}
    for k in ("T", "B", "N", "C"):
        s = sets[k]
        ps = s["per_session"]
        L.append(f"| {k}: {names[k]} | {len(SETS[k])} | {s['tp']}/{s['n_swapped']} = {fmt_ci(s['detection'], s['detection_ci'])} | "
                 f"{s['fp']}/{s['n_real']} = {fmt_ci(s['false_alarm'], s['false_alarm_ci'])} | {s['balanced_accuracy']:.2f} | "
                 f"{ps['daylight']['balanced_accuracy']:.2f} / {ps['dim']['balanced_accuracy']:.2f} / {ps['roomlight']['balanced_accuracy']:.2f} |")
    L += ["", "## Pre-registered success criterion (set C)", "",
          f"1. balanced accuracy ≥ 0.75: **{crit['c1_bal_acc_ge_0.75']}** ({crit['bal_acc_C']:.2f})",
          f"2. ≥ 2 of 3 held-out sessions with balanced accuracy ≥ 0.75: **{crit['c2_two_of_three_sessions']}** ({crit['sessions_ge_0.75']}/3)",
          f"3. C beats baseline B by ≥ 0.10: **{crit['c3_beats_baseline_by_0.10']}** (C {crit['bal_acc_C']:.2f} vs B {crit['bal_acc_B']:.2f})",
          f"\n**Passive detection improved: {'YES' if crit['improved'] else 'NO'}**", "",
          "## Per-feature separation (paired: same recording, swapped − real; 12 pairs, 3 sessions)", "",
          "| Feature | Clips with feature | Median real | Median swapped | Median Δ | Pairs with majority sign | Sessions with that sign | Pre-registered sign | Hypothesis held | Pooled AUC* | Single-feature LOSO bal. acc. |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in BASELINE + NEW:
        p = per_feature[f]
        hs = {1: "+", -1: "−", None: "–"}[p["hypothesised_sign"]]
        held = "–" if p["hypothesised_sign"] is None else ("yes" if p["majority_sign"] == p["hypothesised_sign"] else "no")
        ms = "+" if p["majority_sign"] > 0 else "−"
        L.append(f"| `{f}` | {p['coverage']}/24 | {p['median_real']:.3f} | {p['median_swapped']:.3f} | {p['median_d']:+.3f} | "
                 f"{p['consistent_pairs']}/{p['n_pairs']} ({ms}) | {p['sessions_with_majority_sign']}/{p['n_sessions']} | {hs} | {held} | "
                 f"{p['auc_pooled']:.2f} | {single[f]['balanced_accuracy']:.2f} |")
    L += ["", "*Pooled AUC is confounded by session offsets (informational only).", "",
          "## Controls (face-independent; the swapped clips were encoded one extra time)", "",
          "| Control | Median real | Median swapped | Median Δ | Pairs with majority sign | Sessions |", "|---|---|---|---|---|---|"]
    for f in CONTROLS:
        p = per_feature[f]
        L.append(f"| `{f}` | {p['median_real']:.3f} | {p['median_swapped']:.3f} | {p['median_d']:+.3f} | "
                 f"{p['consistent_pairs']}/{p['n_pairs']} | {p['sessions_with_majority_sign']}/{p['n_sessions']} |")
    (Path(a.out) / "passive_experiment.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
