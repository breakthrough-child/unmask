"""Experiment 2: within-clip swap dropout (protocol: docs/PASSIVE_EXPERIMENT.md, "Experiment 2"). Run once.

  python eval/dropout_experiment.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["UNMASK_DROPOUT_FEATURES"] = "1"
from unmask.vision.dropout import FEATURES  # noqa: E402

_s = importlib.util.spec_from_file_location("passive_experiment", ROOT / "eval" / "passive_experiment.py")
PE = importlib.util.module_from_spec(_s)
_s.loader.exec_module(PE)
RE = PE.RE

HYP = {f: +1 for f in FEATURES}      # pre-registered: swapped - real > 0 for all three
CHALLENGES = ("hand_pass", "head_turn")


def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval by bisection (no scipy)."""
    def solve(f, lo=0.0, hi=1.0):
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: 1 - binom_cdf(k - 1, n, p) < alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p) > alpha / 2)
    return lower, upper


def boot_ci(d: list[float], reps: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    a = np.array(d)
    meds = [np.median(a[rng.integers(0, len(a), len(a))]) for _ in range(reps)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main() -> None:
    rows = [r for r in RE.read_labels(ROOT / "data") if r["challenge"] in CHALLENGES]
    work = ROOT / "eval" / "out" / "dropout"
    clips = RE.analyse_all(rows, work / "cache", work / "evidence")
    assert len(clips) == 12 and all(c["usable"] for c in clips)
    sessions = sorted({c["condition"] for c in clips})

    per_feature, pair_rows = {}, {}
    for f in FEATURES:
        by = {(c["condition"], c["challenge"], c["label"]): c["features"].get(f) for c in clips}
        pairs = []
        for cond in sessions:
            for ch in CHALLENGES:
                r, s = by.get((cond, ch, "real")), by.get((cond, ch, "swapped"))
                ok = isinstance(r, (int, float)) and isinstance(s, (int, float))
                pairs.append({"session": cond, "challenge": ch, "real": r, "swapped": s, "d": (s - r) if ok else None})
        ds = [p["d"] for p in pairs if p["d"] is not None]
        k = sum(d > 0 for d in ds)
        sess_mean = {}
        for cond in sessions:
            v = [p["d"] for p in pairs if p["session"] == cond and p["d"] is not None]
            sess_mean[cond] = float(np.mean(v)) if v else None
        sess_ok = sum(1 for v in sess_mean.values() if v is not None and v > 0)
        per_feature[f] = {"pairs": pairs, "n_pairs": len(ds), "k_positive": k,
                          "k_ci": clopper_pearson(k, len(ds)) if ds else None,
                          "median_d": float(np.median(ds)) if ds else None,
                          "median_d_boot_ci": boot_ci(ds) if ds else None,
                          "session_mean_d": sess_mean, "sessions_positive": sess_ok,
                          "c1": bool(len(ds) >= 6 and sess_ok == 3 and k >= 5)}

    sets = {"D": PE.evaluate(clips, list(FEATURES))}
    single = {f: PE.evaluate(clips, [f]) for f in FEATURES}
    d = sets["D"]
    sess_bal = {s: v["balanced_accuracy"] for s, v in d["per_session"].items()}
    crit = {"c1_any_feature_all3_sessions_and_5of6": any(v["c1"] for v in per_feature.values()),
            "c1_by_feature": {f: v["c1"] for f, v in per_feature.items()},
            "D_balanced_accuracy": d["balanced_accuracy"], "c2_D_bal_acc_ge_0.75": d["balanced_accuracy"] >= 0.75 - 1e-9,
            "D_per_session": sess_bal, "c2b_all_sessions_ge_0.75": all(v >= 0.75 - 1e-9 for v in sess_bal.values())}
    crit["met"] = bool(crit["c1_any_feature_all3_sessions_and_5of6"] and crit["c2_D_bal_acc_ge_0.75"] and crit["c2b_all_sessions_ge_0.75"])

    out = ROOT / "docs" / "results"
    (out / "dropout_experiment.json").write_text(json.dumps({"per_feature": per_feature, "set_D": sets["D"], "single": single,
                                                           "criterion": crit}, indent=2, default=float))
    L = ["# Experiment 2: within-clip swap dropout, results", "",
         "Protocol: [PASSIVE_EXPERIMENT.md](../PASSIVE_EXPERIMENT.md) (Experiment 2, pre-registered, run once). **One participant, 3 sessions, 12 clips "
         "(hand_pass + head_turn; 6 real, 6 swapped), 6 paired recordings.** Hypothesis: swapped − real > 0 for all three features.", "",
         "## Per-pair direction (same recording: swapped − real)", "",
         "| Feature | Session | Challenge | Real | Swapped | Δ (swapped − real) |", "|---|---|---|---|---|---|"]
    for f in FEATURES:
        for p in per_feature[f]["pairs"]:
            fm = lambda v: "–" if v is None else f"{v:.3f}"   # noqa: E731
            L.append(f"| `{f}` | {p['session']} | {p['challenge']} | {fm(p['real'])} | {fm(p['swapped'])} | {'–' if p['d'] is None else format(p['d'], '+.3f')} |")
    L += ["", "## Separation with intervals", "",
          "| Feature | Pairs Δ > 0 (95% exact CI) | Median Δ (95% bootstrap CI over pairs) | Session-mean Δ (daylight / dim / roomlight) | Sessions with Δ > 0 | Criterion 1 |",
          "|---|---|---|---|---|---|"]
    for f in FEATURES:
        v = per_feature[f]
        sm = " / ".join("–" if v["session_mean_d"][s] is None else f"{v['session_mean_d'][s]:+.3f}" for s in sessions)
        ci = v["k_ci"]
        bc = v["median_d_boot_ci"]
        L.append(f"| `{f}` | {v['k_positive']}/{v['n_pairs']} ({100 * ci[0]:.0f}–{100 * ci[1]:.0f}%) | "
                 f"{v['median_d']:+.3f} ({bc[0]:+.3f} to {bc[1]:+.3f}) | {sm} | {v['sessions_positive']}/3 | {'met' if v['c1'] else 'not met'} |")
    L += ["", "## Leave-one-session-out, clip level (flag if score ≥ 0.5)", "",
          "| Set | Swapped flagged | Real flagged | Balanced accuracy | Per session (daylight / dim / roomlight) |", "|---|---|---|---|---|"]
    for name, s in [("D (all three)", sets["D"])] + [(f"`{f}` alone", single[f]) for f in FEATURES]:
        ps = s["per_session"]
        L.append(f"| {name} | {s['tp']}/{s['n_swapped']} = {PE.fmt_ci(s['detection'], s['detection_ci'])} | {s['fp']}/{s['n_real']} = "
                 f"{PE.fmt_ci(s['false_alarm'], s['false_alarm_ci'])} | {s['balanced_accuracy']:.2f} | "
                 f"{ps['daylight']['balanced_accuracy']:.2f} / {ps['dim']['balanced_accuracy']:.2f} / {ps['roomlight']['balanced_accuracy']:.2f} |")
    L += ["", "## Pre-registered success criterion", "",
          f"1. a feature with Δ > 0 in the session mean of all 3 sessions and in ≥ 5 of 6 pairs: **{crit['c1_any_feature_all3_sessions_and_5of6']}** "
          f"(by feature: {crit['c1_by_feature']})",
          f"2. set D balanced accuracy ≥ 0.75: **{crit['c2_D_bal_acc_ge_0.75']}** ({crit['D_balanced_accuracy']:.2f}); "
          f"all 3 held-out sessions ≥ 0.75: **{crit['c2b_all_sessions_ge_0.75']}** ({ {k: round(v, 2) for k, v in sess_bal.items()} })",
          f"\n**Criterion met: {'YES' if crit['met'] else 'NO'}**"]
    (out / "dropout_experiment.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
