"""Deterministic, explainable risk scoring. The verdict is computed here, never by the LLM.

feature value --sigmoid((x - center) / scale * direction)--> feature suspicion
feature suspicions --weighted mean--> signal score in [0, 1]
signal scores --reliability-weighted mean--> session risk --thresholds--> verdict
"""
from __future__ import annotations

import json
import math
from pathlib import Path

CONFIG_DIR = Path(__file__).with_name("config")
DEFAULT_CONFIG = CONFIG_DIR / "scoring_current.json"

VERDICTS = ("LOW_RISK", "SUSPICIOUS", "HIGH_RISK", "INCONCLUSIVE")


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    cfg = json.loads(p.read_text())
    if "extends" in cfg:  # a pointer file: {"extends": "scoring_v1.json"}
        return load_config(p.with_name(cfg["extends"]))
    return cfg


def _sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def feature_score(x: float, spec: dict) -> float:
    return _sigmoid((x - spec["center"]) / max(spec["scale"], 1e-9) * spec.get("direction", 1))


def aggregate_features(clips: list[dict]) -> dict[str, list[float]]:
    """Collect feature values from usable clips (feature -> list of values)."""
    out: dict[str, list[float]] = {}
    for c in clips:
        if not c.get("usable"):
            continue
        for k, v in (c.get("features") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                out.setdefault(k, []).append(float(v))
    return out


def score_signals(clips: list[dict], cfg: dict) -> dict[str, dict]:
    vals = aggregate_features(clips)
    res: dict[str, dict] = {}
    for name, sig in cfg["signals"].items():
        num = den = 0.0
        used = {}
        for feat, spec in sig["features"].items():
            xs = vals.get(feat)
            if not xs:
                continue
            x = sum(xs) / len(xs)
            s = feature_score(x, spec)
            w = spec.get("weight", 1.0)
            num += w * s
            den += w
            used[feat] = {"value": round(x, 4), "suspicion": round(s, 3), "n": len(xs)}
        total_w = sum(sp.get("weight", 1.0) for sp in sig["features"].values())
        if den == 0:
            res[name] = {"score": None, "uncertainty": 1.0, "coverage": 0.0, "features": {}}
            continue
        score = num / den
        coverage = den / total_w
        # Borderline scores and partially measured signals are uncertain.
        unc = 1.0 - abs(2 * score - 1) * (0.5 + 0.5 * coverage)
        res[name] = {"score": round(score, 3), "uncertainty": round(unc, 3),
                     "coverage": round(coverage, 3), "features": used}
    return res


def session_risk(clips: list[dict], cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    sigs = score_signals(clips, cfg)
    usable = [c for c in clips if c.get("usable")]
    challenges_done = list(dict.fromkeys(c["challenge"] for c in usable if c["challenge"] != "passive"))
    num = den = 0.0
    for name, s in sigs.items():
        if s["score"] is None:
            continue
        w = cfg["signals"][name]["weight"] * (0.5 + 0.5 * s["coverage"])
        num += w * s["score"]
        den += w
    risk = num / den if den else None
    v = cfg["verdict"]
    if risk is None or not usable:
        verdict = "INCONCLUSIVE"
    elif risk >= v["high_risk"]:
        verdict = "HIGH_RISK"
    elif risk >= v["suspicious"] or any(
            s["score"] is not None and s["score"] >= v["single_signal_suspicious"]
            and cfg["signals"][n].get("challenge") for n, s in sigs.items()):
        verdict = "SUSPICIOUS"
    elif len(challenges_done) >= v["min_challenges_for_low"]:
        verdict = "LOW_RISK"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "risk": None if risk is None else round(risk, 3),
        "verdict": verdict,
        "signals": sigs,
        "challenges_done": challenges_done,
        "usable_clips": len(usable),
        "top_reasons": explain(sigs, cfg, verdict),
        "advice": ADVICE[verdict],
        "config_version": cfg.get("version"),
    }


def is_confident(risk: dict, cfg: dict) -> bool:
    """Early-stop rule shared by the rule policy and the Bedrock guardrail."""
    r, n = risk["risk"], len(risk["challenges_done"])
    st = cfg["stop"]
    if r is None:
        return False
    measured = {k: s for k, s in risk["signals"].items() if s["score"] is not None}
    challenge_sigs = {k: s for k, s in measured.items() if cfg["signals"][k].get("challenge")}
    if n >= st["min_challenges_high"]:
        if r >= st["high_confidence"]:
            return True
        # one decisive, well-covered challenge signal is enough to stop (a swap that
        # visibly breaks under a hand pass doesn't need a lighting test as well)
        if any(s["score"] >= st.get("single_signal_confident", 1.1) and s["coverage"] >= 0.5
               for s in challenge_sigs.values()):
            return True
    if r <= st["low_confidence"] and n >= st["min_challenges_low"]:
        if any(s["score"] > st.get("max_challenge_score_low", 1.0) for s in challenge_sigs.values()):
            return False
        return max((s["uncertainty"] for s in challenge_sigs.values()), default=1.0) <= st["max_uncertainty_low"]
    return False


def explain(sigs: dict, cfg: dict, verdict: str, k: int = 3) -> list[str]:
    ranked = sorted(((n, s) for n, s in sigs.items() if s["score"] is not None),
                    key=lambda ns: -cfg["signals"][ns[0]]["weight"] * abs(ns[1]["score"] - 0.5))
    out = []
    for name, s in ranked:
        text = cfg["signals"][name]["explain"]
        if s["score"] >= 0.6:
            out.append(text["high"])
        elif s["score"] <= 0.4 and verdict in ("LOW_RISK", "INCONCLUSIVE"):
            out.append(text["low"])
        if len(out) >= k:
            break
    if not out:
        out.append("No single signal stood out strongly either way.")
    return out


ADVICE = {
    "HIGH_RISK": ("Treat this call as likely fake. Do not send money, codes or personal details. "
                  "Hang up and call the person back on a number you already know."),
    "SUSPICIOUS": ("Something looks off. Don't act on any request for money or secrets during this call. "
                   "Verify through another channel you trust (call back on a known number, ask a question only they would know)."),
    "LOW_RISK": ("No strong signs of a face swap were found. This is not proof of identity: "
                 "still verify unusual requests, especially for money, through a channel you trust."),
    "INCONCLUSIVE": ("Unmask couldn't get a clear enough view to decide. Treat the call with caution and "
                     "verify the person's identity another way before acting on any request."),
}
