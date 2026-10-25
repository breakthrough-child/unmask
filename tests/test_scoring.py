import copy

from unmask.scoring import is_confident, load_config, session_risk
from unmask.scoring.scorer import feature_score


def clip(ch, usable=True, **features):
    return {"challenge": ch, "usable": usable, "features": features}


CFG = load_config()


def _neutral(signal):
    """Feature values sitting exactly on each feature's center (score 0.5)."""
    return {f: spec["center"] for f, spec in CFG["signals"][signal]["features"].items()}


def test_feature_score_direction_and_monotonic():
    spec = {"center": 1.0, "scale": 0.1, "direction": 1}
    assert feature_score(0.5, spec) < feature_score(1.0, spec) < feature_score(1.5, spec)
    assert abs(feature_score(1.0, spec) - 0.5) < 1e-9
    spec["direction"] = -1
    assert feature_score(0.5, spec) > feature_score(1.5, spec)


def test_no_usable_clips_is_inconclusive():
    r = session_risk([clip("passive", usable=False)], CFG)
    assert r["verdict"] == "INCONCLUSIVE" and r["risk"] is None


def test_passive_only_cannot_be_low_risk():
    good = {f: spec["center"] - 4 * spec["scale"] * spec.get("direction", 1)
            for sig in ("seam", "texture", "temporal") for f, spec in CFG["signals"][sig]["features"].items()}
    r = session_risk([clip("passive", **good)], CFG)
    assert r["risk"] < 0.2
    assert r["verdict"] == "INCONCLUSIVE"  # needs at least one challenge for LOW_RISK


def _all(direction_sign):
    feats = {}
    for sig in CFG["signals"].values():
        for f, spec in sig["features"].items():
            feats[f] = spec["center"] + direction_sign * 4 * spec["scale"] * spec.get("direction", 1)
    return feats


def test_clear_real_and_clear_fake():
    real = [clip("passive", **_all(-1)), clip("head_turn", **_all(-1))]
    fake = [clip("passive", **_all(+1)), clip("head_turn", **_all(+1))]
    rr, rf = session_risk(real, CFG), session_risk(fake, CFG)
    assert rr["verdict"] == "LOW_RISK" and rf["verdict"] == "HIGH_RISK"
    assert rf["top_reasons"] and "Do not send money" in rf["advice"]
    assert is_confident(rf, CFG)


def test_single_strong_challenge_signal_forces_suspicious():
    feats = _all(-1)
    feats.update({f: spec["center"] + 8 * spec["scale"] * spec.get("direction", 1)
                  for f, spec in CFG["signals"]["occlusion"]["features"].items()})
    r = session_risk([clip("passive", **feats), clip("hand_pass", **feats)], CFG)
    assert r["verdict"] in ("SUSPICIOUS", "HIGH_RISK")


def test_uncertainty_is_high_for_borderline_signal():
    r = session_risk([clip("passive", **_neutral("seam"))], CFG)
    assert r["signals"]["seam"]["uncertainty"] > 0.9
    assert r["signals"]["pose"]["score"] is None and r["signals"]["pose"]["uncertainty"] == 1.0


def test_config_pointer_resolves():
    cfg = copy.deepcopy(CFG)
    assert "signals" in cfg and cfg["version"]
