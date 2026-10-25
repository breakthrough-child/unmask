"""Fit per-feature thresholds from labelled clips (robust, few-sample friendly).

For each feature: center = midpoint of the class medians, direction = which side
swaps fall on, scale = pooled robust spread, weight = clipped separation (d').
Use with leave-one-participant-out (LOPO) to estimate generalisation."""
from __future__ import annotations

import copy
import math

import numpy as np


def _robust_std(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    mad = np.median(np.abs(x - np.median(x))) * 1.4826
    return float(mad if mad > 0 else x.std())


def fit_config(clips: list[dict], base_cfg: dict, version: str, min_per_class: int = 2) -> dict:
    """clips: [{"label": "real"|"swapped", "usable": bool, "features": {...}}]"""
    cfg = copy.deepcopy(base_cfg)
    cfg["version"] = version
    report = {}
    for sname, sig in cfg["signals"].items():
        for feat, spec in sig["features"].items():
            r = [c["features"][feat] for c in clips if c.get("usable") and c["label"] == "real"
                 and isinstance(c["features"].get(feat), (int, float))]
            s = [c["features"][feat] for c in clips if c.get("usable") and c["label"] == "swapped"
                 and isinstance(c["features"].get(feat), (int, float))]
            r = [v for v in r if math.isfinite(v)]
            s = [v for v in s if math.isfinite(v)]
            if len(r) < min_per_class or len(s) < min_per_class:
                report[feat] = {"status": "kept_prior", "n_real": len(r), "n_swapped": len(s)}
                continue
            mr, ms = float(np.median(r)), float(np.median(s))
            spread = math.sqrt((_robust_std(r) ** 2 + _robust_std(s) ** 2) / 2) or abs(ms - mr) / 4 or 1e-3
            dprime = abs(ms - mr) / spread
            spec["center"] = round((mr + ms) / 2, 5)
            spec["direction"] = 1 if ms >= mr else -1
            # scale so that each class median sits ~2 logits from the centre, but
            # never sharper than the within-class spread allows
            spec["scale"] = round(max(abs(ms - mr) / 4, spread / 2, 1e-4), 5)
            spec["weight"] = round(float(np.clip(dprime, 0.2, 2.0)), 3)
            report[feat] = {"status": "fitted", "median_real": round(mr, 4), "median_swapped": round(ms, 4),
                            "dprime": round(dprime, 3), "n_real": len(r), "n_swapped": len(s)}
    cfg["fit_report"] = report
    return cfg
