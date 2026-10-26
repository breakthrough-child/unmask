"""Passive signals (docs/PASSIVE_EXPERIMENT.md): resolution mismatch, sensor-noise mismatch,
skin-tone mismatch and swap dropout, computed on frontal, unoccluded frames of ANY clip.

Everything compares patches of the same pixel size taken from the same frame (same optics, no rescaling).
Definitions were fixed before any result was computed; see the pre-registration for the rationale."""
from __future__ import annotations

import os

import cv2
import numpy as np

from . import regions as R

MIN_SCORE = 0.8          # face confidence
MAX_YAW = 20.0           # degrees
OCC_MAX = 0.08           # motion fraction inside the face box (hand-pass clips)
STRIDE = 2               # every 2nd eligible frame ...
MAX_FRAMES = 60          # ... at most this many
MIN_FRAMES = 8
MIN_FRAMES_TEMPORAL = 12
EPS = 1e-9
NOISE_FLOOR = 0.05       # numerical floor for the noise level (log of 0)

FEATURES = ("p_res_pyr_neck", "p_res_pyr_surround", "p_res_fft_neck",
            "p_noise_face_vs_neck", "p_noise_face_vs_bg", "p_noise_tvar_ratio", "p_noise_tcorr_diff",
            "p_skin_de_neck", "p_skin_de_neck_ab", "p_skin_de_forehead_ab",
            "p_drop_sep", "p_drop_minor_frac", "p_drop_switch_rate")
CONTROLS = ("p_ctl_bg_noise", "p_ctl_bg_hf")


def patch_size(face_w: float) -> int:
    return int(min(64, max(24, round(0.30 * face_w))))


def neck_fit_enabled() -> bool:
    """Post-hoc variant A1 (docs/PASSIVE_EXPERIMENT.md); default off = the pre-registered geometry. Read at call time."""
    return os.environ.get("UNMASK_PASSIVE_NECK_FIT") == "1"


def neck_cy(y: float, h: float, s: int, frame_h: int, fit: bool) -> float | None:
    """Neck-patch centre row. Pre-registered: y + 1.18*h. Variant A1 (post-hoc, UNMASK_PASSIVE_NECK_FIT=1): slide up to the lowest
    position that fits in the frame, but not above y0 >= y + 0.9*h."""
    cy = y + 1.18 * h
    if not fit:
        return cy
    cy = min(cy, frame_h - s / 2)
    return cy if cy - s / 2 >= y + 0.9 * h else None


def crop(img: np.ndarray, cx: float, cy: float, s: int) -> np.ndarray | None:
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    if x0 < 0 or y0 < 0 or x0 + s > img.shape[1] or y0 + s > img.shape[0]:
        return None
    return img[y0:y0 + s, x0:x0 + s]


def _log(x: float) -> float:
    return float(np.log(max(x, EPS)))


def pyr_ratio(patch: np.ndarray) -> float:
    """E0/E1: energy of the finest Laplacian-pyramid band over the next one (lower = upsampled / low resolution)."""
    g0 = patch.astype(np.float32)
    g1 = cv2.pyrDown(g0)
    l0 = g0 - cv2.pyrUp(g1, dstsize=(g0.shape[1], g0.shape[0]))
    g2 = cv2.pyrDown(g1)
    l1 = g1 - cv2.pyrUp(g2, dstsize=(g1.shape[1], g1.shape[0]))
    return float(np.mean(l0 ** 2) / (np.mean(l1 ** 2) + EPS))


_fft_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def hf_power(patch: np.ndarray) -> float:
    """Power at normalised radius > 0.5 over power at 0 < radius <= 1 (Hann window, DC excluded).
    Normalised radius = spatial frequency / Nyquist (0.5 cycles per pixel)."""
    s = patch.shape[0]
    if s not in _fft_cache:
        w = np.hanning(s).astype(np.float32)
        f = np.fft.fftfreq(s)
        r = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2) / 0.5
        _fft_cache[s] = (np.outer(w, w), (r > 0.5) & (r <= 1.0), (r > 0) & (r <= 1.0))
    win, hf, band = _fft_cache[s]
    p = patch.astype(np.float32)
    spec = np.abs(np.fft.fft2((p - p.mean()) * win)) ** 2
    return float(spec[hf].sum() / (spec[band].sum() + EPS))


def residual(patch_u8: np.ndarray) -> np.ndarray:
    return patch_u8.astype(np.float32) - cv2.medianBlur(patch_u8, 3).astype(np.float32)


def noise_level(res: np.ndarray) -> float:
    """25th percentile over 8x8 blocks of the residual's block standard deviation (the flattest blocks = sensor noise)."""
    b = 8
    n = res.shape[0] // b
    blocks = res[:n * b, :n * b].reshape(n, b, n, b).transpose(0, 2, 1, 3).reshape(n * n, b * b)
    return max(float(np.percentile(blocks.std(axis=1), 25)), NOISE_FLOOR)


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return None if d < EPS else float((a * b).sum() / d)


def _med_lab(lab: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    return np.median(lab[mask], axis=0) if mask.sum() >= 30 else None


def _regions_masks(shape, box):
    h, w = shape[:2]
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2
    centre = np.zeros((h, w), np.uint8)
    cv2.ellipse(centre, (int(cx), int(cy + 0.05 * bh)), (max(1, int(0.25 * bw)), max(1, int(0.25 * bh))), 0, 0, 360, 1, -1)
    outer = np.zeros((h, w), np.uint8)
    inner = np.zeros((h, w), np.uint8)
    cv2.ellipse(outer, (int(cx), int(cy)), (max(1, int(0.5 * bw)), max(1, int(0.5 * bh))), 0, 0, 360, 1, -1)
    cv2.ellipse(inner, (int(cx), int(cy)), (max(1, int(0.425 * bw)), max(1, int(0.425 * bh))), 0, 0, 360, 1, -1)
    ann = (outer.astype(bool) & ~inner.astype(bool))
    ann[int(cy - 0.15 * bh):, :] = False          # forehead edge only (upper part of the face)
    return centre.astype(bool), ann, R.frame_masks(shape[:2], box)["neck"]


def kmeans_dropout(desc: np.ndarray, duration_s: float) -> tuple[float, float, float] | None:
    """(separation, minor-cluster fraction, label switches per second) of a 2-means split of per-frame descriptors."""
    n = len(desc)
    if n < MIN_FRAMES_TEMPORAL:
        return None
    z = (desc - desc.mean(0)) / np.maximum(desc.std(0), 1e-6)
    cv2.setRNGSeed(0)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    _, labels, centers = cv2.kmeans(z.astype(np.float32), 2, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    within = float(np.sqrt(np.mean(np.sum((z - centers[labels]) ** 2, axis=1))))
    sep = float(np.linalg.norm(centers[0] - centers[1]) / (within + EPS))
    minor = float(min((labels == 0).sum(), (labels == 1).sum()) / n)
    switches = float(np.sum(labels[1:] != labels[:-1]) / max(duration_s, EPS))
    return sep, minor, switches


def passive_features(frames, faces, yaws, times, occ_frac=None) -> dict:
    """Clip-level passive features (None when a feature cannot be measured on enough eligible frames)."""
    out: dict = {k: None for k in FEATURES + CONTROLS}
    fit_neck = neck_fit_enabled()
    elig = []
    for i, (f, fc) in enumerate(zip(frames, faces)):
        if fc is None or fc.score < MIN_SCORE or yaws[i] is None or abs(yaws[i]) > MAX_YAW:
            continue
        if occ_frac and i < len(occ_frac) and occ_frac[i] >= OCC_MAX:
            continue
        elig.append(i)
    out["p_n_eligible"] = len(elig)
    if len(elig) < MIN_FRAMES:
        return out
    sampled = elig[::STRIDE][:MAX_FRAMES]

    rows = []   # per sampled frame
    for i in sampled:
        f, fc = frames[i], faces[i]
        x, y, w, h = fc.box
        cx, cy = x + w / 2, y + h / 2
        s = patch_size(w)
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        pf = crop(gray, cx, cy + 0.10 * h, s)
        if pf is None:
            continue
        ncy = neck_cy(y, h, s, gray.shape[0], fit_neck)
        pn = crop(gray, cx, ncy, s) if ncy is not None else None
        ps = [q for q in (crop(gray, cx - 0.75 * w, cy, s), crop(gray, cx + 0.75 * w, cy, s)) if q is not None]
        corners = [gray[:s, :s], gray[:s, -s:], gray[-s:, :s], gray[-s:, -s:]]
        r = {"i": i}
        rf = residual(pf)
        r["nF"] = noise_level(rf)
        r["hfF"] = hf_power(pf)
        r["pyrF"] = pyr_ratio(pf)
        r["nG"] = float(np.mean([_log(noise_level(residual(c))) for c in corners]))
        r["hfG"] = float(np.mean([_log(hf_power(c)) for c in corners]))
        if pn is not None:
            r["nN"] = noise_level(residual(pn))
            r["hfN"] = hf_power(pn)
            r["pyrN"] = pyr_ratio(pn)
        if ps:
            r["pyrS"] = float(np.mean([_log(pyr_ratio(q)) for q in ps]))
        lab = cv2.cvtColor(f.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
        centre, forehead, neck = _regions_masks(f.shape, fc.box)
        c = _med_lab(lab, centre)
        if c is not None:
            r["lab"] = c
            nk, fh = _med_lab(lab, neck), _med_lab(lab, forehead)
            if nk is not None:
                r["de_neck"], r["de_neck_ab"] = float(np.linalg.norm(c - nk)), float(np.linalg.norm(c[1:] - nk[1:]))
            if fh is not None:
                r["de_fh_ab"] = float(np.linalg.norm(c[1:] - fh[1:]))
            r["lapF"] = _log(float(cv2.Laplacian(pf, cv2.CV_32F).var()) + 1.0)
        rows.append(r)
    if len(rows) < MIN_FRAMES:
        return out

    def med(vals):
        v = [x for x in vals if x is not None and np.isfinite(x)]
        return float(np.median(v)) if len(v) >= MIN_FRAMES else None

    g = lambda k: [r.get(k) for r in rows]   # noqa: E731
    out["p_res_pyr_neck"] = med([None if "pyrN" not in r else _log(r["pyrF"]) - _log(r["pyrN"]) for r in rows])
    out["p_res_pyr_surround"] = med([None if "pyrS" not in r else _log(r["pyrF"]) - r["pyrS"] for r in rows])
    out["p_res_fft_neck"] = med([None if "hfN" not in r else _log(r["hfF"]) - _log(r["hfN"]) for r in rows])
    out["p_noise_face_vs_neck"] = med([None if "nN" not in r else _log(r["nF"]) - _log(r["nN"]) for r in rows])
    out["p_noise_face_vs_bg"] = med([_log(r["nF"]) - r["nG"] for r in rows])
    out["p_skin_de_neck"] = med(g("de_neck"))
    out["p_skin_de_neck_ab"] = med(g("de_neck_ab"))
    out["p_skin_de_forehead_ab"] = med(g("de_fh_ab"))
    out["p_ctl_bg_noise"] = med([r["nG"] for r in rows])
    out["p_ctl_bg_hf"] = med([r["hfG"] for r in rows])
    # temporal noise variation (needs >= 12 frames with both patches)
    both = [r for r in rows if "nN" in r]
    if len(both) >= MIN_FRAMES_TEMPORAL:
        sf, sn = np.std([_log(r["nF"]) for r in both]), np.std([_log(r["nN"]) for r in both])
        out["p_noise_tvar_ratio"] = float(sf / (sn + 1e-6))
    # residual correlation between consecutive eligible frames (box centre moved < 2 px)
    cf, cn = [], []
    for a, b in zip(elig, elig[1:]):
        if b != a + 1:
            continue
        fa, fb = faces[a], faces[b]
        if np.linalg.norm(fa.center - fb.center) >= 2.0:
            continue
        w = fb.box[2]
        s = patch_size(w)
        ga, gb = cv2.cvtColor(frames[a], cv2.COLOR_BGR2GRAY), cv2.cvtColor(frames[b], cv2.COLOR_BGR2GRAY)
        cx, cy = fb.center
        yb = fb.box[1]
        hb = fb.box[3]
        pa, pb = crop(ga, cx, cy + 0.10 * hb, s), crop(gb, cx, cy + 0.10 * hb, s)
        ncy = neck_cy(yb, hb, s, ga.shape[0], fit_neck)
        if ncy is None:
            continue
        na, nb = crop(ga, cx, ncy, s), crop(gb, cx, ncy, s)
        if pa is None or na is None or nb is None:
            continue
        c1, c2 = _corr(residual(pa), residual(pb)), _corr(residual(na), residual(nb))
        if c1 is not None and c2 is not None:
            cf.append(c1)
            cn.append(c2)
    if len(cf) >= 6:
        out["p_noise_tcorr_diff"] = float(np.mean(cf) - np.mean(cn))
    # swap dropout
    d = np.array([[*r["lab"], r["lapF"], _log(r["hfF"])] for r in rows if "lab" in r and "lapF" in r], np.float32)
    dur = (times[sampled[-1]] - times[sampled[0]]) if len(sampled) > 1 else 0.0
    kd = kmeans_dropout(d, dur) if len(d) else None
    if kd:
        out["p_drop_sep"], out["p_drop_minor_frac"], out["p_drop_switch_rate"] = kd
    return out
