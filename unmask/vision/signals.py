"""Per-clip OpenCV measurements. Every function returns plain floats (or None when
the signal could not be measured) so results serialise straight to JSON."""
from __future__ import annotations

import cv2
import numpy as np

from .face import FACE_MIN_SCORE, Face
from .pose import head_pose
from .pose import MODEL_POINTS
from . import regions as R

EPS = 1e-6


def _nanstat(fn, xs, default=None):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(fn(xs)) if xs else default


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 4 or np.std(a) < EPS or np.std(b) < EPS:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="valid")


def present(fc: Face | None) -> bool:
    return fc is not None and fc.score >= FACE_MIN_SCORE


# --------------------------------------------------------------------------- 1
def detection_stats(faces: list[Face | None]) -> dict:
    scores = np.array([fc.score if fc is not None else 0.0 for fc in faces])
    pres = np.array([present(fc) for fc in faces])
    flips = int(np.sum(pres[1:] != pres[:-1])) if len(pres) > 1 else 0
    return {
        "face_present_frac": float(pres.mean()) if len(pres) else 0.0,
        "det_score_mean": float(scores[pres].mean()) if pres.any() else 0.0,
        "det_score_std": float(scores[pres].std()) if pres.sum() > 1 else 0.0,
        "det_flips_per_s": None,  # filled in by analyze() which knows the clip duration
        "_flips": flips,
        "face_width_median": float(np.median([fc.width for fc in faces if present(fc)])) if pres.any() else 0.0,
    }


def landmark_jitter(faces: list[Face | None]) -> float | None:
    """Median landmark acceleration (2nd difference) normalised by face width.

    Real heads move smoothly; face-swap renderers often wobble frame to frame."""
    accs = []
    for a, b, c in zip(faces, faces[1:], faces[2:]):
        if present(a) and present(b) and present(c):
            acc = np.linalg.norm(a.landmarks - 2 * b.landmarks + c.landmarks, axis=1).mean()
            accs.append(acc / max(b.width, 1.0))
    return float(np.median(accs)) if len(accs) >= 3 else None


def landmark_jumps(faces: list[Face | None], idx: list[int] | None = None) -> float | None:
    """Largest frame-to-frame landmark displacement (relative to face width),
    optionally restricted to transitions touching frames in idx."""
    best = None
    sel = set(idx) if idx is not None else None
    for i in range(1, len(faces)):
        a, b = faces[i - 1], faces[i]
        if not (present(a) and present(b)):
            continue
        if sel is not None and i not in sel and (i - 1) not in sel:
            continue
        # remove rigid translation so that a fast but coherent move is not a "jump"
        da = a.landmarks - a.landmarks.mean(0)
        db = b.landmarks - b.landmarks.mean(0)
        d = float(np.linalg.norm(da - db, axis=1).max() / max(b.width, 1.0))
        best = d if best is None else max(best, d)
    return best


# --------------------------------------------------------------------------- 2
def pose_series(faces: list[Face | None], frame_size) -> tuple[list[float | None], list[float | None]]:
    """Yaw trajectory (deg) and PnP reprojection error (fraction of eye distance)."""
    from .pose import yaw_proxy
    yaws: list[float | None] = []
    errs: list[float | None] = []
    w, h = frame_size
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    for fc in faces:
        if not present(fc):
            yaws.append(None)
            errs.append(None)
            continue
        proxy = yaw_proxy(fc.landmarks)
        img = fc.landmarks.astype(np.float64).reshape(5, 1, 2)
        ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, img, cam, None, flags=cv2.SOLVEPNP_EPNP)
        err = None
        if ok:
            ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, img, cam, None, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
        if ok:
            proj, _ = cv2.projectPoints(MODEL_POINTS, rvec, tvec, cam, None)
            eye = np.linalg.norm(fc.landmarks[1] - fc.landmarks[0]) + EPS
            err = float(np.linalg.norm(proj.reshape(5, 2) - fc.landmarks, axis=1).mean() / eye)
        hp = head_pose(fc.landmarks, frame_size) if ok else None
        if hp is None or not np.isfinite(hp[0]) or abs(hp[0] - proxy) > 25:
            yaws.append(float(proxy))
        else:
            yaws.append(float(0.5 * hp[0] + 0.5 * proxy))
        errs.append(err)
    return yaws, errs


def head_turn_features(faces, yaws, errs, seam_series) -> dict:
    vals = [y for y in yaws if y is not None]
    if len(vals) < 3:
        return {"yaw_range": None}
    base = float(np.median(vals[: max(3, len(vals) // 5)]))
    dev = [abs(y - base) if y is not None else None for y in yaws]
    yaw_range = float(max(vals) - min(vals))
    hi = [i for i, d in enumerate(dev) if d is not None and d > 0.6 * max(d2 for d2 in dev if d2 is not None)]
    lo = [i for i, d in enumerate(dev) if d is not None and d < 10]
    # Detector behaviour across the turn: real faces lose confidence smoothly
    # towards profile; swaps tend to flicker or collapse abruptly.
    scores = np.array([fc.score if fc is not None else 0.0 for fc in faces])
    ds = np.abs(np.diff(scores))
    seam_hi = _nanstat(np.mean, [seam_series[i] for i in hi])
    seam_lo = _nanstat(np.mean, [seam_series[i] for i in lo])
    return {
        "yaw_range": yaw_range,
        "yaw_max_abs_dev": float(max(d for d in dev if d is not None)),
        "turn_score_jerk": float(np.percentile(ds, 90)) if len(ds) else None,
        "turn_reproj_err": _nanstat(np.median, [errs[i] for i in hi]),
        "turn_reproj_err_gain": (None if not lo or not hi else
                                 _nanstat(np.median, [errs[i] for i in hi], 0) - _nanstat(np.median, [errs[i] for i in lo], 0)),
        "turn_seam_gain": None if seam_hi is None or seam_lo is None else float(seam_hi - seam_lo),
        "turn_landmark_jump": landmark_jumps(faces, hi),
    }


# --------------------------------------------------------------------------- 3
def seam_frame(frame: np.ndarray, fc: Face) -> dict:
    """Edge and colour discontinuity in an elliptical band at the face boundary."""
    crop = R.crop_face(frame, fc.box)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    m_in, m_band, m_out = (R.ring_mask(fc.box, r) for r in (R.RING_IN, R.BAND, R.RING_OUT))
    g_in, g_band, g_out = (float(mag[m].mean()) if m.any() else 0.0 for m in (m_in, m_band, m_out))
    edge = g_band / (0.5 * (g_in + g_out) + EPS)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab).astype(np.float32)
    # Colour step across the seam, per angular sector, using the lower 2/3 of
    # the face (cheeks/jaw) where skin should be continuous; median is robust
    # to hair and ears.
    yy, xx = np.mgrid[0:R.CROP, 0:R.CROP]
    ang = np.arctan2(yy - R.CROP / 2, xx - R.CROP / 2)
    r = R.ellipse_radius_map(float(fc.box[3] / max(fc.box[2], 1e-3)))
    steps = []
    for a0 in np.linspace(-np.pi / 6, np.pi + np.pi / 6, 9)[:-1]:
        sec = (ang >= a0) & (ang < a0 + np.pi * (4 / 3) / 8)
        ia = sec & (r >= 0.74) & (r < 0.86)
        ib = sec & (r >= 0.86) & (r < 0.98)
        if ia.sum() > 5 and ib.sum() > 5:
            steps.append(float(np.linalg.norm(lab[ia].mean(0) - lab[ib].mean(0))))
    return {"edge": float(edge), "color_step": float(np.median(steps)) if steps else 0.0,
            "crop_gray": gray, "band_mask": m_band, "in_mask": m_in}


def seam_features(frames, faces) -> tuple[dict, list[float | None]]:
    edges, steps, series = [], [], []
    band_flicker, prev = [], None
    for f, fc in zip(frames, faces):
        if not present(fc):
            series.append(None)
            prev = None
            continue
        s = seam_frame(f, fc)
        edges.append(s["edge"])
        steps.append(s["color_step"])
        series.append(s["edge"])
        if prev is not None:
            d = np.abs(s["crop_gray"] - prev)
            band_flicker.append(float(d[s["band_mask"]].mean() / (d[s["in_mask"]].mean() + 1.0)))
        prev = s["crop_gray"]
    if not edges:
        return {"seam_edge": None, "seam_color_step": None, "seam_flicker": None, "seam_edge_cv": None}, series
    e = np.array(edges)
    return {
        "seam_edge": float(np.median(e)),
        "seam_edge_cv": float(e.std() / (e.mean() + EPS)) if len(e) > 2 else None,
        "seam_color_step": float(np.median(steps)),
        "seam_flicker": float(np.median(band_flicker)) if band_flicker else None,
    }, series


# --------------------------------------------------------------------------- 4
def motion_masks(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Moving-pixel masks from Farneback optical flow combined with MOG2
    background subtraction. masks[i] describes motion into frame i."""
    small_w = 320
    bg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25, detectShadows=False)
    out = [np.zeros(frames[0].shape[:2], bool)] if frames else []
    prev = None
    for i, f in enumerate(frames):
        scale = small_w / f.shape[1]
        g = cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        fg = bg.apply(g, learningRate=-1)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = cv2.magnitude(flow[..., 0], flow[..., 1])
            thr = max(1.0, 3.0 * float(np.median(mag)))
            m = (mag > thr) | ((fg > 0) & (mag > 0.5 * thr))
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            out.append(cv2.resize(m, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool))
        prev = g
    return out


def occlusion_features(frames, faces, masks) -> tuple[dict, list[float]]:
    """Hand-pass analysis: does a face 'survive' a hand moving across it?"""
    last_box, last_tmpl = None, None
    occ_frac: list[float] = []
    persist, leak, occ_idx, pre_scores, occ_scores = [], [], [], [], []
    for i, (f, fc, m) in enumerate(zip(frames, faces, masks)):
        if present(fc) and (last_box is None or not occ_frac or occ_frac[-1] < 0.15):
            last_box = fc.box.copy()
        if last_box is None:
            occ_frac.append(0.0)
            continue
        x, y, w, h = (int(v) for v in last_box)
        x0, y0 = max(0, x), max(0, y)
        sub = m[y0:y + h, x0:x + w]
        frac = float(sub.mean()) if sub.size else 0.0
        occ_frac.append(frac)
        crop = cv2.cvtColor(R.crop_face(f, last_box, 96), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if frac < 0.08:
            if present(fc):
                last_tmpl = crop
                pre_scores.append(fc.score)
            continue
        if frac >= 0.2:
            occ_idx.append(i)
            occ_scores.append(fc.score if fc is not None else 0.0)
            persist.append(1.0 if (fc is not None and fc.score >= 0.75) else 0.0)
            if last_tmpl is not None:
                mm = R.crop_face(m.astype(np.uint8) * 255, last_box, 96) > 127
                if mm.sum() > 50:
                    c = _corr(crop[mm], last_tmpl[mm])
                    if c is not None:
                        leak.append(c)
    feats = {
        "occ_max_frac": float(max(occ_frac)) if occ_frac else 0.0,
        "occ_frames": len(occ_idx),
        "occ_persist": float(np.mean(persist)) if persist else None,
        "occ_score_drop": (float(np.mean(pre_scores) - min(occ_scores)) if pre_scores and occ_scores else None),
        "occ_texture_leak": float(np.median(leak)) if leak else None,
        "occ_landmark_jump": landmark_jumps(faces, occ_idx) if occ_idx else None,
    }
    return feats, occ_frac


# --------------------------------------------------------------------------- 5
def lighting_series(frames, faces) -> dict[str, list[float | None]]:
    keys = ("face", "neck", "bg", "face_lr", "neck_lr")
    out: dict[str, list[float | None]] = {k: [] for k in keys}
    last_box = None
    for f, fc in zip(frames, faces):
        if present(fc):
            last_box = fc.box
        if last_box is None:
            for k in keys:
                out[k].append(None)
            continue
        L = cv2.cvtColor(f, cv2.COLOR_BGR2Lab)[..., 0].astype(np.float32)
        ms = R.frame_masks(L.shape, last_box)
        for k in ("face", "neck", "bg"):
            out[k].append(float(L[ms[k]].mean()) if ms[k].sum() > 30 else None)
        cx = int(last_box[0] + last_box[2] / 2)
        for k, mk in (("face_lr", "face"), ("neck_lr", "neck")):
            left, right = ms[mk].copy(), ms[mk].copy()
            left[:, cx:] = False
            right[:, :cx] = False
            out[k].append(float(L[left].mean() - L[right].mean()) if left.sum() > 15 and right.sum() > 15 else None)
    return out


def lighting_features(series: dict) -> dict:
    face = series["face"]
    res: dict = {"light_ref_range": None, "light_corr": None, "light_gain_dev": None, "light_lr_corr": None}
    best_ref = None
    for ref in ("neck", "bg"):
        idx = [i for i, (a, b) in enumerate(zip(face, series[ref])) if a is not None and b is not None]
        if len(idx) < 6:
            continue
        fa = _smooth(np.array([face[i] for i in idx]))
        rb = _smooth(np.array([series[ref][i] for i in idx]))
        rng = float(rb.max() - rb.min())
        frng = float(fa.max() - fa.min())
        c = _corr(np.diff(fa), np.diff(rb))
        if c is None:
            c = _corr(fa, rb)
        gain = float(np.polyfit(rb - rb.mean(), fa - fa.mean(), 1)[0]) if np.std(rb) > EPS else None
        cand = {"ref": ref, "range": max(rng, frng), "corr": c, "gain": gain}
        if best_ref is None or (c is not None and (best_ref["corr"] is None or c > best_ref["corr"])):
            best_ref = cand
    if best_ref:
        res["light_ref_range"] = best_ref["range"]
        res["light_corr"] = best_ref["corr"]
        if best_ref["gain"] is not None:
            res["light_gain_dev"] = float(abs(np.log(max(abs(best_ref["gain"]), 0.05))))
    idx = [i for i, (a, b) in enumerate(zip(series["face_lr"], series["neck_lr"])) if a is not None and b is not None]
    if len(idx) >= 6:
        res["light_lr_corr"] = _corr(np.array([series["face_lr"][i] for i in idx]),
                                     np.array([series["neck_lr"][i] for i in idx]))
    return res


# --------------------------------------------------------------------------- 6
def _hf_ratio(gray: np.ndarray) -> float:
    g = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = (g - g.mean()) * np.outer(np.hanning(64), np.hanning(64))
    spec = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    yy, xx = np.mgrid[-32:32, -32:32]
    rad = np.sqrt(xx ** 2 + yy ** 2)
    return float(spec[rad > 12].sum() / (spec[rad > 1].sum() + EPS))


def texture_features(frames, faces) -> dict:
    lap_ratio, hf_face, hf_ratio = [], [], []
    for f, fc in list(zip(frames, faces))[::2]:
        if not present(fc):
            continue
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        ms = R.frame_masks(gray.shape, fc.box)
        if ms["bg"].sum() < 100:
            continue
        v_face = float(lap[ms["face"]].var())
        v_bg = float(lap[ms["bg"]].var())
        lap_ratio.append(np.log((v_face + 1.0) / (v_bg + 1.0)))
        crop = cv2.cvtColor(R.crop_face(f, fc.box), cv2.COLOR_BGR2GRAY)
        inner = crop[R.CROP // 2 - 30:R.CROP // 2 + 30, R.CROP // 2 - 30:R.CROP // 2 + 30]
        hf = _hf_ratio(inner)
        hf_face.append(hf)
        # same-size reference patch from the frame itself at the same pixel scale
        s = int(fc.box[2] * 60 / (R.CROP / R.CROP_MARGIN))
        H, W = gray.shape
        for (px, py) in ((0, 0), (W - s, 0), (0, H - s), (W - s, H - s)):
            patch = gray[max(0, py):py + s, max(0, px):px + s]
            if patch.size and patch.std() > 4:
                hf_ratio.append(np.log((hf + 1e-4) / (_hf_ratio(patch) + 1e-4)))
                break
    return {
        "tex_lap_ratio": _nanstat(np.median, lap_ratio),
        "tex_hf_face": _nanstat(np.median, hf_face),
        "tex_hf_ratio": _nanstat(np.median, hf_ratio),
    }


# --------------------------------------------------------------------------- 7
def response_onset(challenge: str, times: list[float], yaws, occ_frac, light: dict) -> float | None:
    """Seconds from clip start (challenge issued) to the start of the requested motion."""
    if challenge == "head_turn":
        vals = [(t, y) for t, y in zip(times, yaws) if y is not None]
        if len(vals) < 3:
            return None
        base = float(np.median([y for _, y in vals[:3]]))
        for t, y in vals:
            if abs(y - base) > 15:
                return float(t)
    elif challenge == "hand_pass":
        for t, o in zip(times, occ_frac):
            if o >= 0.15:
                return float(t)
    elif challenge == "light_change":
        ref = light.get("neck") or []
        ref = [r if r is not None else b for r, b in zip(ref, light.get("bg", []))]
        vals = [(t, r) for t, r in zip(times, ref) if r is not None]
        if len(vals) < 4:
            return None
        base = float(np.median([r for _, r in vals[:3]]))
        rng = max(r for _, r in vals) - min(r for _, r in vals)
        for t, r in vals:
            if abs(r - base) > max(4.0, 0.3 * rng):
                return float(t)
    return None
