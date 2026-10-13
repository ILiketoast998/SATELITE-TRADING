"""Find tanks as bright discs in a high-sun scene and fit their radius."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_laplace, maximum_filter

from .fit import fit_tank
from .measure import TankFit
from .model import TankWindow
from .parallel import CTX, pmap
from .scene import Scene

log = logging.getLogger(__name__)


@dataclass
class DetectConfig:
    band: str = "B04"             # red: white roofs against dark vegetation
    min_diameter: float = 25.0    # smaller tanks are 1-2 pixels; not measurable anyway
    max_diameter: float = 110.0
    min_contrast: float = 0.08    # roof minus ground reflectance
    max_nrmse: float = 0.35
    psf_sigma: float = 4.5
    h_max: float = 25.0
    ignore_shadows: bool = False  # fit discs with no shadows (to see the bias)
    max_candidates: int = 2000
    min_roundness: float = 0.35   # Hessian eigenvalue ratio; 1 = round blob, 0 = line


def find_candidates(img: np.ndarray, cfg: DetectConfig, px: float = 10.0):
    """Scale-normalised Laplacian-of-Gaussian blob detection.

    Returns a list of (row, col, radius_m, response). For a disc of contrast c
    the normalised response at the matched scale is about 0.74 c.
    """
    filled = np.where(np.isfinite(img), img, np.nanmedian(img))
    radii = np.geomspace(cfg.min_diameter / 2, cfg.max_diameter / 2, 10) / px
    stack = np.stack([-(r / math.sqrt(2)) ** 2 * gaussian_laplace(filled, r / math.sqrt(2))
                      for r in radii])
    best = stack.max(axis=0)
    scale = stack.argmax(axis=0)
    size = max(3, int(round(cfg.min_diameter / px)))
    peaks = (best == maximum_filter(best, size=size)) & (best > 0.5 * cfg.min_contrast)
    rr, cc = np.nonzero(peaks)
    order = np.argsort(-best[rr, cc])
    out = []
    h, w = img.shape
    hess = {}
    for k in order:
        r0, c0 = int(rr[k]), int(cc[k])
        si = int(scale[r0, c0])
        rad = float(radii[si])                               # pixels
        # Reject lines (roads, pipe racks): at a blob both Hessian eigenvalues
        # are strongly negative; along a line one of them is near zero.
        if si not in hess:
            sg = rad / math.sqrt(2)
            hess[si] = [gaussian_filter(filled, sg, order=o) for o in ((0, 2), (2, 0), (1, 1))]
        hxx, hyy, hxy = (a[r0, c0] for a in hess[si])
        tr_, det = hxx + hyy, hxx * hyy - hxy ** 2
        disc = math.sqrt(max(tr_ ** 2 / 4 - det, 0.0))
        l1, l2 = tr_ / 2 - disc, tr_ / 2 + disc                # l1 <= l2
        if l1 >= 0 or l2 / l1 < cfg.min_roundness:
            continue
        # Non-maximum suppression: one candidate per disc.
        if any(math.hypot(r0 - a, c0 - b) <= max(rad, rb / px) for a, b, rb, _ in out):
            continue
        # Cheap contrast check before the expensive model fit: mean of the
        # disc core against the median of a ring of ground around it.
        n = int(np.ceil(rad + 3.5))
        ys, xs = np.mgrid[max(r0 - n, 0):min(r0 + n + 1, h), max(c0 - n, 0):min(c0 + n + 1, w)]
        dist = np.hypot(ys - r0, xs - c0)
        core_sel = dist <= max(rad - 0.5, 1.0)
        core = filled[ys[core_sel], xs[core_sel]]
        ring_sel = (dist >= rad + 1.5) & (dist <= rad + 3.5)
        ring = filled[ys[ring_sel], xs[ring_sel]]
        if core.size == 0 or ring.size < 8:
            continue
        # The lower quartile of the core must be bright, which rejects roads
        # and other thin lines that the smallest LoG scale also responds to.
        if np.percentile(core, 25) - np.median(ring) < 0.7 * cfg.min_contrast:
            continue
        out.append((r0, c0, rad * px, float(best[r0, c0])))
        if len(out) >= cfg.max_candidates:
            break
    return out


def _window(x, y, r):
    c = CTX
    shift = 0.6 * r + 10
    win = TankWindow(c["img"], c["tr"], x, y, c["d"], min(1.4 * r + 10, c["r_hi"]) + shift,
                     c["le_max"], psf_sigma=c["cfg"].psf_sigma, valid=c["valid"])
    return win, shift


def _full_fit(win, shift, r, s_fixed):
    c, cfg = CTX, CTX["cfg"]
    bounds = {"dx": (-shift, shift), "dy": (-shift, shift), "R": (c["r_lo"], c["r_hi"]),
              "Li": (0.0, c["le_max"]), "Le": (0.0, c["le_max"])}
    init = dict(dx=0.0, dy=0.0, R=r, Li=0.0, Le=0.0 if cfg.ignore_shadows else 0.5 * c["le_max"])
    free = ["dx", "dy", "R"] if cfg.ignore_shadows else ["dx", "dy", "R", "Li", "Le"]
    return fit_tank(win, init, free, bounds, s_fixed=s_fixed,
                    grid_step=0 if cfg.ignore_shadows else 2.0,
                    steps={"R": 3.0, "Li": 2.0, "Le": 2.0}, uncertainty=False)


def _first_fit(cand):
    """A quick shadow-free disc fit rejects non-tanks and centres the window,
    then the full model is fitted with the shadow brightness free."""
    x, y, r = cand
    c, cfg = CTX, CTX["cfg"]
    try:
        win, shift = _window(x, y, r)
    except ValueError:
        return None
    quick = fit_tank(win, dict(dx=0.0, dy=0.0, R=r, Li=0.0, Le=0.0), ["dx", "dy", "R"],
                     {"dx": (-shift, shift), "dy": (-shift, shift), "R": (c["r_lo"], c["r_hi"])},
                     grid_step=0, steps={"R": 3.0}, uncertainty=False, maxiter=150)
    g, rr, _ = quick.levels
    if rr - g < cfg.min_contrast or quick.nrmse > 1.5 * cfg.max_nrmse:
        return None
    x, y, r = x + quick.params["dx"], y + quick.params["dy"], quick.params["R"]
    try:
        win, shift = _window(x, y, r)
    except ValueError:
        return None
    fit = _full_fit(win, shift, r, None)
    g, rr, _ = fit.levels
    if rr - g < cfg.min_contrast or fit.nrmse > cfg.max_nrmse:
        return None
    p = fit.params
    return dict(x=x + p["dx"], y=y + p["dy"], R=p["R"], fit=fit, win=win)


def _second_fit(k):
    """Refit one disc with its neighbours masked and the shadow level fixed."""
    c, cfg = CTX, CTX["cfg"]
    x, y, R, _ = c["others"][k]
    try:
        win, shift = _window(x, y, R)
    except ValueError:
        return None
    reach = 2 * c["r_hi"] + c["le_max"] + 60
    for j, (ox, oy, oR, oLe) in enumerate(c["others"]):
        if j == k or math.hypot(ox - x, oy - y) > reach:
            continue
        _, S = win.coverages(ox - win.cx, oy - win.cy, oR, 0.0, oLe)
        win.mask &= ~(S > 0.05)
    if win.mask.sum() < 12:
        return None
    fit = _full_fit(win, shift, R, c["s_level"])
    g, rr, _ = fit.levels
    p = fit.params
    if (rr - g < cfg.min_contrast or fit.nrmse > cfg.max_nrmse
            or not (c["r_lo"] + 0.5 < p["R"] < c["r_hi"] - 0.5)):
        return None
    return dict(x=x + p["dx"], y=y + p["dy"], R=p["R"], Le=p["Le"], nrmse=fit.nrmse,
                contrast=rr - g, fit=fit, win=win)


def detect_tanks(scene: Scene, cfg: DetectConfig | None = None, candidates: pd.DataFrame | None = None,
                 workers: int | None = None):
    """Return (tank table, list of TankFit for diagnostics)."""
    cfg = cfg or DetectConfig()
    img = scene.reflectance(cfg.band)
    valid = np.isfinite(img)
    tr = scene.transform
    d = scene.shadow_direction()
    tan_el = math.tan(math.radians(scene.sun_elevation))
    le_max = cfg.h_max / tan_el

    if candidates is None:
        cands = []
        for row, col, r, resp in find_candidates(img, cfg, tr.a):
            x, y = tr.c + tr.a * (col + 0.5), tr.f + tr.e * (row + 0.5)
            cands.append((x, y, r))
    else:
        xs, ys = scene.lonlat_to_xy(candidates.lon.values, candidates.lat.values)
        rs = (candidates.diameter_m.values / 2 if "diameter_m" in candidates
              else np.full(len(xs), 20.0))
        cands = list(zip(xs, ys, rs))
    log.info("%d candidate discs", len(cands))

    r_lo, r_hi = cfg.min_diameter / 2 * 0.7, cfg.max_diameter / 2 * 1.2
    ctx = dict(img=img, valid=valid, tr=tr, d=d, cfg=cfg, le_max=le_max, r_lo=r_lo, r_hi=r_hi)

    # Pass 1: every candidate alone, shadow brightness free.
    first = [q for q in pmap(_first_fit, cands, ctx, workers) if q is not None]
    first = _dedupe(first)
    log.info("%d discs survive the first fit", len(first))

    # Scene shadow brightness from tanks whose (small) shadows cover >= 1.5 px.
    # Summer shadows are small, so the per-tank values are noisy; clip them to
    # the physical range and fall back to a fraction of the ground brightness.
    s_level = None
    if not cfg.ignore_shadows and first:
        ss, gs = [], []
        for q in first:
            p = q["fit"].params
            g, _, s = q["fit"].levels
            gs.append(g)
            L, S = q["win"].coverages(p["dx"], p["dy"], p["R"], p["Li"], p["Le"])
            if ((S - L) * q["win"].mask).sum() >= 1.5:
                ss.append(min(max(s, 0.0), g))
        g_med = float(np.median(gs))
        s_level = float(np.median(ss)) if len(ss) >= 3 else 0.15 * g_med
        if not (0.02 * g_med < s_level < 0.8 * g_med):
            s_level = 0.15 * g_med
        log.info("scene shadow level: %.4f (ground %.4f)", s_level, g_med)

    # Pass 2: neighbours masked, shadow level fixed.
    ctx["s_level"] = s_level
    ctx["others"] = [(o["x"], o["y"], o["R"], 0.0 if cfg.ignore_shadows else o["fit"].params["Le"])
                     for o in first]
    out = [o for o in pmap(_second_fit, range(len(first)), ctx, workers) if o is not None]
    for o in out:
        o["H"] = o["Le"] * tan_el
        if not (5.0 <= o["H"] < 0.98 * cfg.h_max):
            o["H"] = np.nan        # summer shadows are too short to pin the height down
    out = _dedupe(out)

    out.sort(key=lambda o: (-o["y"], o["x"]))          # north to south, west to east
    lon, lat = scene.xy_to_lonlat(np.array([o["x"] for o in out]), np.array([o["y"] for o in out]))
    rows, tank_fits = [], []
    for k, o in enumerate(out):
        tid = f"T{k + 1:04d}"
        rows.append(dict(id=tid, lon=round(float(lon[k]), 7), lat=round(float(lat[k]), 7),
                         diameter_m=round(2 * o["R"], 2),
                         height_m=np.nan if cfg.ignore_shadows else round(o["H"], 2),
                         height_source="summer_fit", height_sigma_m=np.nan, height_locked=False,
                         roof_type="unknown", detect_contrast=round(o["contrast"], 4),
                         detect_nrmse=round(o["nrmse"], 3), detect_scene=scene.scene_id))
        tank_fits.append(TankFit(dict(id=tid, flag="detected"), o["win"], o["fit"]))
    return pd.DataFrame(rows), tank_fits


def _dedupe(items):
    """Drop discs whose centre lies inside a better-fitting disc."""
    items = sorted(items, key=lambda q: q["fit"].nrmse)
    kept = []
    for q in items:
        if all(math.hypot(q["x"] - k["x"], q["y"] - k["y"]) > max(q["R"], k["R"]) for k in kept):
            kept.append(q)
    return kept
