"""Measure interior/exterior shadow lengths and fill on a low-sun scene."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fit import FitResult, fit_tank
from .model import TankWindow
from .parallel import pmap
from .scene import SCL_CLOUD, SCL_NODATA, SCL_SATURATED, Scene

log = logging.getLogger(__name__)

BBL_PER_M3 = 6.289811
DEFAULT_HEIGHT = 15.0


@dataclass
class MeasureConfig:
    band: str = "B08"
    psf_sigma: float = 4.5        # metres, Gaussian sensor blur
    h_min: float = 5.0            # plausible tank heights, metres
    h_max: float = 25.0
    near_full_freeboard: float = 3.0   # roof within this of the top -> near_full
    max_nrmse: float = 0.25       # rmse / (lit - shadow contrast) above this -> poor_fit
    min_contrast: float = 0.04    # reflectance; less than this -> poor_fit
    max_fill_sigma: float = 0.2   # 1-sigma fill error above this -> poor_fit
    max_offset: float = 20.0      # metres, bound on the geolocation search
    overlap_loss: float = 0.25    # fraction of own signal hidden by neighbours -> shadow_overlap
    min_offset_tanks: int = 3
    max_offset_tanks: int = 40    # pass 1 uses at most this many (largest) tanks
    offset: tuple | None = None   # force a scene offset (dx, dy) instead of estimating it
    shadow_level: float | None = None  # force the scene shadow brightness


@dataclass
class TankFit:
    row: dict
    win: TankWindow | None = None
    fit: FitResult | None = None
    neighbour_mask: np.ndarray | None = None


@dataclass
class SceneFit:
    scene: Scene
    table: pd.DataFrame
    tanks: list = field(default_factory=list)
    offset: tuple = (0.0, 0.0)
    shadow_level: float | None = None


def _locked(t) -> bool:
    v = t.get("height_locked", False)
    return bool(v) and str(v).lower() not in ("false", "0", "nan", "")


def _height(t) -> float:
    h = t.get("height_m", np.nan)
    try:
        h = float(h)
    except (TypeError, ValueError):
        h = np.nan
    return h if np.isfinite(h) and h > 0 else DEFAULT_HEIGHT


def prepare_windows(scene: Scene, tanks: pd.DataFrame, cfg: MeasureConfig):
    """Build one TankWindow per tank and mask pixels touched by neighbours."""
    img = scene.reflectance(cfg.band)
    valid = np.isfinite(img)
    scl = scene.scl()
    cloud = np.zeros_like(valid)
    if scl is not None:
        valid &= ~np.isin(scl, (SCL_NODATA, SCL_SATURATED))
        cloud = np.isin(scl, SCL_CLOUD)
    d = scene.shadow_direction()
    tan_el = math.tan(math.radians(scene.sun_elevation))
    xs, ys = scene.lonlat_to_xy(tanks.lon.values, tanks.lat.values)
    radii = tanks.diameter_m.values / 2.0
    heights = np.array([_height(t) for _, t in tanks.iterrows()])

    out = []
    for i, (_, t) in enumerate(tanks.iterrows()):
        row = dict(id=t["id"], date=scene.date, scene_id=scene.scene_id,
                   sun_elevation=scene.sun_elevation, sun_azimuth=scene.sun_azimuth)
        le_max = (heights[i] if _locked(t) else cfg.h_max) / tan_el
        try:
            win = TankWindow(img, scene.transform, xs[i], ys[i], d, radii[i] + cfg.max_offset,
                             le_max + cfg.max_offset, psf_sigma=cfg.psf_sigma, valid=valid)
        except ValueError:
            row["flag"] = "outside_scene"
            out.append(TankFit(row))
            continue

        # Clouds: the scene classifier sometimes calls bright roofs "cloud", so
        # only count cloud pixels away from the tank itself.
        r0, r1 = win.rows
        c0, c1 = win.cols
        cl = cloud[r0:r1, c0:c1]
        cols, rows = np.meshgrid(np.arange(c0, c1) + 0.5, np.arange(r0, r1) + 0.5)
        tr = scene.transform
        px, py = tr.c + tr.a * cols, tr.f + tr.e * rows
        away = np.hypot(px - xs[i], py - ys[i]) > radii[i] + 15
        cloud_frac = float((cl & away).sum() / max(away.sum(), 1))
        win.mask &= ~(cl & away)

        # Neighbours: mask every pixel their footprint or ground shadow reaches.
        # Relative positions are unaffected by the scene-wide offset.
        nmask = np.zeros(win.shape, bool)
        for j in range(len(tanks)):
            if j == i:
                continue
            le_j = heights[j] / tan_el
            reach = radii[i] + radii[j] + le_max + le_j + 2 * cfg.max_offset + 20
            if math.hypot(xs[j] - xs[i], ys[j] - ys[i]) > reach:
                continue
            _, S = win.coverages(xs[j] - xs[i], ys[j] - ys[i], radii[j], 0.0, le_j)
            nmask |= S > 0.03
        _, own = win.coverages(0.0, 0.0, radii[i], 0.0, heights[i] / tan_el)
        lost = float((own * nmask).sum() / max(own.sum(), 1e-9))
        win.mask &= ~nmask
        row.update(cloud_fraction=round(cloud_frac, 3), neighbour_loss=round(lost, 3))
        if cloud_frac > 0.2:
            row["flag"] = "cloud"
        elif lost > cfg.overlap_loss:
            row["flag"] = "shadow_overlap"
        elif win.mask.sum() < 12:
            row["flag"] = "too_few_pixels"
        out.append(TankFit(row, win, None, nmask))
    return out, radii, heights, tan_el


def _fit_job(job):
    win, init, free, bounds, s_fixed, unc = job
    return fit_tank(win, init, free, bounds, s_fixed=s_fixed, uncertainty=unc)


def measure_scene(scene: Scene, tanks: pd.DataFrame, cfg: MeasureConfig | None = None,
                  workers: int | None = None) -> SceneFit:
    cfg = cfg or MeasureConfig()
    tanks = tanks.reset_index(drop=True)
    fits, radii, heights, tan_el = prepare_windows(scene, tanks, cfg)
    locked = [_locked(t) for _, t in tanks.iterrows()]
    off_b = (-cfg.max_offset, cfg.max_offset)

    def bounds_for(i):
        b = {"Li": (0.0, cfg.h_max / tan_el), "Le": (cfg.h_min / tan_el * 0.5, cfg.h_max / tan_el)}
        b["dx"] = b["dy"] = off_b
        return b

    def init_for(i, dx=0.0, dy=0.0):
        le = heights[i] / tan_el
        return dict(dx=dx, dy=dy, R=radii[i], Li=0.5 * le, Le=le)

    # ---- pass 1: every tank on its own, offset and shadow level free ------
    offset = cfg.offset
    s_level = cfg.shadow_level
    if offset is None or s_level is None:
        p1 = []
        usable = [i for i, tf in enumerate(fits) if "flag" not in tf.row]
        usable = sorted(usable, key=lambda i: -radii[i])[:cfg.max_offset_tanks]
        jobs = []
        for i in usable:
            free = ["Li"] if locked[i] else ["Li", "Le"]
            if offset is None:
                free = ["dx", "dy"] + free
            init = init_for(i, *(offset or (0.0, 0.0)))
            jobs.append((fits[i].win, init, free, bounds_for(i), s_level, False))
        for i, r in zip(usable, pmap(_fit_job, jobs, {}, workers)):
            tf = fits[i]
            g, rr, s = r.levels
            L, S = tf.win.coverages(r.params["dx"], r.params["dy"], radii[i],
                                    r.params["Li"], r.params["Le"])
            shadow_px = float(((S - L) * tf.win.mask).sum())
            p1.append(dict(i=i, dx=r.params["dx"], dy=r.params["dy"], s=s, g=g, r=rr,
                           nrmse=r.nrmse, contrast=r.contrast, shadow_px=shadow_px))
        good = [q for q in p1 if q["nrmse"] < cfg.max_nrmse and q["contrast"] > cfg.min_contrast
                and q["s"] < min(q["g"], q["r"])]
        if offset is None:
            if len(good) >= cfg.min_offset_tanks:
                offset = (float(np.median([q["dx"] for q in good])),
                          float(np.median([q["dy"] for q in good])))
            else:
                log.warning("%s: only %d well-fitted tanks, assuming zero geolocation offset",
                            scene.date, len(good))
                offset = (0.0, 0.0)
        if s_level is None:
            sh = [q for q in good if q["shadow_px"] >= 2.0]
            if len(sh) >= cfg.min_offset_tanks:
                s_level = float(np.median([q["s"] for q in sh]))
            else:
                log.warning("%s: too few shadows to fix the shadow brightness; fitting it per tank",
                            scene.date)
    log.info("%s: offset dx=%.1f m dy=%.1f m, shadow level %s", scene.date, offset[0], offset[1],
             "free" if s_level is None else f"{s_level:.4f}")

    # ---- pass 2: offset and shadow level fixed, only shadow lengths --------
    todo = [i for i, tf in enumerate(fits) if "flag" not in tf.row]
    jobs = [(fits[i].win, init_for(i, *offset), ["Li"] if locked[i] else ["Li", "Le"],
             bounds_for(i), s_level, True) for i in todo]
    results = dict(zip(todo, pmap(_fit_job, jobs, {}, workers)))
    rows = []
    for i, tf in enumerate(fits):
        row = tf.row
        t = tanks.iloc[i]
        R = radii[i]
        row.update(diameter_m=2 * R, offset_dx=offset[0], offset_dy=offset[1])
        if "flag" in row:
            rows.append(row)
            continue
        r = results[i]
        tf.fit = r
        Li, Le = r.params["Li"], r.params["Le"]
        H = heights[i] if locked[i] else Le * tan_el
        fill = 1.0 - Li / Le if Le > 0 else float("nan")
        sLi = r.sigma.get("Li", float("nan"))
        sLe = r.sigma.get("Le", 0.0 if locked[i] else float("nan"))
        cov_ie = 0.0
        if r.cov is not None and "Le" in r.free:
            cov_ie = r.cov[r.free.index("Li"), r.free.index("Le")]
        if locked[i]:
            h_sig = float(t.get("height_sigma_m", 0.0) or 0.0)
            sLe = h_sig / tan_el if np.isfinite(h_sig) else 0.0
        # fill = 1 - Li/Le  ->  var = (sLi/Le)^2 + (Li sLe/Le^2)^2 - 2 Li cov/Le^3
        var = (sLi / Le) ** 2 + (Li * sLe / Le ** 2) ** 2 - 2 * Li * cov_ie / Le ** 3
        fill_sigma = math.sqrt(var) if np.isfinite(var) and var >= 0 else float("nan")
        cap = math.pi * R ** 2 * H * BBL_PER_M3
        g, rr, s = r.levels
        row.update(Li_m=Li, Le_m=Le, Li_sigma_m=sLi, Le_sigma_m=sLe, height_m=H,
                   height_locked=locked[i], fill=fill, fill_sigma=fill_sigma,
                   freeboard_m=Li * tan_el, capacity_bbl=cap, volume_bbl=fill * cap,
                   r2=r.r2(tf.win), nrmse=r.nrmse, contrast=r.contrast,
                   level_ground=g, level_roof=rr, level_shadow=s, npix=r.npix)
        row["flag"] = _flag(row, t, cfg)
        rows.append(row)

    cols = ["id", "date", "flag", "fill", "fill_sigma", "volume_bbl", "capacity_bbl", "Li_m", "Le_m",
            "Li_sigma_m", "Le_sigma_m", "height_m", "height_locked", "freeboard_m", "diameter_m",
            "r2", "nrmse", "contrast", "level_ground", "level_roof", "level_shadow", "npix", "offset_dx", "offset_dy",
            "cloud_fraction", "neighbour_loss", "sun_elevation", "sun_azimuth", "scene_id"]
    table = pd.DataFrame(rows)
    table = table[[c for c in cols if c in table.columns]]
    return SceneFit(scene, table, fits, offset, s_level)


def _flag(row, t, cfg: MeasureConfig) -> str:
    roof_type = str(t.get("roof_type", "") or "")
    if roof_type == "fixed_suspect":
        return "fixed_roof"
    if (row["contrast"] < cfg.min_contrast or row["nrmse"] > cfg.max_nrmse
            or not np.isfinite(row["fill_sigma"]) or row["fill_sigma"] > cfg.max_fill_sigma):
        # A near-full tank has an ill-defined Li (it sits on the 0 bound), so a
        # missing error bar there is expected, not a bad fit.
        if not (row["contrast"] >= cfg.min_contrast and row["nrmse"] <= cfg.max_nrmse
                and row["freeboard_m"] < cfg.near_full_freeboard):
            return "poor_fit"
    if not row["height_locked"] and not (cfg.h_min <= row["height_m"] <= cfg.h_max * 0.98):
        return "height_implausible"
    if row["freeboard_m"] < cfg.near_full_freeboard:
        return "near_full"
    return "ok"
