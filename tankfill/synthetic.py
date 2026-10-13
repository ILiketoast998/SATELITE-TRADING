"""Synthetic Sentinel-2-like scenes with known tanks, for testing the method.

The renderer deliberately differs from the fitting model so a test is not an
"inverse crime": classes are point-sampled at 1 m (no anti-aliasing), the ground
is textured, roofs have their own brightness, and shadows are a fixed fraction
of whatever surface they fall on (so interior and exterior shadows differ).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

from .scene import write_scene

BBL_PER_M3 = 6.289811


def default_tanks(seed=0) -> pd.DataFrame:
    """A small tank farm: a grid of tanks with varied size, height and fill."""
    rng = np.random.default_rng(seed)
    rows = []
    tid = 0
    for i in range(4):
        for j in range(5):
            R = float(rng.uniform(15, 40))
            rows.append(dict(id=f"T{tid:03d}", x=300.0 + j * 230 + rng.uniform(-20, 20),
                             y=300.0 + i * 230 + rng.uniform(-20, 20), R=R,
                             H=float(rng.uniform(12, 20)), fill=float(rng.uniform(0.15, 0.9)),
                             roof="floating"))
            tid += 1
    df = pd.DataFrame(rows)
    df.loc[3, "roof"] = "fixed"          # a cone-roof tank: never an interior shadow
    # Two tanks packed so close that one shades the other.
    df.loc[len(df)] = dict(id=f"T{tid:03d}", x=1500.0, y=1250.0, R=25.0, H=16.0, fill=0.5, roof="floating")
    df.loc[len(df)] = dict(id=f"T{tid+1:03d}", x=1500.0, y=1190.0, R=25.0, H=16.0, fill=0.5, roof="floating")
    return df


def render_scene(path, tanks: pd.DataFrame, sun_azimuth=160.0, sun_elevation=31.0,
                 date="2024-01-15", size_m=1800.0, origin=(700000.0, 3982000.0),
                 epsg=32614, offset=(0.0, 0.0), psf_sigma=4.5, noise=0.004,
                 shadow_fraction=0.18, seed=1, fine=1.0, fills=None):
    """Render a 10 m scene. Tank x, y in the CSV are relative to `origin`'s
    upper-left corner (x east, y south). `offset` shifts everything in the image
    (metres, east/north) to imitate misregistration.
    """
    rng = np.random.default_rng(seed)
    n10 = int(size_m // 10)
    nf = int(round(size_m / fine))
    xs = (np.arange(nf) + 0.5) * fine
    ys = (np.arange(nf) + 0.5) * fine
    X, Y = np.meshgrid(xs, ys)                  # Y grows southwards (image rows)
    az, el = math.radians(sun_azimuth), math.radians(sun_elevation)
    # Shadow direction in (east, south) image coordinates.
    ux, uy = -math.sin(az), math.cos(az)
    tan_el = math.tan(el)

    def band(ground_mean, ground_sd, roof_lo, roof_hi, band_seed):
        r = np.random.default_rng(band_seed)
        tex = gaussian_filter(r.standard_normal((nf, nf)), 15 / fine)
        ground = ground_mean + ground_sd * tex / tex.std()
        img = ground.copy()
        shade = np.ones_like(img)
        for _, t in tanks.iterrows():
            cx, cy = t.x + offset[0], t.y - offset[1]
            fill = t.fill if fills is None else fills.get(t.id, t.fill)
            Le = t.H / tan_el
            Li = 0.0 if t.roof == "fixed" else (1 - fill) * t.H / tan_el
            sl = (slice(max(int((cy - t.R - Le - 5) / fine), 0), int((cy + t.R + Le + 5) / fine)),
                  slice(max(int((cx - t.R - Le - 5) / fine), 0), int((cx + t.R + Le + 5) / fine)))
            x, y = X[sl] - cx, Y[sl] - cy
            disc = np.hypot(x, y) <= t.R
            lit = disc & (np.hypot(x - Li * ux, y - Li * uy) <= t.R)
            tt = np.clip(x * ux + y * uy, 0, Le)
            stad = np.hypot(x - tt * ux, y - tt * uy) <= t.R
            roof = roof_lo + (roof_hi - roof_lo) * r.uniform()
            img[sl][disc] = roof
            shade[sl][(stad & ~disc) | (disc & ~lit)] = shadow_fraction
        return img * shade

    red = band(0.08, 0.02, 0.22, 0.40, seed + 10)
    nir = band(0.24, 0.04, 0.25, 0.42, seed + 20)
    out = []
    for img in (red, nir):
        img = gaussian_filter(img, psf_sigma / fine, mode="nearest")
        k = int(round(10 / fine))
        img = img.reshape(n10, k, n10, k).mean(axis=(1, 3))
        img = img + noise * rng.standard_normal(img.shape)
        out.append(np.clip(np.round(img * 10000), 1, 65535))
    scl = np.full(out[0].shape, 5, dtype=np.uint16)        # "not vegetated"
    transform = from_origin(origin[0], origin[1], 10.0, 10.0)
    meta = dict(id=f"SYNTH_{date}", datetime=f"{date}T17:20:00Z", sun_azimuth=sun_azimuth,
                sun_elevation=sun_elevation, boa_add_offset=0.0, source="synthetic",
                aoi_cloud_fraction=0.0, aoi_snow_fraction=0.0, aoi_nodata_fraction=0.0,
                true_offset=list(offset))
    return write_scene(path, out[0], out[1], scl, transform, f"EPSG:{epsg}", meta)


def truth_table(tanks: pd.DataFrame, origin=(700000.0, 3982000.0), epsg=32614) -> pd.DataFrame:
    """Tank truth in the same columns `detect` writes (lon/lat from origin)."""
    from pyproj import Transformer
    t = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon, lat = t.transform(origin[0] + tanks.x.values, origin[1] - tanks.y.values)
    df = pd.DataFrame(dict(id=tanks.id, lon=lon, lat=lat, diameter_m=2 * tanks.R,
                           height_m=tanks.H, true_fill=tanks.fill, roof=tanks.roof))
    df["capacity_bbl"] = math.pi * tanks.R ** 2 * tanks.H * BBL_PER_M3
    return df
