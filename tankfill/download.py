"""Download Sentinel-2 L2A B04, B08 and SCL for a bounding box, cloud-filtered.

Sources (none needs an account, key or payment):

* ``earth-search`` (default) - Element 84's STAC API over the public AWS
  bucket ``s3://sentinel-cogs``. Plain HTTPS, nothing to sign.
* ``planetary-computer`` - Microsoft's STAC API. Anonymous, but asset URLs must
  be signed with short-lived SAS tokens: ``pip install planetary-computer``.
* ``aws-s3`` - no search API at all: lists the public ``sentinel-cogs`` bucket
  by MGRS tile and reads each scene's STAC JSON. Use it if the STAC APIs are
  down or blocked. Same imagery as earth-search.

Only the pixels inside the bounding box are read (Cloud-Optimised GeoTIFF
range requests), so a scene costs a few hundred kB, not 1 GB.
"""
from __future__ import annotations

import logging
import math
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from .scene import SCL_CLOUD, SCL_NODATA, SCL_SNOW, write_scene

log = logging.getLogger(__name__)

STAC_URLS = {
    "earth-search": "https://earth-search.aws.element84.com/v1",
    "planetary-computer": "https://planetarycomputer.microsoft.com/api/stac/v1",
}
ASSETS = {
    "earth-search": {"B04": "red", "B08": "nir", "SCL": "scl"},
    "aws-s3": {"B04": "red", "B08": "nir", "SCL": "scl"},
    "planetary-computer": {"B04": "B04", "B08": "B08", "SCL": "SCL"},
}
BUCKET = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"
PREFIX = "sentinel-s2-l2a-cogs"

GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES",
                GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
                GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2")


# ---------------------------------------------------------------- searching --
def search_stac(source, bbox, start, end, max_cloud):
    from pystac_client import Client
    modifier = None
    if source == "planetary-computer":
        try:
            import planetary_computer
        except ImportError as e:
            raise SystemExit("planetary-computer source needs: pip install planetary-computer") from e
        modifier = planetary_computer.sign_inplace
    client = Client.open(STAC_URLS[source], modifier=modifier)
    search = client.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{start}/{end}",
                           query={"eo:cloud_cover": {"lt": max_cloud}}, max_items=5000)
    return [it.to_dict() for it in search.items()]


def mgrs_tile(lon, lat) -> str:
    """MGRS 100 km square containing a point, e.g. '14SQE' for Cushing.

    Sentinel-2 tiles are these squares plus a 4.9 km overlap, so the tile with
    this name always contains the point. Ignores the Norway/Svalbard zone
    exceptions.
    """
    zone = int((lon + 180) // 6) + 1
    band = "CDEFGHJKLMNPQRSTUVWX"[int((lat + 80) // 8)]
    epsg = (32600 if lat >= 0 else 32700) + zone
    e, n = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
    col = ("STUVWXYZ", "ABCDEFGH", "JKLMNPQR")[zone % 3][int(e // 100000) - 1]
    row = "ABCDEFGHJKLMNPQRSTUV"[(int(n // 100000) + (5 if zone % 2 == 0 else 0)) % 20]
    return f"{zone}{band}{col}{row}"


def search_s3(bbox, start, end, max_cloud, tiles=None):
    """List the public bucket month by month for the tiles covering the bbox."""
    w, s, e, n = bbox
    if not tiles:
        tiles = sorted({mgrs_tile(x, y) for x in (w, e, (w + e) / 2) for y in (s, n, (s + n) / 2)})
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    items = []
    for tile in tiles:
        m = re.fullmatch(r"(\d{1,2})([C-X])([A-Z]{2})", tile)
        if not m:
            raise ValueError(f"bad MGRS tile {tile!r}")
        y, mo = d0.year, d0.month
        while (y, mo) <= (d1.year, d1.month):
            prefix = f"{PREFIX}/{m[1]}/{m[2]}/{m[3]}/{y}/{mo}/"
            r = requests.get(BUCKET, params={"list-type": "2", "prefix": prefix, "delimiter": "/"},
                             timeout=60)
            r.raise_for_status()
            for scene in re.findall(r"<Prefix>([^<]+_L2A)/</Prefix>", r.text):
                name = scene.rsplit("/", 1)[1]
                day = datetime.strptime(name.split("_")[2], "%Y%m%d").date()
                if not d0 <= day <= d1:
                    continue
                js = requests.get(f"{BUCKET}/{scene}/{name}.json", timeout=60)
                if js.status_code != 200:
                    continue
                it = js.json()
                if it["properties"].get("eo:cloud_cover", 100) < max_cloud:
                    items.append(it)
            y, mo = (y + 1, 1) if mo == 12 else (y, mo + 1)
    return items


# ------------------------------------------------------------- item reading --
def item_sun(item) -> tuple[float, float]:
    p = item["properties"]
    if "view:sun_elevation" in p:
        return float(p["view:sun_azimuth"]), float(p["view:sun_elevation"])
    return float(p["s2:mean_solar_azimuth"]), 90.0 - float(p["s2:mean_solar_zenith"])


def boa_add_offset(item) -> float:
    """Additive DN offset to apply before dividing by 10000.

    From processing baseline 04.00 (25 Jan 2022) L2A DNs carry +1000 so dark
    pixels are not clipped at zero. Earth Search may already have removed it.
    """
    p = item["properties"]
    if p.get("earthsearch:boa_offset_applied"):
        return 0.0
    try:
        baseline = float(p.get("s2:processing_baseline", "0"))
    except ValueError:
        baseline = 0.0
    return -1000.0 if baseline >= 4.0 else 0.0


def _href(item, source, band):
    return item["assets"][ASSETS[source][band]]["href"].replace(
        "s3://sentinel-cogs/", f"{BUCKET}/")


def read_item(item, source, bbox):
    """Read B04/B08 (10 m) and SCL (20 m, repeated to 10 m) inside bbox."""
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(_href(item, source, "B04")) as src:
            b = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
            win = from_bounds(*b, transform=src.transform)
            # Snap to an even pixel grid so the 20 m SCL pixels line up exactly.
            c0 = max(int(math.floor(win.col_off / 2)) * 2, 0)
            r0 = max(int(math.floor(win.row_off / 2)) * 2, 0)
            c1 = min(int(math.ceil((win.col_off + win.width) / 2)) * 2, src.width)
            r1 = min(int(math.ceil((win.row_off + win.height) / 2)) * 2, src.height)
            if c1 - c0 < 4 or r1 - r0 < 4:
                return None
            win = Window(c0, r0, c1 - c0, r1 - r0)
            red = src.read(1, window=win)
            transform = src.window_transform(win)
            crs = src.crs
        with rasterio.open(_href(item, source, "B08")) as src:
            nir = src.read(1, window=win)
        with rasterio.open(_href(item, source, "SCL")) as src:
            w20 = Window(c0 // 2, r0 // 2, (c1 - c0) // 2, (r1 - r0) // 2)
            scl = src.read(1, window=w20, boundless=True, fill_value=0)
            scl = np.repeat(np.repeat(scl, 2, axis=0), 2, axis=1)
    return red, nir, scl, transform, crs


# ----------------------------------------------------------------- download --
def download(bbox, start, end, out_dir, source="earth-search", max_cloud=60.0,
             max_aoi_cloud=0.05, max_nodata=0.2, tiles=None, overwrite=False):
    """Fetch every acceptable scene. Returns the manifest DataFrame.

    Scenes are filtered twice: by the whole-tile cloud percentage in the
    catalogue (cheap, coarse), then by the cloud fraction from the scene
    classification layer inside the bounding box (what actually matters).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if source == "aws-s3":
        items = search_s3(bbox, start, end, max_cloud, tiles)
    else:
        items = search_stac(source, bbox, start, end, max_cloud)
    log.info("%d catalogue items under %.0f%% tile cloud", len(items), max_cloud)

    rows = []
    best_per_day = {}
    for it in sorted(items, key=lambda i: i["properties"]["datetime"]):
        p = it["properties"]
        day = p["datetime"][:10]
        row = dict(id=it["id"], date=day, tile_cloud=p.get("eo:cloud_cover"), status="")
        try:
            got = read_item(it, source, bbox)
        except Exception as e:  # network errors, missing assets
            row["status"] = f"error: {e}"
            rows.append(row)
            continue
        if got is None:
            row["status"] = "outside_tile"
            rows.append(row)
            continue
        red, nir, scl, transform, crs = got
        n = scl.size
        nodata = float((scl == SCL_NODATA).sum() / n)
        valid = scl != SCL_NODATA
        nv = max(int(valid.sum()), 1)
        cloud = float(np.isin(scl, SCL_CLOUD).sum() / nv)
        snow = float((scl == SCL_SNOW).sum() / nv)
        row.update(aoi_nodata=round(nodata, 4), aoi_cloud=round(cloud, 4), aoi_snow=round(snow, 4))
        if nodata > max_nodata:
            row["status"] = "partial_coverage"
        elif cloud > max_aoi_cloud:
            row["status"] = "cloudy"
        else:
            az, el = item_sun(it)
            meta = dict(id=it["id"], datetime=p["datetime"], sun_azimuth=az, sun_elevation=el,
                        boa_add_offset=boa_add_offset(it), source=source,
                        processing_baseline=p.get("s2:processing_baseline"),
                        platform=p.get("platform"), tile_cloud=p.get("eo:cloud_cover"),
                        aoi_cloud_fraction=cloud, aoi_snow_fraction=snow,
                        aoi_nodata_fraction=nodata, bbox=list(bbox))
            prev = best_per_day.get(day)
            if prev is not None and prev[1] <= nodata:
                row["status"] = "duplicate_day"
            else:
                if prev is not None:
                    prev[0].unlink(missing_ok=True)
                    prev[0].with_suffix(".json").unlink(missing_ok=True)
                    for r in rows:
                        if r["id"] == prev[2]:
                            r["status"] = "duplicate_day"
                path = out_dir / f"{day}_{it['id']}.tif"
                write_scene(path, red, nir, scl, transform, crs, meta)
                best_per_day[day] = (path, nodata, it["id"])
                row.update(status="kept", path=str(path), sun_elevation=round(el, 2),
                           sun_azimuth=round(az, 2))
        log.info("%s %s %s", day, it["id"], row["status"])
        rows.append(row)
    manifest = pd.DataFrame(rows)
    mpath = out_dir / "manifest.csv"
    if mpath.exists() and not overwrite:
        manifest = pd.concat([pd.read_csv(mpath), manifest]).drop_duplicates("id", keep="last")
    manifest.to_csv(mpath, index=False)
    return manifest
