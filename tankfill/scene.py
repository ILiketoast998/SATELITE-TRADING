"""Loading downloaded scenes and converting between lon/lat and scene pixels."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer

# Band order inside the GeoTIFFs written by `tankfill download`.
BAND_INDEX = {"B04": 1, "B08": 2, "SCL": 3}

# Sentinel-2 L2A scene classification codes.
SCL_NODATA = 0
SCL_SATURATED = 1
SCL_CLOUD = (8, 9, 10)          # medium prob., high prob., thin cirrus
SCL_SNOW = 11


@dataclass
class Scene:
    path: Path
    meta: dict
    transform: object
    crs: object
    bands: dict = field(default_factory=dict)

    @property
    def sun_azimuth(self) -> float:
        return float(self.meta["sun_azimuth"])

    @property
    def sun_elevation(self) -> float:
        return float(self.meta["sun_elevation"])

    @property
    def date(self) -> str:
        return str(self.meta["datetime"])[:10]

    @property
    def scene_id(self) -> str:
        return str(self.meta.get("id", self.path.stem))

    def reflectance(self, band: str) -> np.ndarray:
        """Surface reflectance (0-1 scale) with nodata as NaN."""
        dn = self.bands[band].astype(float)
        out = (dn + float(self.meta.get("boa_add_offset", 0.0))) / 10000.0
        out[dn == 0] = np.nan
        return out

    def scl(self) -> np.ndarray | None:
        return self.bands.get("SCL")

    # -- coordinates ---------------------------------------------------------
    def lonlat_to_xy(self, lon, lat):
        t = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        return t.transform(lon, lat)

    def xy_to_lonlat(self, x, y):
        t = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        return t.transform(x, y)

    def grid_north_offset(self) -> float:
        """Degrees to add to a true azimuth to get the azimuth in the map grid.

        UTM grid north is not true north away from the central meridian
        (about 1.3 degrees at Cushing). Measured numerically, so no sign
        convention can be got wrong.
        """
        lon, lat = self.centre_lonlat()
        t = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        x0, y0 = t.transform(lon, lat)
        x1, y1 = t.transform(lon, lat + 0.01)
        return math.degrees(math.atan2(x1 - x0, y1 - y0))

    def centre_lonlat(self):
        h, w = next(iter(self.bands.values())).shape
        t = self.transform
        x, y = t.c + t.a * w / 2.0, t.f + t.e * h / 2.0
        return self.xy_to_lonlat(x, y)

    def shadow_direction(self) -> tuple[float, float]:
        """Unit vector in the grid frame pointing away from the sun."""
        az = math.radians(self.sun_azimuth + self.grid_north_offset())
        return (-math.sin(az), -math.cos(az))


def load_scene(path, bands=("B04", "B08", "SCL")) -> Scene:
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    with rasterio.open(path) as src:
        data = {b: src.read(BAND_INDEX[b]) for b in bands if BAND_INDEX[b] <= src.count}
        return Scene(path=path, meta=meta, transform=src.transform, crs=src.crs, bands=data)


def list_scenes(directory) -> list[Path]:
    return sorted(p for p in Path(directory).glob("*.tif") if p.with_suffix(".json").exists())


def write_scene(path, red, nir, scl, transform, crs, meta):
    """Write a scene in the layout `load_scene` expects (used by download and simulate)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", height=red.shape[0], width=red.shape[1], count=3,
                   dtype="uint16", crs=crs, transform=transform, compress="deflate",
                   nodata=0)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(red.astype(np.uint16), 1)
        dst.write(nir.astype(np.uint16), 2)
        dst.write(scl.astype(np.uint16), 3)
        dst.set_band_description(1, "B04")
        dst.set_band_description(2, "B08")
        dst.set_band_description(3, "SCL")
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path
