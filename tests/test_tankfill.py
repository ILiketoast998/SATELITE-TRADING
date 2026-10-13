import math

import numpy as np
import pandas as pd
import pytest
from affine import Affine

from tankfill.detect import DetectConfig, detect_tanks
from tankfill.download import mgrs_tile
from tankfill.fit import fit_tank
from tankfill.heights import lock_heights
from tankfill.measure import measure_scene
from tankfill.model import TankWindow, fine_classes
from tankfill.scene import load_scene
from tankfill.synthetic import render_scene, truth_table


def small_farm():
    rows = [("A", 200, 200, 30, 16, 0.3), ("B", 450, 200, 22, 14, 0.7), ("C", 700, 220, 36, 18, 0.5),
            ("D", 220, 470, 26, 12, 0.85), ("E", 470, 480, 34, 17, 0.2), ("F", 720, 470, 28, 15, 0.6)]
    df = pd.DataFrame(rows, columns=["id", "x", "y", "R", "H", "fill"])
    df["roof"] = "floating"
    return df


@pytest.fixture(scope="module")
def winter(tmp_path_factory):
    d = tmp_path_factory.mktemp("w")
    tk = small_farm()
    p = render_scene(d / "w.tif", tk, sun_azimuth=163, sun_elevation=31, size_m=900,
                     offset=(7.0, -5.0), seed=3)
    return load_scene(p), truth_table(tk)


def test_classes_partition_unity():
    X, Y = np.meshgrid(np.linspace(-60, 60, 61), np.linspace(-60, 60, 61))
    parts = fine_classes(X, Y, 20.0, 8.0, 25.0, (0.0, 1.0), 2.0)
    assert np.allclose(sum(parts), 1.0)
    assert all((p >= -1e-12).all() for p in parts)
    # No interior shadow when the roof is at the top.
    _, _, interior, _ = fine_classes(X, Y, 20.0, 0.0, 25.0, (0.0, 1.0), 2.0)
    assert interior.max() == 0


def test_nelder_mead_moves_from_zero(winter):
    """Li starts at exactly 0; with scipy's default simplex it would not move."""
    scene, truth = winter
    t = truth.iloc[0]
    x, y = scene.lonlat_to_xy(t.lon, t.lat)
    tan_el = math.tan(math.radians(scene.sun_elevation))
    win = TankWindow(scene.reflectance("B08"), scene.transform, x, y, scene.shadow_direction(),
                     t.diameter_m / 2 + 20, 25 / tan_el + 20)
    init = dict(dx=7.0, dy=-5.0, R=t.diameter_m / 2, Li=0.0, Le=t.height_m / tan_el)
    r = fit_tank(win, init, ["Li", "Le"], {"Li": (0, 45), "Le": (5, 45)}, grid_step=0,
                 uncertainty=False)
    true_li = (1 - t.true_fill) * t.height_m / tan_el
    assert abs(r.params["Li"] - true_li) < 4.0


def test_measure_recovers_offset_and_fill(winter):
    scene, truth = winter
    res = measure_scene(scene, truth.drop(columns=["height_m"]))
    assert abs(res.offset[0] - 7.0) < 2.0 and abs(res.offset[1] + 5.0) < 2.0
    m = res.table.merge(truth[["id", "true_fill", "height_m"]], on="id", suffixes=("", "_true"))
    usable = m[m.flag.isin(["ok", "near_full"])]
    assert len(usable) >= 5
    assert (usable.fill - usable.true_fill).abs().mean() < 0.08
    assert (usable.height_m - usable.height_m_true).abs().mean() < 2.0


def test_locked_height_measure(winter):
    scene, truth = winter
    tanks = truth.copy()
    tanks["height_locked"] = True
    tanks["height_sigma_m"] = 0.3
    res = measure_scene(scene, tanks)
    m = res.table.merge(truth[["id", "true_fill"]], on="id")
    usable = m[m.flag.isin(["ok", "near_full"])]
    assert (usable.fill - usable.true_fill).abs().mean() < 0.06


def test_detect_radius_unbiased_only_with_shadows(tmp_path):
    tk = small_farm()
    p = render_scene(tmp_path / "s.tif", tk, sun_azimuth=125, sun_elevation=70, size_m=900, seed=5)
    scene = load_scene(p)
    truth = truth_table(tk)
    errs = {}
    for ignore in (False, True):
        found, _ = detect_tanks(scene, DetectConfig(ignore_shadows=ignore))
        assert len(found) == len(tk)
        xy = np.c_[found.lon, found.lat]
        k = [int(np.argmin(np.hypot(*(np.c_[truth.lon, truth.lat] - v).T))) for v in xy]
        errs[ignore] = (found.diameter_m.values - truth.diameter_m.values[k]).mean()
    assert abs(errs[False]) < 1.5
    assert errs[True] < -1.0


def test_lock_heights_and_fixed_roof():
    tanks = pd.DataFrame(dict(id=["A", "B"], lon=[0, 0], lat=[0, 0], diameter_m=[50, 50],
                              height_m=[np.nan, np.nan]))
    res = [pd.DataFrame(dict(id=["A", "B"], flag=["ok", "near_full"], height_m=[h, 15.0],
                             fill=[0.4, 0.97], height_locked=[False, False]))
           for h in (14.0, 15.0, 16.5)]
    out = lock_heights(tanks, res)
    a, b = out.set_index("id").loc["A"], out.set_index("id").loc["B"]
    assert a.height_m == 15.0 and a.height_locked
    assert a.roof_type == "floating" and b.roof_type == "fixed_suspect"


def test_mgrs_tile():
    assert mgrs_tile(-96.75, 35.95) == "14SQE"
    assert mgrs_tile(119.8, 35.6) == "50SQE"


def test_window_rejects_rotated_raster():
    img = np.zeros((50, 50))
    tr = Affine(10, 1, 0, 1, -10, 500)
    with pytest.raises(ValueError):
        TankWindow(img, tr, 250, 250, (0, 1), 20, 20)
