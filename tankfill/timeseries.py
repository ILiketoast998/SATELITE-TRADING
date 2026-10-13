"""Run measure over every usable winter scene and total the barrels per date."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .diagnostics import measure_label, plot_fits  # noqa: E402
from .heights import lock_heights  # noqa: E402
from .measure import MeasureConfig, measure_scene  # noqa: E402
from .scene import list_scenes, load_scene  # noqa: E402

log = logging.getLogger(__name__)

USABLE = ("ok", "near_full")


def select_scenes(scenes_dir, max_sun_elevation=40.0, max_aoi_cloud=0.05, max_snow=0.02):
    keep, skipped = [], []
    for p in list_scenes(scenes_dir):
        m = json.loads(p.with_suffix(".json").read_text())
        why = None
        if m["sun_elevation"] > max_sun_elevation:
            why = f"sun {m['sun_elevation']:.0f} deg too high: shadows too short"
        elif m.get("aoi_cloud_fraction", 0) > max_aoi_cloud:
            why = "cloud"
        elif m.get("aoi_snow_fraction", 0) > max_snow:
            why = "snow"
        (skipped if why else keep).append((p, why))
    return [p for p, _ in keep], skipped


def _measure_all(paths, tanks, cfg, out_dir, diagnostics, workers=None):
    results = []
    for p in paths:
        scene = load_scene(p)
        fit = measure_scene(scene, tanks, cfg, workers)
        if out_dir is not None:
            fit.table.to_csv(out_dir / f"{scene.date}_{scene.scene_id}.csv", index=False)
            if diagnostics:
                plot_fits(fit.tanks, scene.shadow_direction(),
                          out_dir / "diagnostics" / f"{scene.date}_{scene.scene_id}",
                          title=f"{scene.date}  sun el {scene.sun_elevation:.1f}°",
                          label_fn=measure_label)
        n_ok = int(fit.table.flag.isin(USABLE).sum())
        log.info("%s: %d/%d tanks usable, offset (%.1f, %.1f) m", scene.date, n_ok, len(tanks),
                 *fit.offset)
        results.append(fit.table)
    return results


def site_totals(results: list[pd.DataFrame], tanks: pd.DataFrame, max_gap_days=30) -> pd.DataFrame:
    """Per-date totals.

    total_bbl_measured: sum over tanks usable on that date (changes with how
        many tanks happened to fit, so it jumps when tanks drop out).
    total_bbl_filled: every floating-roof tank counted; a tank missing on a
        date takes its fill from the nearest usable date within max_gap_days.
    site_fill: total measured volume / total measured capacity.
    """
    allres = pd.concat(results, ignore_index=True)
    allres["date"] = pd.to_datetime(allres["date"])
    good = allres[allres.flag.isin(USABLE)]
    excluded = set(tanks.loc[tanks.get("roof_type", pd.Series(dtype=str)).eq("fixed_suspect"), "id"])
    ids = [i for i in tanks.id if i not in excluded]
    caps = good.groupby("id").capacity_bbl.median()
    rows = []
    for d in sorted(allres.date.unique()):
        g = good[good.date == d]
        meas = g.volume_bbl.sum()
        cap = g.capacity_bbl.sum()
        var = float(((g.fill_sigma.fillna(0) * g.capacity_bbl) ** 2).sum())
        filled_total, n_filled = 0.0, 0
        for tid in ids:
            row = g[g.id == tid]
            if len(row):
                filled_total += float(row.volume_bbl.iloc[0])
                continue
            hist = good[good.id == tid]
            if hist.empty:
                continue
            gap = (hist.date - d).abs()
            k = gap.idxmin()
            if gap[k].days <= max_gap_days:
                filled_total += float(hist.fill[k] * caps[tid])
                n_filled += 1
        rows.append(dict(date=pd.Timestamp(d).date().isoformat(), n_tanks=len(ids),
                         n_measured=len(g), n_gap_filled=n_filled,
                         total_bbl_measured=round(meas), total_bbl_measured_sigma=round(np.sqrt(var)),
                         capacity_bbl_measured=round(cap),
                         site_fill=round(meas / cap, 4) if cap > 0 else np.nan,
                         total_bbl_filled=round(filled_total)))
    return pd.DataFrame(rows)


def read_eia(path) -> pd.DataFrame:
    """EIA weekly Cushing stocks (series W_EPC0_SAX_YCUOK_MBBL), thousand barrels.

    Accepts a CSV whose first column is a date and second the value, e.g. the
    "Download Data" export from the EIA page, saved as CSV.
    """
    if str(path).lower().endswith((".xls", ".xlsx")):
        # The EIA "Download Data (XLS File)" link: sheet "Data 1", 2 title rows.
        # Reading .xls needs `pip install xlrd`.
        df = pd.read_excel(path, sheet_name="Data 1", skiprows=2)
    else:
        df = pd.read_csv(path, comment="#")
    df = df.iloc[:, :2]
    df.columns = ["date", "kbbl"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["kbbl"] = pd.to_numeric(df["kbbl"], errors="coerce")
    return df.dropna()


def plot_totals(totals: pd.DataFrame, path, title="", eia: pd.DataFrame | None = None):
    t = totals.copy()
    t["date"] = pd.to_datetime(t["date"])
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1]))
    blue, orange, ink, muted, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    ax.plot(t.date, t.total_bbl_filled / 1e6, color=blue, lw=2, marker="o", ms=4,
            label="All floating-roof tanks (gaps filled from nearest date)")
    ax.errorbar(t.date, t.total_bbl_measured / 1e6, yerr=t.total_bbl_measured_sigma / 1e6,
                color=muted, lw=0, elinewidth=1, marker="o", ms=3, mfc="none",
                label="Tanks measured on that date only (±1σ)")
    if eia is not None and len(eia):
        e = eia[(eia.date >= t.date.min() - pd.Timedelta(days=14))
                & (eia.date <= t.date.max() + pd.Timedelta(days=14))]
        ax.step(e.date, e.kbbl / 1e3, where="post", color=orange, lw=2, label="EIA weekly stocks")
    ax.set_ylabel("Million barrels", color=ink)
    ax.set_title(title, loc="left", fontsize=11, color=ink)
    ax.legend(frameon=False, fontsize=8, loc="best")
    share = t.n_measured / t.n_tanks.where(t.n_tanks > 0)
    ax2.bar(t.date, share * 100, width=2.0, color=blue)
    ax2.set_ylabel("% tanks\nmeasured", color=ink)
    ax2.set_ylim(0, 100)
    for a in (ax, ax2):
        a.grid(axis="y", color=grid, lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(muted)
        a.tick_params(colors=muted, labelsize=8)
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="#fcfcfb")
    plt.close(fig)


def run_timeseries(scenes_dir, tanks: pd.DataFrame, out_dir, cfg: MeasureConfig | None = None,
                   max_sun_elevation=40.0, max_aoi_cloud=0.05, max_snow=0.02,
                   lock=True, diagnostics=False, eia_csv=None, title="Site total", workers=None):
    cfg = cfg or MeasureConfig()
    out_dir = Path(out_dir)
    (out_dir / "scenes").mkdir(parents=True, exist_ok=True)
    paths, skipped = select_scenes(scenes_dir, max_sun_elevation, max_aoi_cloud, max_snow)
    for p, why in skipped:
        log.info("skip %s: %s", p.name, why)
    if not paths:
        raise SystemExit("no usable winter scenes (check --max-sun-elevation and the scene folder)")
    log.info("%d usable scenes", len(paths))

    locked = tanks.get("height_locked", pd.Series(False, index=tanks.index))
    locked = locked.astype(str).str.lower().isin(["true", "1"])
    if lock and locked.mean() < 0.5:
        if len(paths) < 2:
            log.warning("only one scene: heights cannot be locked, fitting them per scene")
        else:
            log.info("pass A: measuring with free heights to lock them")
            first = _measure_all(paths, tanks, cfg, None, False, workers)
            tanks = lock_heights(tanks, first, h_min=cfg.h_min, h_max=cfg.h_max)
            tanks.to_csv(out_dir / "tanks_locked.csv", index=False)
            log.info("locked %d/%d heights", int(tanks.height_locked.astype(bool).sum()), len(tanks))

    results = _measure_all(paths, tanks, cfg, out_dir / "scenes", diagnostics, workers)
    totals = site_totals(results, tanks)
    totals.to_csv(out_dir / "totals.csv", index=False)
    eia = read_eia(eia_csv) if eia_csv else None
    plot_totals(totals, out_dir / "totals.png", title=title, eia=eia)
    return totals
