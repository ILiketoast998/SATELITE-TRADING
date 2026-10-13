"""Command-line interface: `tankfill <command> --help` for details."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SITES = {
    # lon/lat boxes around the tank farms (W, S, E, N)
    "cushing": (-96.80, 35.88, -96.70, 35.97),
    "dongjiakou": (119.76, 35.56, 119.86, 35.64),
}


def _bbox(args):
    if args.site:
        return SITES[args.site]
    if not args.bbox:
        raise SystemExit("give --site or --bbox W S E N")
    return tuple(args.bbox)


def cmd_download(args):
    from .download import download
    m = download(_bbox(args), args.start, args.end, args.out, source=args.source,
                 max_cloud=args.max_tile_cloud, max_aoi_cloud=args.max_aoi_cloud,
                 tiles=args.tile)
    kept = (m.status == "kept").sum() if len(m) else 0
    print(f"{kept} scenes kept in {args.out} (see manifest.csv for the rest)")


def cmd_detect(args):
    from .detect import DetectConfig, detect_tanks
    from .diagnostics import plot_fits
    from .scene import load_scene
    scene = load_scene(args.scene)
    if scene.sun_elevation < 55:
        logging.warning("sun elevation %.0f deg: detect expects a high-sun (summer) scene",
                        scene.sun_elevation)
    cfg = DetectConfig(band=args.band, min_diameter=args.min_diameter, max_diameter=args.max_diameter,
                       min_contrast=args.min_contrast, ignore_shadows=args.ignore_shadows)
    cands = pd.read_csv(args.candidates) if args.candidates else None
    tanks, fits = detect_tanks(scene, cfg, cands, workers=args.workers)
    tanks.to_csv(args.out, index=False)
    print(f"{len(tanks)} tanks -> {args.out}")
    if args.diagnostics:
        def label(tf):
            p = tf.fit.params
            return f"{tf.row['id']}  D {2 * p['R']:.1f} m"
        for p in plot_fits(fits, scene.shadow_direction(), args.diagnostics,
                           title=f"detect {scene.date}", label_fn=label):
            print(f"diagnostics -> {p}")


def _measure_cfg(args):
    from .measure import MeasureConfig
    return MeasureConfig(band=args.band, psf_sigma=args.psf_sigma, h_min=args.h_min, h_max=args.h_max,
                         near_full_freeboard=args.near_full_freeboard)


def cmd_measure(args):
    from .diagnostics import measure_label, plot_fits
    from .measure import measure_scene
    from .scene import load_scene
    scene = load_scene(args.scene)
    if scene.sun_elevation > 45:
        logging.warning("sun elevation %.0f deg: shadows are short, expect poor fits",
                        scene.sun_elevation)
    tanks = pd.read_csv(args.tanks)
    fit = measure_scene(scene, tanks, _measure_cfg(args), workers=args.workers)
    fit.table.to_csv(args.out, index=False)
    t = fit.table
    use = t[t.flag.isin(["ok", "near_full"])]
    print(f"{scene.date}: offset dx={fit.offset[0]:.1f} m dy={fit.offset[1]:.1f} m; flags: "
          + ", ".join(f"{k} {v}" for k, v in t.flag.value_counts().items()))
    print(f"usable tanks {len(use)}: {use.volume_bbl.sum() / 1e6:.2f} M bbl of "
          f"{use.capacity_bbl.sum() / 1e6:.2f} M bbl capacity -> {args.out}")
    if args.diagnostics:
        for p in plot_fits(fit.tanks, scene.shadow_direction(), args.diagnostics,
                           title=f"{scene.date}  sun el {scene.sun_elevation:.1f}°",
                           label_fn=measure_label):
            print(f"diagnostics -> {p}")


def cmd_heights(args):
    from .heights import lock_heights
    tanks = pd.read_csv(args.tanks)
    res = [pd.read_csv(p) for p in args.results]
    out = lock_heights(tanks, res, min_scenes=args.min_scenes)
    out.to_csv(args.out, index=False)
    n = out.height_locked.astype(bool).sum()
    fixed = (out.roof_type == "fixed_suspect").sum()
    print(f"locked {n}/{len(out)} heights; {fixed} tanks look fixed-roof (or full every time) -> {args.out}")


def cmd_timeseries(args):
    from .timeseries import run_timeseries
    tanks = pd.read_csv(args.tanks)
    totals = run_timeseries(args.scenes, tanks, args.out, _measure_cfg(args),
                            max_sun_elevation=args.max_sun_elevation, max_aoi_cloud=args.max_aoi_cloud,
                            lock=not args.no_lock, diagnostics=args.diagnostics, eia_csv=args.eia,
                            title=args.title, workers=args.workers)
    pd.set_option("display.width", 160)
    print(totals.to_string(index=False))
    print(f"-> {Path(args.out) / 'totals.csv'}, {Path(args.out) / 'totals.png'}")


def cmd_simulate(args):
    from .synthetic import default_tanks, render_scene, truth_table
    out = Path(args.out)
    tk = default_tanks(args.seed)
    rng = np.random.default_rng(args.seed + 99)
    out.mkdir(parents=True, exist_ok=True)
    truth_table(tk).to_csv(out / "truth.csv", index=False)
    render_scene(out / "summer" / "2024-06-20_SYNTH.tif", tk, sun_azimuth=125, sun_elevation=73,
                 date="2024-06-20", offset=tuple(rng.uniform(-6, 6, 2)), seed=args.seed)
    rows = []
    for k, (day, el, az) in enumerate([("2023-11-20", 34.0, 162.0), ("2023-12-10", 31.0, 163.0),
                                       ("2023-12-30", 30.5, 162.0), ("2024-01-19", 32.0, 160.0),
                                       ("2024-02-08", 37.0, 157.0)]):
        fills = {t.id: float(min(max(t.fill + 0.04 * k * (1 if i % 2 else -1), 0.05), 0.95))
                 for i, t in tk.iterrows()}
        rows += [dict(date=day, id=i, true_fill=f) for i, f in fills.items()]
        render_scene(out / "winter" / f"{day}_SYNTH.tif", tk, sun_azimuth=az, sun_elevation=el,
                     date=day, offset=tuple(rng.uniform(-8, 8, 2)), seed=args.seed + k + 1, fills=fills)
    pd.DataFrame(rows).to_csv(out / "truth_fills.csv", index=False)
    print(f"synthetic scenes in {out}/summer and {out}/winter, truth in {out}/truth*.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tankfill", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch Sentinel-2 L2A B04/B08/SCL for a bbox")
    d.add_argument("--site", choices=sorted(SITES))
    d.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"))
    d.add_argument("--start", required=True, help="YYYY-MM-DD")
    d.add_argument("--end", required=True, help="YYYY-MM-DD")
    d.add_argument("--out", default="data/scenes")
    d.add_argument("--source", default="earth-search", choices=["earth-search", "planetary-computer", "aws-s3"])
    d.add_argument("--tile", nargs="*", help="MGRS tiles for --source aws-s3 (default: from bbox)")
    d.add_argument("--max-tile-cloud", type=float, default=60.0, help="catalogue cloud %% per tile")
    d.add_argument("--max-aoi-cloud", type=float, default=0.05, help="cloud fraction inside bbox")
    d.set_defaults(func=cmd_download)

    t = sub.add_parser("detect", help="find tanks in a high-sun summer scene")
    t.add_argument("scene")
    t.add_argument("--out", default="tanks.csv")
    t.add_argument("--band", default="B04")
    t.add_argument("--min-diameter", type=float, default=25.0)
    t.add_argument("--max-diameter", type=float, default=110.0)
    t.add_argument("--min-contrast", type=float, default=0.08)
    t.add_argument("--candidates", help="CSV with lon,lat[,diameter_m] to refine instead of searching")
    t.add_argument("--ignore-shadows", action="store_true", help="fit plain discs (shows the radius bias)")
    t.add_argument("--diagnostics", help="PNG prefix for observed-vs-model panels")
    t.add_argument("--workers", type=int, help="processes (default: all CPUs)")
    t.set_defaults(func=cmd_detect)

    def measure_opts(p):
        p.add_argument("--band", default="B08")
        p.add_argument("--psf-sigma", type=float, default=4.5, help="sensor blur, metres")
        p.add_argument("--h-min", type=float, default=5.0)
        p.add_argument("--h-max", type=float, default=25.0)
        p.add_argument("--near-full-freeboard", type=float, default=3.0)
        p.add_argument("--workers", type=int, help="processes (default: all CPUs)")

    m = sub.add_parser("measure", help="measure shadows and fill on a low-sun scene")
    m.add_argument("scene")
    m.add_argument("--tanks", required=True)
    m.add_argument("--out", default="measure.csv")
    m.add_argument("--diagnostics", help="PNG prefix for observed-vs-model panels")
    measure_opts(m)
    m.set_defaults(func=cmd_measure)

    h = sub.add_parser("heights", help="lock tank heights to the median of several measure results")
    h.add_argument("results", nargs="+")
    h.add_argument("--tanks", required=True)
    h.add_argument("--out", default="tanks_locked.csv")
    h.add_argument("--min-scenes", type=int, default=2)
    h.set_defaults(func=cmd_heights)

    s = sub.add_parser("timeseries", help="measure every usable winter scene and total the barrels")
    s.add_argument("scenes", help="folder written by download")
    s.add_argument("--tanks", required=True)
    s.add_argument("--out", default="timeseries")
    s.add_argument("--max-sun-elevation", type=float, default=40.0)
    s.add_argument("--max-aoi-cloud", type=float, default=0.05)
    s.add_argument("--no-lock", action="store_true", help="do not lock heights first")
    s.add_argument("--diagnostics", action="store_true")
    s.add_argument("--eia", help="CSV of EIA weekly stocks (date, thousand barrels) to overlay")
    s.add_argument("--title", default="Estimated crude in floating-roof tanks")
    measure_opts(s)
    s.set_defaults(func=cmd_timeseries)

    y = sub.add_parser("simulate", help="write synthetic summer + winter scenes with known answers")
    y.add_argument("--out", default="data/synthetic")
    y.add_argument("--seed", type=int, default=0)
    y.set_defaults(func=cmd_simulate)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
