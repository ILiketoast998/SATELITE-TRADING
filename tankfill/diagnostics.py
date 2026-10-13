"""Observed pixels next to the fitted model, one row of panels per tank."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

INK = "#0b0b0b"
MUTED = "#52514e"


def _outline(ax, fit, d):
    """Rim (solid), lit-roof boundary (dashed) and shadow end (dotted)."""
    p = fit.params
    ux, uy = d
    cx, cy = p["dx"], p["dy"]
    kw = dict(fill=False, lw=0.8)
    ax.add_patch(Circle((cx, cy), p["R"], ec="#ffffff", **kw))
    ax.add_patch(Circle((cx + p["Li"] * ux, cy + p["Li"] * uy), p["R"], ec="#eda100", ls="--", **kw))
    ax.add_patch(Circle((cx + p["Le"] * ux, cy + p["Le"] * uy), p["R"], ec="#2a78d6", ls=":", **kw))


def plot_fits(tank_fits, shadow_dir, out_prefix, title="", per_page=12, label_fn=None):
    """Write PNG pages of observed | model | residual panels. Returns the paths."""
    items = [t for t in tank_fits if t.fit is not None and t.win is not None]
    paths = []
    if not items:
        return paths
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    pages = math.ceil(len(items) / per_page)
    for page in range(pages):
        chunk = items[page * per_page:(page + 1) * per_page]
        ncol_tanks = 2
        nrow = math.ceil(len(chunk) / ncol_tanks)
        fig, axes = plt.subplots(nrow, 3 * ncol_tanks, figsize=(3 * ncol_tanks * 2.1, nrow * 2.45),
                                 squeeze=False)
        for ax in axes.flat:
            ax.set_axis_off()
        for k, tf in enumerate(chunk):
            r, c = divmod(k, ncol_tanks)
            a_obs, a_mod, a_res = axes[r, 3 * c: 3 * c + 3]
            win, fit = tf.win, tf.fit
            obs = np.where(win.mask, win.obs, np.nan)
            lo, hi = np.nanpercentile(np.r_[win.obs[win.mask], fit.model[win.mask]], [1, 99])
            ext = win.extent()
            kw = dict(extent=ext, origin="upper", interpolation="nearest")
            a_obs.imshow(obs, cmap="gray", vmin=lo, vmax=hi, **kw)
            a_mod.imshow(fit.model, cmap="gray", vmin=lo, vmax=hi, **kw)
            res = np.where(win.mask, win.obs - fit.model, np.nan)
            lim = max(np.nanmax(np.abs(res)), 1e-6) if np.isfinite(res).any() else 1.0
            a_res.imshow(res, cmap="RdBu_r", vmin=-lim, vmax=lim, **kw)
            for ax in (a_obs, a_mod):
                _outline(ax, fit, shadow_dir)
                ax.set_xlim(ext[0], ext[1])
                ax.set_ylim(ext[2], ext[3])
            label = label_fn(tf) if label_fn else tf.row.get("id", "")
            a_obs.set_title(label, fontsize=7, loc="left", color=INK)
            a_mod.set_title("model", fontsize=7, color=MUTED)
            a_res.set_title(f"residual ±{lim:.3f}", fontsize=7, color=MUTED)
        fig.suptitle(f"{title}   rim — solid white · roof-shadow edge — dashed · ground-shadow end — dotted"
                     "   (masked pixels blank)", fontsize=8, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = Path(f"{out_prefix}_p{page + 1:02d}.png")
        fig.savefig(path, dpi=130, facecolor="#fcfcfb")
        plt.close(fig)
        paths.append(path)
    return paths


def measure_label(tf):
    r = tf.row
    if "fill" not in r:
        return f"{r.get('id')}  {r.get('flag')}"
    return (f"{r['id']}  {r['flag']}\nfill {r['fill']:.2f}±{r['fill_sigma']:.2f}  "
            f"H {r['height_m']:.1f} m\nLi {r['Li_m']:.1f}  Le {r['Le_m']:.1f}  nrmse {r['nrmse']:.2f}")
