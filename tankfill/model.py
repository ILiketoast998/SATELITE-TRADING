"""Forward model of a floating-roof tank as seen by a 10 m sensor.

Geometry, in a local frame centred on the tank (x east, y north, metres):

* the wall top (rim) is a circle of radius R at height H;
* the floating roof sits at height h = fill * H;
* the sun is at grid azimuth A (clockwise from grid north) and elevation el.
  The horizontal unit vector *away* from the sun is d = -(sin A, cos A).

Then, with  Le = H / tan(el)  and  Li = (H - h) / tan(el):

* exterior shadow on the ground = stadium(segment 0 -> Le*d, radius R) minus the
  tank footprint disc;
* lit roof = disc(0, R) intersected with disc(Li*d, R)  (a roof point is lit if
  the ray towards the sun passes inside the rim circle);
* interior shadow = disc(0, R) minus the lit roof (a crescent on the sunward side).

Every class is rendered on a fine grid (default 2 m) with anti-aliased edges, so
the coverage changes continuously with Li and Le and the optimiser sees a
continuous objective. The fine coverage maps are blurred with a Gaussian point
spread function and box-averaged into the sensor's 10 m pixels.

Only two coverage maps are needed because there are three brightness levels:

    pixel = g * (1 - S) + r * L + s * (S - L)

with L = blurred lit-roof coverage and S = blurred "roof or shadow" coverage.
g, r, s (ground, lit roof, shadow) enter linearly and are solved by least squares.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def coverage(sdf: np.ndarray, cell: float) -> np.ndarray:
    """Anti-aliased area coverage from a signed distance (negative = inside)."""
    return np.clip(0.5 - sdf / cell, 0.0, 1.0)


def fine_coverages(X, Y, R, Li, Le, d, cell):
    """Return (lit_roof, roof_or_shadow) coverage on the fine grid.

    X, Y are coordinates relative to the tank centre; d is the unit shadow
    direction (pointing away from the sun).
    """
    ux, uy = d
    sd0 = np.hypot(X, Y) - R                                   # rim disc
    sd1 = np.hypot(X - Li * ux, Y - Li * uy) - R               # rim shifted onto roof
    sd_lit = np.maximum(sd0, sd1)                              # intersection
    t = np.clip(X * ux + Y * uy, 0.0, max(Le, 0.0))
    sd_stad = np.hypot(X - t * ux, Y - t * uy) - R             # tank + ground shadow
    return coverage(sd_lit, cell), coverage(sd_stad, cell)


def fine_classes(X, Y, R, Li, Le, d, cell):
    """Separate class coverages (ground, roof, interior shadow, exterior shadow).

    Used by the synthetic-scene generator and the diagnostics, not by the fit.
    """
    ux, uy = d
    sd0 = np.hypot(X, Y) - R
    sd1 = np.hypot(X - Li * ux, Y - Li * uy) - R
    t = np.clip(X * ux + Y * uy, 0.0, max(Le, 0.0))
    sd_stad = np.hypot(X - t * ux, Y - t * uy) - R
    c0 = coverage(sd0, cell)
    cl = coverage(np.maximum(sd0, sd1), cell)
    cs = coverage(sd_stad, cell)
    return 1.0 - cs, cl, c0 - cl, cs - c0


class TankWindow:
    """A rectangular block of image pixels around one tank, plus its fine grid.

    Parameters
    ----------
    image : 2-D array of reflectance for the whole scene.
    transform : affine (rasterio) transform of the scene, north-up, square pixels.
    cx, cy : tank centre in scene CRS metres.
    shadow_dir : unit vector (grid frame) pointing away from the sun.
    radius_max, shadow_max : largest radius / exterior shadow length the fit may
        explore. They set the window size.
    psf_sigma : Gaussian sensor blur, metres (excluding the pixel box itself).
    sub : fine cells per pixel side (5 -> 2 m cells for 10 m pixels).
    margin : metres of ground context around tank + longest shadow.
    """

    def __init__(self, image, transform, cx, cy, shadow_dir, radius_max,
                 shadow_max, psf_sigma=4.5, sub=5, margin=15.0, valid=None):
        self.cx, self.cy = float(cx), float(cy)
        self.d = (float(shadow_dir[0]), float(shadow_dir[1]))
        px = float(transform.a)
        if abs(transform.e + px) > 1e-6 or transform.b or transform.d:
            raise ValueError("expected a north-up raster with square pixels")
        self.px = px
        self.sub = sub
        self.cell = px / sub
        x0, y0 = transform.c, transform.f
        self.x0, self.y0 = x0, y0

        # Bounding box of the tank disc plus its longest possible shadow.
        ex = self.d[0] * shadow_max
        ey = self.d[1] * shadow_max
        xmin = cx + min(0.0, ex) - radius_max - margin
        xmax = cx + max(0.0, ex) + radius_max + margin
        ymin = cy + min(0.0, ey) - radius_max - margin
        ymax = cy + max(0.0, ey) + radius_max + margin
        nrows, ncols = image.shape
        c0 = max(int(np.floor((xmin - x0) / px)), 0)
        c1 = min(int(np.ceil((xmax - x0) / px)), ncols)
        r0 = max(int(np.floor((y0 - ymax) / px)), 0)
        r1 = min(int(np.ceil((y0 - ymin) / px)), nrows)
        if c1 - c0 < 3 or r1 - r0 < 3:
            raise ValueError("tank window falls outside the image")
        self.rows = (r0, r1)
        self.cols = (c0, c1)
        self.obs = np.asarray(image[r0:r1, c0:c1], dtype=float)
        mask = np.isfinite(self.obs)
        if valid is not None:
            mask &= valid[r0:r1, c0:c1]
        self.mask = mask

        # Fine grid, padded so the blur has room at the window edges.
        self.pad = int(np.ceil(3.0 * psf_sigma / px)) if psf_sigma > 0 else 0
        h = (r1 - r0 + 2 * self.pad) * sub
        w = (c1 - c0 + 2 * self.pad) * sub
        xs = x0 + (c0 - self.pad) * px + (np.arange(w) + 0.5) * self.cell
        ys = y0 - (r0 - self.pad) * px - (np.arange(h) + 0.5) * self.cell
        self.Xrel = xs[None, :] - self.cx
        self.Yrel = ys[:, None] - self.cy
        self.sigma_fine = psf_sigma / self.cell
        self.shape = (r1 - r0, c1 - c0)

    def extent(self):
        """matplotlib imshow extent of the window, metres relative to the tank centre."""
        c0, c1 = self.cols
        r0, r1 = self.rows
        return (self.x0 + c0 * self.px - self.cx, self.x0 + c1 * self.px - self.cx,
                self.y0 - r1 * self.px - self.cy, self.y0 - r0 * self.px - self.cy)

    # -- rendering -----------------------------------------------------------
    def _down(self, a):
        if self.sigma_fine > 0:
            a = gaussian_filter(a, self.sigma_fine, mode="nearest")
        s, p = self.sub, self.pad
        a = a[p * s: a.shape[0] - p * s, p * s: a.shape[1] - p * s]
        h, w = self.shape
        return a.reshape(h, s, w, s).mean(axis=(1, 3))

    def coverages(self, dx, dy, R, Li, Le):
        """Blurred, pixel-averaged (L, S) maps for the given geometry."""
        X = self.Xrel - dx
        Y = self.Yrel - dy
        lit, stad = fine_coverages(X, Y, R, Li, Le, self.d, self.cell)
        return self._down(lit), self._down(stad)

    def solve(self, dx, dy, R, Li, Le, s_fixed=None):
        """Least-squares brightness levels for one geometry.

        Returns (sse, levels(g, r, s), model image, n_valid_pixels).
        """
        L, S = self.coverages(dx, dy, R, Li, Le)
        m = self.mask
        y = self.obs[m]
        cols = [1.0 - S[m], L[m], S[m] - L[m]]
        if s_fixed is None:
            A = np.stack(cols, axis=1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=1e-6)
            g, r, s = coef
        else:
            A = np.stack(cols[:2], axis=1)
            coef, *_ = np.linalg.lstsq(A, y - s_fixed * cols[2], rcond=1e-6)
            (g, r), s = coef, s_fixed
        model = g * (1.0 - S) + r * L + s * (S - L)
        resid = (self.obs - model)[m]
        return float(resid @ resid), (float(g), float(r), float(s)), model, int(m.sum())
