"""Optimising tank geometry against observed pixels."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .model import TankWindow

PARAMS = ("dx", "dy", "R", "Li", "Le")


@dataclass
class FitResult:
    params: dict
    sse: float
    levels: tuple            # (ground, roof, shadow)
    model: np.ndarray
    npix: int
    nfree_levels: int
    sigma: dict = field(default_factory=dict)   # 1-sigma of free params
    cov: np.ndarray | None = None
    free: tuple = ()

    @property
    def rmse(self) -> float:
        return float(np.sqrt(self.sse / max(self.npix, 1)))

    def r2(self, win: TankWindow) -> float:
        y = win.obs[win.mask]
        sst = float(((y - y.mean()) ** 2).sum())
        return 1.0 - self.sse / sst if sst > 0 else float("nan")

    @property
    def contrast(self) -> float:
        """Lit-minus-shadow brightness: the signal the shadow lengths ride on."""
        g, r, s = self.levels
        return 0.5 * (g + r) - s

    @property
    def nrmse(self) -> float:
        c = self.contrast
        return self.rmse / c if c > 1e-6 else float("inf")


def _physical(p: dict) -> dict:
    """Clamp a parameter set to the physically meaningful region."""
    q = dict(p)
    q["R"] = max(q["R"], 1.0)
    q["Le"] = max(q["Le"], 0.0)
    q["Li"] = min(max(q["Li"], 0.0), q["Le"])
    return q


def _violation(p: dict, bounds: dict) -> float:
    v = 0.0
    for k, (lo, hi) in bounds.items():
        if p[k] < lo:
            v += lo - p[k]
        elif p[k] > hi:
            v += p[k] - hi
    v += max(p["Li"] - p["Le"], 0.0)
    return v


def fit_tank(win: TankWindow, init: dict, free, bounds: dict, s_fixed=None,
             grid_step: float = 3.0, steps: dict | None = None,
             maxiter: int = 400, uncertainty: bool = True) -> FitResult:
    """Fit the free geometric parameters of one tank.

    init holds a value for every name in PARAMS; `free` lists the ones to fit.
    Brightness levels are solved by linear least squares at every evaluation,
    so only geometry is searched.

    A coarse grid over (Li, Le) runs first to avoid the local minima that the
    pixel-scale objective has, then Nelder-Mead polishes all free parameters.
    Nelder-Mead gets an explicit initial simplex: scipy's default perturbs a
    parameter that starts at 0 by only 0.00025, so Li = 0 would never move.
    """
    free = tuple(free)
    steps = {"dx": 3.0, "dy": 3.0, "R": 2.0, "Li": 4.0, "Le": 4.0, **(steps or {})}
    base = dict(init)

    def sse_at(p):
        q = _physical(p)
        return win.solve(q["dx"], q["dy"], q["R"], q["Li"], q["Le"], s_fixed)[0]

    # Penalise leaving the box instead of hard clipping, so the simplex is
    # pushed back rather than collapsing onto a wall. The penalty scale is the
    # sum of squares of the data, which dwarfs any achievable SSE.
    scale = float((win.obs[win.mask] ** 2).sum()) + 1.0

    def objective(x):
        p = dict(base)
        p.update(zip(free, x))
        v = _violation(p, bounds)
        return sse_at(p) + (scale * (1.0 + v) if v > 0 else 0.0)

    # Coarse grid over the shadow lengths.
    if grid_step and ("Li" in free or "Le" in free):
        le_lo, le_hi = bounds.get("Le", (base["Le"], base["Le"]))
        les = np.arange(le_lo, le_hi + 1e-9, grid_step) if "Le" in free else [base["Le"]]
        best = (np.inf, base["Li"], base["Le"])
        for le in les:
            lis = np.arange(0.0, le + 1e-9, grid_step) if "Li" in free else [base["Li"]]
            for li in lis:
                p = dict(base, Li=float(li), Le=float(le))
                if _violation(p, bounds) > 0:
                    continue
                e = sse_at(p)
                if e < best[0]:
                    best = (e, float(li), float(le))
        base["Li"], base["Le"] = best[1], best[2]

    x0 = np.array([base[k] for k in free], dtype=float)
    if len(free) == 1:
        # 1-D: a bracketed golden-section search is more reliable than a simplex.
        from scipy.optimize import minimize_scalar
        lo, hi = bounds[free[0]]
        if free[0] == "Li":
            hi = min(hi, base["Le"])
        a, b = max(lo, x0[0] - grid_step), min(hi, x0[0] + grid_step)
        if b - a < 1e-6:
            xbest = x0
        else:
            r = minimize_scalar(lambda t: objective([t]), bounds=(a, b),
                                method="bounded", options={"xatol": 0.02})
            xbest = np.array([r.x]) if r.fun <= objective(x0) else x0
    else:
        simplex = [x0]
        for i, k in enumerate(free):
            xi = x0.copy()
            lo, hi = bounds.get(k, (-np.inf, np.inf))
            # Step inward if the start sits on an upper bound.
            xi[i] = xi[i] + steps[k] if xi[i] + steps[k] <= hi else xi[i] - steps[k]
            simplex.append(xi)
        r = minimize(objective, x0, method="Nelder-Mead",
                     options={"initial_simplex": np.array(simplex), "xatol": 0.02,
                              "fatol": 1e-12, "maxiter": maxiter * len(free)})
        xbest = r.x

    p = dict(base)
    p.update(zip(free, map(float, xbest)))
    p = _physical(p)
    sse, levels, model, n = win.solve(p["dx"], p["dy"], p["R"], p["Li"], p["Le"], s_fixed)
    res = FitResult(params=p, sse=sse, levels=levels, model=model, npix=n,
                    nfree_levels=3 if s_fixed is None else 2, free=free)
    if uncertainty:
        _add_uncertainty(res, win, s_fixed)
    return res


def _add_uncertainty(res: FitResult, win: TankWindow, s_fixed, h: float = 1.0):
    """Gaussian standard errors from the curvature of the SSE surface.

    SSE ~ SSE_min + 1/2 dx^T H dx, and with noise variance
    sigma^2 = SSE_min / (n - p) the covariance is 2 sigma^2 H^-1.
    The residuals include model error (ground texture, roof detail), so this
    is an honest-ish error bar, not a lower bound from pure sensor noise.
    """
    free = res.free
    k = len(free)
    dof = res.npix - k - res.nfree_levels
    if k == 0 or dof <= 0:
        return
    p0 = res.params

    def f(delta):
        p = dict(p0)
        for name, dv in zip(free, delta):
            p[name] += dv
        p["Li"] = abs(p["Li"])
        p["Le"] = max(p["Le"], p["Li"])
        return win.solve(p["dx"], p["dy"], p["R"], p["Li"], p["Le"], s_fixed)[0]

    f0 = res.sse
    H = np.zeros((k, k))
    e = np.eye(k) * h
    for i in range(k):
        H[i, i] = (f(e[i]) - 2 * f0 + f(-e[i])) / h ** 2
    for i, j in itertools.combinations(range(k), 2):
        H[i, j] = H[j, i] = (f(e[i] + e[j]) - f(e[i] - e[j]) - f(-e[i] + e[j])
                             + f(-e[i] - e[j])) / (4 * h ** 2)
    sigma2 = res.sse / dof
    try:
        cov = 2.0 * sigma2 * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return
    diag = np.diag(cov)
    if np.any(diag <= 0):
        # Not at a well-defined minimum (flat or saddle): no error bar.
        res.sigma = {n: float("nan") for n in free}
        return
    res.cov = cov
    res.sigma = {n: float(np.sqrt(v)) for n, v in zip(free, diag)}
