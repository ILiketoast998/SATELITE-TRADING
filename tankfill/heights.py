"""Lock each tank's height to the median of several winter measurements."""
from __future__ import annotations

import numpy as np
import pandas as pd

# A tank whose interior shadow is (near) zero in every scene is either a
# fixed-roof tank or one that stayed full all season; we cannot tell which.
FIXED_ROOF_FILL = 0.92


def lock_heights(tanks: pd.DataFrame, results: list[pd.DataFrame], min_scenes: int = 2,
                 h_min: float = 5.0, h_max: float = 25.0, fixed_min_scenes: int = 3) -> pd.DataFrame:
    """Return a copy of `tanks` with height_m set to the median of unlocked winter fits.

    Only rows flagged ok or near_full are used: the exterior shadow (and so the
    height) is still well measured when the roof is near the top.
    Robust spread: height_sigma_m = 1.4826 * MAD / sqrt(n), the standard error
    of the median-ish estimate, with a 0.3 m floor.
    """
    allres = pd.concat(results, ignore_index=True)
    good = allres[allres.flag.isin(["ok", "near_full"])]
    if "height_locked" in good:
        good = good[~good.height_locked.astype(str).str.lower().isin(["true", "1"])]
    out = tanks.copy()
    for col, default in (("height_sigma_m", np.nan), ("height_n", 0), ("height_locked", False),
                         ("height_source", ""), ("roof_type", "unknown")):
        if col not in out:
            out[col] = default
    out["height_locked"] = out["height_locked"].astype(object)
    for idx, t in out.iterrows():
        g = good[good.id == t["id"]]
        n = len(g)
        out.at[idx, "height_n"] = n
        if n >= min_scenes:
            h = float(np.median(g.height_m))
            mad = float(np.median(np.abs(g.height_m - h)))
            sigma = max(1.4826 * mad / np.sqrt(n), 0.3)
            if h_min <= h <= h_max:
                out.at[idx, "height_m"] = round(h, 2)
                out.at[idx, "height_sigma_m"] = round(sigma, 2)
                out.at[idx, "height_locked"] = True
                out.at[idx, "height_source"] = f"median_of_{n}_winter_fits"
            else:
                out.at[idx, "height_locked"] = False
                out.at[idx, "height_source"] = "implausible_median"
        allt = allres[(allres.id == t["id"]) & allres.flag.isin(["ok", "near_full"])]
        if len(allt) >= fixed_min_scenes:
            out.at[idx, "roof_type"] = ("fixed_suspect" if (allt.fill > FIXED_ROOF_FILL).all()
                                        else "floating")
    return out
