"""Small, tested helpers for the climate-analytics outputs."""
import numpy as np
from scipy import stats


def annual_trend(years, values) -> dict:
    """Mann-Kendall-type trend test (Kendall tau) and Theil-Sen slope, per decade.

    Plain tests: they ignore year-to-year autocorrelation, so p-values are optimistic.
    """
    years = np.asarray(years, float)
    values = np.asarray(values, float)
    ok = ~np.isnan(values)
    years, values = years[ok], values[ok]
    if len(values) < 8:
        raise ValueError("need at least 8 years of data")
    tau, p = stats.kendalltau(years, values)
    slope, _, low, high = stats.theilslopes(values, years)
    return {"n_years": int(len(values)), "kendall_tau": float(tau), "p_value": float(p),
            "sen_slope_per_decade": float(slope * 10), "sen_low_per_decade": float(low * 10),
            "sen_high_per_decade": float(high * 10)}


def count_runs(flags, min_len: int = 3) -> int:
    """Number of runs of consecutive True values that last at least min_len days."""
    count = run = 0
    for f in np.asarray(flags, bool):
        if f:
            run += 1
        else:
            if run >= min_len:
                count += 1
            run = 0
    if run >= min_len:
        count += 1
    return count
