"""Reliability (calibration) tools for probability forecasts."""
import numpy as np
import pandas as pd


def reliability_table(y, p, bins: int = 10) -> pd.DataFrame:
    """For each probability bin: how many forecasts, mean forecast, observed frequency."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(mask.sum()),
                     "mean_pred": float(p[mask].mean()), "obs_freq": float(y[mask].mean())})
    return pd.DataFrame(rows)


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Weighted average gap between forecast probability and observed frequency."""
    t = reliability_table(y, p, bins)
    w = t["n"] / t["n"].sum()
    return float((w * (t["mean_pred"] - t["obs_freq"]).abs()).sum())
