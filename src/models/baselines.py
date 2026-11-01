"""Reference forecasts every ML model must beat.

LEAKAGE RULE: everything that is *fitted* (climatology, damping factors) uses
rows with split == 'train' only. Validation/test values never influence a baseline.

Forecast task: at the end of day t, predict the value on day t + horizon.
"""
import numpy as np
import pandas as pd


def doy365(dates: pd.Series) -> np.ndarray:
    """Day of year on a 365-day calendar (29 Feb is merged into 28 Feb)."""
    doy = dates.dt.day_of_year.to_numpy().copy()
    leap = dates.dt.is_leap_year.to_numpy()
    return np.where(leap & (doy >= 60), doy - 1, doy)


def make_samples(df: pd.DataFrame, col: str, horizon: int) -> pd.DataFrame:
    """One row per (city, issue date t): y_now = value at t, y_true = value at t+horizon.

    Gap rows are never used as samples. The last `horizon` days of each city have
    no target and are dropped. Relies on contiguous daily data (checked upstream).
    """
    d = df.sort_values(["city", "date"]).reset_index(drop=True)
    out = d[["city", "date", "split"]].copy()
    out["target_date"] = d.groupby("city")["date"].shift(-horizon)
    out["y_now"] = d[col].to_numpy()
    out["y_true"] = d.groupby("city")[col].shift(-horizon).to_numpy()
    out = out[(out["split"] != "gap") & out["y_true"].notna()].copy()
    lag_days = (out["target_date"] - out["date"]).dt.days
    if not (lag_days == horizon).all():
        raise ValueError("Dates are not contiguous: target is not exactly `horizon` days ahead.")
    return out.reset_index(drop=True)


class Climatology:
    """Smoothed day-of-year mean per city (for a 0/1 column this is a wet-day probability)."""

    def __init__(self, window: int = 15):
        if window % 2 == 0:
            raise ValueError("window must be odd")
        self.window = window

    def fit(self, train_df: pd.DataFrame, col: str):
        d = train_df[["city", "date", col]].copy()
        d["doy"] = doy365(d["date"])
        tab = d.groupby(["doy", "city"])[col].mean().unstack("city").reindex(range(1, 366))
        half = self.window // 2
        padded = pd.concat([tab.iloc[-half:], tab, tab.iloc[:half]])  # wrap around the year
        smooth = padded.rolling(self.window, center=True).mean().iloc[half:-half]
        smooth.index = range(1, 366)
        if smooth.isna().any().any():
            raise ValueError("Training data does not cover every day of the year for every city.")
        self.table_ = smooth
        return self

    def predict(self, cities, dates) -> np.ndarray:
        idx = self.table_.columns.get_indexer(pd.Series(cities))
        if (idx < 0).any():
            raise KeyError("City not seen during climatology fit.")
        return self.table_.to_numpy()[doy365(pd.Series(dates)) - 1, idx]


def baseline_predictions(df: pd.DataFrame, col: str, horizon: int,
                         kinds=("persistence", "climatology",
                                "anomaly_persistence", "damped_anomaly")) -> pd.DataFrame:
    """Samples table with one pred_<name> column per baseline."""
    clim = Climatology().fit(df[df["split"] == "train"], col)
    s = make_samples(df, col, horizon)

    c_now = clim.predict(s["city"], s["date"])
    c_tgt = clim.predict(s["city"], s["target_date"])
    anom_now = s["y_now"].to_numpy() - c_now

    if "persistence" in kinds:
        s["pred_persistence"] = s["y_now"]
    if "climatology" in kinds:
        s["pred_climatology"] = c_tgt
    if "anomaly_persistence" in kinds:           # keep today's anomaly unchanged
        s["pred_anomaly_persistence"] = c_tgt + anom_now
    if "damped_anomaly" in kinds:                # anomaly decays by a fitted factor rho
        anom_tgt = s["y_true"].to_numpy() - c_tgt
        tr = (s["split"] == "train").to_numpy()
        a = pd.DataFrame({"city": s["city"].to_numpy()[tr], "num": (anom_now * anom_tgt)[tr],
                          "den": (anom_now ** 2)[tr]})
        g = a.groupby("city")[["num", "den"]].sum()
        rho = g["num"] / g["den"]
        s["pred_damped_anomaly"] = c_tgt + s["city"].map(rho).to_numpy() * anom_now
        s.attrs["rho"] = rho
    return s
