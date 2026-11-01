"""Feature engineering for next-day(s) forecasts.

LEAKAGE RULES (each one is covered by a unit test):
1. A feature for issue day t may only use data from day t or EARLIER
   (lags, trailing rolling windows; never centred windows, never future values).
2. Climatologies, which turn raw values into anomalies, are fitted on
   split == 'train' rows only.
3. The model predicts the target's ANOMALY (truth minus climatology of the target
   date), and the climatology is added back afterwards.
"""
import numpy as np
import pandas as pd

from src.models.baselines import Climatology, doy365, make_samples

# short name -> raw column. These variables are converted to anomalies.
ANOM = {
    "tmean": "temperature_2m_mean", "tmax": "temperature_2m_max",
    "tmin": "temperature_2m_min", "rh": "relative_humidity_2m_mean",
    "pres": "surface_pressure_mean", "cloud": "cloud_cover_mean",
    "dew": "dew_point_2m_mean", "wind": "wind_speed_10m_max",
    "rad": "shortwave_radiation_sum",
}
LOCATION_COLS = ["lat", "lon", "elevation_m"]


def fit_climatologies(df: pd.DataFrame) -> dict:
    """One smoothed day-of-year climatology per variable, from TRAIN rows only."""
    train = df[df["split"] == "train"]
    return {raw: Climatology().fit(train, raw) for raw in ANOM.values()}


def build_features(df: pd.DataFrame, clims: dict) -> pd.DataFrame:
    """One row per (city, date) with features known at the END of that day."""
    d = df.sort_values(["city", "date"]).reset_index(drop=True)
    f = d[["city", "date"]].copy()

    anom = pd.DataFrame({"city": d["city"]})
    for short, raw in ANOM.items():
        anom[short] = d[raw].to_numpy() - clims[raw].predict(d["city"], d["date"])
    g = anom.groupby("city")
    for short in ANOM:
        f[f"{short}_a0"] = anom[short]                                    # today
        f[f"{short}_a1"] = g[short].shift(1)                              # yesterday
        f[f"{short}_a2"] = g[short].shift(2)
        f[f"{short}_a7mean"] = g[short].transform(lambda s: s.rolling(7).mean())
    f["tmean_a30mean"] = g["tmean"].transform(lambda s: s.rolling(30).mean())

    precip = d.groupby("city")["precipitation_sum"]
    f["logp0"] = np.log1p(d["precipitation_sum"])
    f["logp1"] = np.log1p(precip.shift(1))
    for k in (3, 7, 30):
        f[f"logp_sum{k}"] = np.log1p(precip.transform(lambda s, k=k: s.rolling(k).sum()))

    pres = d.groupby("city")["surface_pressure_mean"]
    f["pres_d1"] = d["surface_pressure_mean"] - pres.shift(1)             # pressure tendency
    f["pres_d3"] = d["surface_pressure_mean"] - pres.shift(3)
    f["dtr0"] = d["temperature_2m_max"] - d["temperature_2m_min"]         # diurnal range

    doy = doy365(d["date"])
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    for c in LOCATION_COLS:
        f[c] = d[c].to_numpy()
    return f


def make_supervised(df: pd.DataFrame, feats: pd.DataFrame, clims: dict, target: str, horizon: int):
    """Join features (day t) with the target (day t + horizon).

    Returns (table, feature_columns). `y_anom` is what the model learns;
    prediction = clim_tgt + model output.
    """
    raw = ANOM[target]
    s = make_samples(df, raw, horizon)
    s["clim_tgt"] = clims[raw].predict(s["city"], s["target_date"])
    s["y_anom"] = s["y_true"] - s["clim_tgt"]
    m = s.merge(feats, on=["city", "date"], how="left", validate="one_to_one")
    cols = [c for c in feats.columns if c not in ("city", "date")] + ["clim_tgt"]
    m = m.dropna(subset=cols).reset_index(drop=True)   # drops the first ~30 days of each city
    return m, cols
