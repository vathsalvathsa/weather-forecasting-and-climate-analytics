import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from src.data.prepare import assign_split
from src.features import build as F
from src.models import baselines as B


def make_full_df(n_cities=3, ar=0.8, seed=0):
    dates = pd.date_range("1991-01-01", "2023-12-31")
    n = len(dates)
    rng = np.random.default_rng(seed)
    season = np.sin(2 * np.pi * B.doy365(pd.Series(dates)) / 365)
    parts = []
    for i in range(n_cities):
        anom = lfilter([1.0], [1.0, -ar], rng.normal(0, 1, n))
        t = 15 + 5 * i + 10 * season + anom
        parts.append(pd.DataFrame({
            "city": f"C{i}", "date": dates, "lat": 10.0 + 10 * i, "lon": 20.0 * i,
            "elevation_m": 100.0 * i, "temperature_2m_mean": t,
            "temperature_2m_max": t + 4, "temperature_2m_min": t - 4,
            "relative_humidity_2m_mean": 60 + 10 * season + 3 * anom + rng.normal(0, 2, n),
            "surface_pressure_mean": 1000 + rng.normal(0, 3, n),
            "cloud_cover_mean": 50 + rng.normal(0, 10, n),
            "dew_point_2m_mean": t - 5, "wind_speed_10m_max": 12 + rng.normal(0, 2, n),
            "shortwave_radiation_sum": 15 + 8 * season + rng.normal(0, 2, n),
            "precipitation_sum": np.where(rng.random(n) < 0.3, rng.gamma(0.8, 8, n), 0.0),
        }))
    df = pd.concat(parts, ignore_index=True)
    df["split"] = assign_split(df["date"])
    return df


RAW_COLS = list(F.ANOM.values()) + ["precipitation_sum"]


def test_features_never_use_future_values():
    df = make_full_df()
    clims = F.fit_climatologies(df)
    f1 = F.build_features(df, clims)
    df2 = df.copy()
    future = df2["date"] >= "2010-01-01"
    for c in RAW_COLS:
        df2.loc[future, c] += 50.0                           # corrupt everything from 2010 on
    f2 = F.build_features(df2, clims)
    past = f1["date"] < pd.Timestamp("2010-01-01")
    pd.testing.assert_frame_equal(f1[past].reset_index(drop=True), f2[past].reset_index(drop=True))


def test_climatologies_use_train_rows_only():
    df = make_full_df()
    a = F.fit_climatologies(df)
    df2 = df.copy()
    held_out = df2["split"].isin(["val", "test"])
    for c in RAW_COLS:
        df2.loc[held_out, c] += 100.0
    b = F.fit_climatologies(df2)
    for raw in F.ANOM.values():
        assert np.allclose(a[raw].table_.to_numpy(), b[raw].table_.to_numpy())


def test_supervised_table_is_consistent():
    df = make_full_df()
    clims = F.fit_climatologies(df)
    m, cols = F.make_supervised(df, F.build_features(df, clims), clims, "tmean", 3)
    assert not m[cols].isna().any().any()
    assert (m["split"] != "gap").all()
    assert np.allclose(m["y_anom"] + m["clim_tgt"], m["y_true"])
    first = m.groupby("city")["date"].min()
    assert (first >= pd.Timestamp("1991-01-01") + pd.Timedelta(days=29)).all()   # needs 30 days of history


def test_target_is_exactly_horizon_days_ahead():
    df = make_full_df(n_cities=1)
    df["temperature_2m_mean"] = np.arange(len(df), dtype=float)        # value == day index
    clims = F.fit_climatologies(df)
    m, _ = F.make_supervised(df, F.build_features(df, clims), clims, "tmean", 3)
    assert ((m["y_true"] - m["y_now"]) == 3).all()


def test_ridge_beats_climatology_on_synthetic_persistent_anomalies():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    df = make_full_df(ar=0.8)
    clims = F.fit_climatologies(df)
    m, cols = F.make_supervised(df, F.build_features(df, clims), clims, "tmean", 1)
    tr, ev = m[m["split"] == "train"], m[m["split"] == "val"]
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(tr[cols], tr["y_anom"])
    mse_model = np.mean((ev["y_anom"] - model.predict(ev[cols])) ** 2)
    mse_clim = np.mean(ev["y_anom"] ** 2)                               # climatology predicts anomaly 0
    assert mse_model < 0.7 * mse_clim
