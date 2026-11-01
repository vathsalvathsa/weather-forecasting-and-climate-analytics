import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from src.data.prepare import assign_split
from src.models import baselines as B


def make_df(n_cities=2, start="1991-01-01", end="2023-12-31", seed=0, ar=0.8, amp=10.0, noise=1.0):
    dates = pd.date_range(start, end)
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(n_cities):
        anomaly = lfilter([1.0], [1.0, -ar], rng.normal(0, noise, len(dates)))
        doy = B.doy365(pd.Series(dates))
        x = 15 + 5 * i + amp * np.sin(2 * np.pi * doy / 365) + anomaly
        parts.append(pd.DataFrame({"city": f"C{i}", "date": dates, "x": x}))
    df = pd.concat(parts, ignore_index=True)
    df["split"] = assign_split(df["date"])
    return df


def test_doy365_merges_leap_day():
    d = pd.Series(pd.to_datetime(["2020-02-28", "2020-02-29", "2020-03-01", "2020-12-31",
                                  "2021-03-01", "2021-12-31"]))
    assert B.doy365(d).tolist() == [59, 59, 60, 365, 60, 365]


def test_make_samples_alignment_and_exclusions():
    df = make_df(n_cities=1, start="2017-12-01", end="2021-02-01")
    df["x"] = np.arange(len(df), dtype=float)                 # value == day index
    s = B.make_samples(df, "x", horizon=3)
    assert (s["y_true"] - s["y_now"] == 3).all()              # target is exactly 3 days ahead
    assert (s["split"] != "gap").all()                        # gap rows never sampled
    assert s["date"].max() == pd.Timestamp("2021-01-29")      # last 3 days have no target


def test_climatology_ignores_validation_and_test_values():
    df = make_df()
    before = B.baseline_predictions(df, "x", 1)["pred_climatology"].to_numpy()
    df2 = df.copy()
    df2.loc[df2["split"].isin(["val", "test"]), "x"] += 100.0
    after = B.baseline_predictions(df2, "x", 1)["pred_climatology"].to_numpy()
    assert np.allclose(before, after)                         # no leakage from val/test


def test_climatology_recovers_a_clean_seasonal_cycle():
    df = make_df(noise=0.0, ar=0.0)                           # pure sinusoid, no noise
    s = B.baseline_predictions(df, "x", 1, kinds=("persistence", "climatology"))
    assert np.abs(s["pred_climatology"] - s["y_true"]).max() < 0.1   # only smoothing error


def test_persistence_equals_todays_value():
    df = make_df(n_cities=1)
    s = B.baseline_predictions(df, "x", 1)
    assert np.array_equal(s["pred_persistence"].to_numpy(), s["y_now"].to_numpy())


def test_damped_factor_recovers_ar1_coefficient():
    df = make_df(n_cities=2, ar=0.8)
    rho1 = B.baseline_predictions(df, "x", 1).attrs["rho"]
    rho2 = B.baseline_predictions(df, "x", 2).attrs["rho"]
    assert np.allclose(rho1, 0.8, atol=0.05)
    assert np.allclose(rho2, 0.64, atol=0.06)                 # 0.8 ** 2


def test_unknown_city_raises():
    df = make_df(n_cities=1)
    clim = B.Climatology().fit(df[df["split"] == "train"], "x")
    with pytest.raises(KeyError):
        clim.predict(pd.Series(["Nowhere"]), pd.Series(pd.to_datetime(["2020-06-01"])))
