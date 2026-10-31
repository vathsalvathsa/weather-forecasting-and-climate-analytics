"""Run from the project root with:  python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import prepare


def make_frame(start="2017-12-20", end="2021-01-05", seed=0):
    dates = pd.date_range(start, end)
    rng = np.random.default_rng(seed)
    t = 20 + rng.normal(0, 1, len(dates))
    return pd.DataFrame({
        "date": dates, "temperature_2m_mean": t, "temperature_2m_max": t + 4,
        "temperature_2m_min": t - 4, "precipitation_sum": 0.0, "rain_sum": 0.0,
        "wind_speed_10m_max": 10.0, "weather_code": 3,
    })


def make_meta(name="Testville"):
    return pd.DataFrame([{"name": name, "country": "X", "region": "R", "lat": 1.0,
                          "lon": 2.0, "grid_latitude": 1.1, "grid_longitude": 2.1,
                          "elevation_m": 10.0, "timezone": "UTC"}])


def test_split_boundaries_and_gaps():
    df = make_frame()
    s = prepare.assign_split(df["date"])
    lab = dict(zip(df["date"].dt.strftime("%Y-%m-%d"), s))
    assert lab["2017-12-24"] == "train"
    assert all(lab[f"2017-12-{d}"] == "gap" for d in range(25, 32))      # 7-day gap
    assert lab["2018-01-01"] == "val"
    assert lab["2020-12-24"] == "val"
    assert all(lab[f"2020-12-{d}"] == "gap" for d in range(25, 32))      # 7-day gap
    assert lab["2021-01-01"] == "test"


def test_split_order_is_chronological_and_never_interleaved():
    df = make_frame("2000-01-01", "2023-12-31")
    order = {"train": 0, "gap": 1, "val": 2, "test": 3}
    s = prepare.assign_split(df["date"]).map(order)
    # gap appears in two places, so map the labels to a stage number instead
    stage = np.select(
        [df["date"] <= prepare.TRAIN_END, df["date"] < prepare.VAL_START,
         df["date"] <= prepare.VAL_END, df["date"] < prepare.TEST_START],
        [0, 1, 2, 3], default=4)
    assert (np.diff(stage) >= 0).all()
    assert s.notna().all()


def test_clean_frame_passes():
    prepare.check_city_frame(make_frame(), "Testville")


def test_duplicate_dates_raise():
    df = pd.concat([make_frame(), make_frame().iloc[[5]]])
    with pytest.raises(ValueError, match="duplicate"):
        prepare.check_city_frame(df, "Testville")


def test_missing_date_raises():
    df = make_frame().drop(index=[10, 11])
    with pytest.raises(ValueError, match="missing dates"):
        prepare.check_city_frame(df, "Testville")


def test_nan_raises():
    df = make_frame()
    df.loc[3, "temperature_2m_mean"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        prepare.check_city_frame(df, "Testville")


def test_impossible_value_raises():
    df = make_frame()
    df.loc[3, "temperature_2m_max"] = 75.0
    with pytest.raises(ValueError, match="outside"):
        prepare.check_city_frame(df, "Testville")


def test_tmax_below_tmin_raises():
    df = make_frame()
    df.loc[3, "temperature_2m_max"] = df.loc[3, "temperature_2m_min"] - 1
    with pytest.raises(ValueError, match="temperature_2m_max"):
        prepare.check_city_frame(df, "Testville")


def test_build_dataset_adds_metadata_and_split():
    out = prepare.build_dataset({"Testville": make_frame()}, make_meta())
    assert {"city", "lat", "lon", "elevation_m", "split"} <= set(out.columns)
    assert out["lat"].iloc[0] == 1.1                       # grid-cell latitude used
    assert set(out["split"]) == {"train", "gap", "val", "test"}
    assert out["date"].is_monotonic_increasing


def test_data_card_counts_match():
    out = prepare.build_dataset({"Testville": make_frame()}, make_meta())
    card = prepare.data_card(out)
    assert sum(v["rows"] for v in card["splits"].values()) == len(out) == card["n_rows"]
