"""Turn raw per-city ERA5 files into ONE clean, split-labelled dataset.

Design rules (defend these in the viva):
- Historical ERA5 should be complete. A gap, duplicate, NaN or physically
  impossible value is a HARD ERROR: stop and investigate, never silently repair.
- Extreme-but-possible days (heatwaves, cold snaps, floods) are KEPT: they are the
  phenomena this project studies.
- Chronological split only (no shuffling). Seven-day gaps separate the parts, so a
  target up to 7 days ahead of any training row can never fall inside validation
  or test. Gap rows stay in the table as history but are not used as samples.
"""
import pandas as pd

from src import config

TRAIN_END = pd.Timestamp("2017-12-24")
VAL_START = pd.Timestamp("2018-01-01")    # 7-day gap: 2017-12-25 .. 2017-12-31
VAL_END = pd.Timestamp("2020-12-24")
TEST_START = pd.Timestamp("2021-01-01")   # 7-day gap: 2020-12-25 .. 2020-12-31

# Physically plausible daily ranges (units as returned by the API).
RANGES = {
    "temperature_2m_max": (-90, 60), "temperature_2m_min": (-90, 60),
    "temperature_2m_mean": (-90, 60), "precipitation_sum": (0, 1000),
    "rain_sum": (0, 1000), "wind_speed_10m_max": (0, 250),
    "relative_humidity_2m_mean": (0, 100), "cloud_cover_mean": (0, 100),
    "surface_pressure_mean": (300, 1100), "dew_point_2m_mean": (-90, 40),
}


def assign_split(dates: pd.Series) -> pd.Series:
    """Label each date train / val / test, or 'gap' for the separating days."""
    out = pd.Series("gap", index=dates.index, dtype="object")
    out[dates <= TRAIN_END] = "train"
    out[(dates >= VAL_START) & (dates <= VAL_END)] = "val"
    out[dates >= TEST_START] = "test"
    return out


def check_city_frame(df: pd.DataFrame, city: str) -> None:
    """Raise ValueError if the historical series for one city is not clean."""
    missing_cols = [c for c in config.CORE_DAILY + ["date"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{city}: missing columns {missing_cols}")

    problems = []
    if df["date"].duplicated().any():
        problems.append(f"{int(df['date'].duplicated().sum())} duplicate dates")
    span = (df["date"].max() - df["date"].min()).days + 1
    n_missing = span - df["date"].nunique()
    if n_missing:
        problems.append(f"{n_missing} missing dates")

    value_cols = [c for c in df.columns if c != "date"]
    na = df[value_cols].isna().sum()
    na = na[na > 0]
    if len(na):
        problems.append(f"NaN values {na.to_dict()}")

    for col, (lo, hi) in RANGES.items():
        if col in df.columns and ((df[col] < lo) | (df[col] > hi)).any():
            problems.append(f"{col} outside [{lo}, {hi}]")
    if (df["temperature_2m_max"] < df["temperature_2m_min"] - 0.01).any():
        problems.append("temperature_2m_max < temperature_2m_min")

    if problems:
        raise ValueError(f"{city}: " + "; ".join(problems))


def build_dataset(frames: dict, meta: pd.DataFrame) -> pd.DataFrame:
    """frames: {city name: raw daily DataFrame}; meta: city metadata table."""
    meta_idx = meta.set_index("name")
    parts = []
    for city, raw in frames.items():
        check_city_frame(raw, city)
        d = raw.sort_values("date").reset_index(drop=True).copy()
        m = meta_idx.loc[city]
        # Use the GRID-CELL coordinates the data actually describes.
        d.insert(0, "timezone", m["timezone"])
        d.insert(0, "elevation_m", m["elevation_m"])
        d.insert(0, "lon", m["grid_longitude"])
        d.insert(0, "lat", m["grid_latitude"])
        d.insert(0, "region", m["region"])
        d.insert(0, "country", m["country"])
        d.insert(0, "city", city)
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out["split"] = assign_split(out["date"])
    return out.sort_values(["city", "date"]).reset_index(drop=True)


def data_card(df: pd.DataFrame) -> dict:
    """Small machine-readable summary to cite in the report/README."""
    splits = {}
    for name, g in df.groupby("split"):
        splits[name] = {"rows": int(len(g)), "from": str(g["date"].min().date()),
                        "to": str(g["date"].max().date())}
    return {
        "cities": sorted(df["city"].unique().tolist()),
        "n_rows": int(len(df)),
        "columns": df.columns.tolist(),
        "splits": splits,
        "source": "Open-Meteo Historical Weather API (ERA5 reanalysis), CC BY 4.0",
        "note": "ERA5 is a reanalysis (model product), not station observations; "
                "weather_code is model-derived.",
    }
