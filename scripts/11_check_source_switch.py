"""Does the default 'best_match' data change source around 2017? Compare with explicit ERA5.

Re-downloads 2014-2020 for three cities with models=era5 (a few dozen API call-units)
and compares it with the data you already saved. Read-only: nothing in data/ is changed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.ingest.open_meteo import _get

CITIES = ["Nairobi", "New York", "Delhi"]
VARS = ["temperature_2m_mean", "precipitation_sum"]
START, END = "2014-01-01", "2020-12-31"
pd.set_option("display.width", 200)

for name in CITIES:
    c = next(x for x in config.CITIES if x["name"] == name)
    params = {"latitude": c["lat"], "longitude": c["lon"], "start_date": START, "end_date": END,
              "daily": ",".join(VARS), "timezone": "auto", "models": "era5"}
    data = _get(config.ARCHIVE_URL, params)
    era = pd.DataFrame(data["daily"]).rename(columns={"time": "date"})
    era["date"] = pd.to_datetime(era["date"])
    saved = pd.read_parquet(config.DATA_RAW / f"{config.slug(name)}.parquet")[["date"] + VARS]
    m = saved.merge(era, on="date", suffixes=("_saved", "_era5"))

    rows = []
    for year, g in m.groupby(m["date"].dt.year):
        rows.append({
            "year": year,
            "wet_share_saved": (g["precipitation_sum_saved"] >= 1).mean(),
            "wet_share_era5": (g["precipitation_sum_era5"] >= 1).mean(),
            "tmean_abs_diff": (g["temperature_2m_mean_saved"] - g["temperature_2m_mean_era5"]).abs().mean(),
            "precip_abs_diff_mm": (g["precipitation_sum_saved"] - g["precipitation_sum_era5"]).abs().mean(),
        })
    print(f"\n=== {name}: saved data vs explicit ERA5 ===")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
