"""Step 5.0: export the small files the Streamlit app needs into app_data/.

The deployed app cannot see data/raw or data/processed (they are not in git), so everything
it shows must come from app_data/ (committed, a few MB) or from live API calls.

    python scripts/14_export_app_data.py

Climate definitions (simple and city-relative, NOT official warning criteria):
  hot day      daily max above the city's own 95th percentile (1991-2020)
  heatwave     3 or more consecutive hot days
  heavy-rain day  wet-day rain at or above the city's own 99th percentile of wet days (1991-2020)
Annual statistics use complete years 1991-2025 only; monthly ones stop at 2026-09.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.climate.trends import annual_trend, count_runs
from src.features.build import ANOM

OUT = config.ROOT / "app_data"
BASE = (1991, 2020)                      # WMO standard normal period
KEEP = (["city", "country", "region", "lat", "lon", "elevation_m", "timezone", "date", "split"]
        + list(ANOM.values()) + ["precipitation_sum"])


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_parquet(config.DATA_PROCESSED / "daily_all_cities.parquet")
    df["year"], df["month"] = df["date"].dt.year, df["date"].dt.month

    # 1) cities and the recent daily window used by the backtest explorer
    df[["city", "country", "region", "lat", "lon", "elevation_m"]].drop_duplicates().to_csv(
        OUT / "cities.csv", index=False)
    df[df["date"] >= "2020-11-01"][KEEP].to_parquet(OUT / "daily_recent.parquet", index=False)

    # 2) annual and monthly climate summaries
    full = df[df["year"].between(1991, 2025)]
    annual = full.groupby(["city", "year"]).agg(
        tmean=("temperature_2m_mean", "mean"), tmax=("temperature_2m_max", "mean"),
        tmin=("temperature_2m_min", "mean"), precip_mm=("precipitation_sum", "sum"),
        wet_days=("precipitation_sum", lambda s: int((s >= 1).sum()))).reset_index()
    annual.to_csv(OUT / "climate_annual.csv", index=False)

    monthly = df[df["date"] < "2026-10-01"].groupby(["city", "year", "month"]).agg(
        tmean=("temperature_2m_mean", "mean"), precip_mm=("precipitation_sum", "sum")).reset_index()
    base = monthly[monthly["year"].between(*BASE)].groupby(["city", "month"])["tmean"].mean().rename("tmean_normal")
    monthly = monthly.join(base, on=["city", "month"])
    monthly["tmean_anomaly"] = monthly["tmean"] - monthly["tmean_normal"]
    monthly["date"] = pd.to_datetime(dict(year=monthly["year"], month=monthly["month"], day=1))
    monthly.to_csv(OUT / "climate_monthly.csv", index=False)

    # 3) trends: Kendall tau and Theil-Sen slope per decade
    rows = []
    for city, g in annual.groupby("city"):
        for var, label in [("tmean", "annual mean temperature (°C)"), ("precip_mm", "annual rain (mm)"),
                           ("wet_days", "wet days per year")]:
            rows.append({"city": city, "variable": label, "first_year": int(g["year"].min()),
                         "last_year": int(g["year"].max()), **annual_trend(g["year"], g[var])})
    pd.DataFrame(rows).to_csv(OUT / "climate_trends.csv", index=False)

    # 4) extremes per year, relative to each city's own 1991-2020 climate
    basedf = df[df["year"].between(*BASE)]
    hot_thr = basedf.groupby("city")["temperature_2m_max"].quantile(0.95)
    rain_thr = basedf[basedf["precipitation_sum"] >= 1].groupby("city")["precipitation_sum"].quantile(0.99)
    ext = []
    for (city, year), g in full.sort_values("date").groupby(["city", "year"]):
        hot = (g["temperature_2m_max"] > hot_thr[city]).to_numpy()
        ext.append({"city": city, "year": year, "hot_days": int(hot.sum()), "heatwaves": count_runs(hot, 3),
                    "heavy_rain_days": int((g["precipitation_sum"] >= rain_thr[city]).sum()),
                    "hot_threshold_c": round(float(hot_thr[city]), 1),
                    "heavy_rain_threshold_mm": round(float(rain_thr[city]), 1)})
    pd.DataFrame(ext).to_csv(OUT / "extremes_annual.csv", index=False)

    print("Exported to", OUT)
    for p in sorted(OUT.glob("*")):
        print(f"  {p.name:28s} {p.stat().st_size / 1e6:6.2f} MB")
    print("\n=== Temperature trends (Theil-Sen slope per decade; p from Kendall test) ===")
    t = pd.DataFrame(rows)
    t = t[t["variable"].str.contains("temperature")][["city", "sen_slope_per_decade", "sen_low_per_decade",
                                                      "sen_high_per_decade", "p_value"]]
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
