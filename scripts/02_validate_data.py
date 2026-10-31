"""Step 3.3: data-quality audit. REPORTS problems; never silently modifies data.

Output: reports/data_quality_summary.csv (one row per city) + printed summary.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config

# Physically plausible daily ranges (units as returned by the API).
RANGES = {
    "temperature_2m_max": (-90, 60), "temperature_2m_min": (-90, 60),
    "temperature_2m_mean": (-90, 60), "precipitation_sum": (0, 1000),
    "rain_sum": (0, 1000), "wind_speed_10m_max": (0, 250),
    "relative_humidity_2m_mean": (0, 100), "cloud_cover_mean": (0, 100),
    "surface_pressure_mean": (300, 1100), "dew_point_2m_mean": (-90, 40),
}


def analyse_city(df: pd.DataFrame, name: str) -> dict:
    df = df.sort_values("date")
    out = {"city": name, "rows": len(df)}
    out["start"], out["end"] = df["date"].min().date(), df["date"].max().date()
    expected = (df["date"].max() - df["date"].min()).days + 1
    out["missing_dates"] = int(expected - df["date"].nunique())
    out["duplicate_dates"] = int(df["date"].duplicated().sum())

    for col in df.columns.drop("date"):
        out[f"na_{col}"] = int(df[col].isna().sum())

    out["range_violations"] = int(sum(
        ((df[c] < lo) | (df[c] > hi)).sum()
        for c, (lo, hi) in RANGES.items() if c in df.columns))

    # Logical consistency
    tol = 0.01
    out["tmax_lt_tmin"] = int((df["temperature_2m_max"] < df["temperature_2m_min"]).sum())
    out["tmean_outside_minmax"] = int(
        ((df["temperature_2m_mean"] < df["temperature_2m_min"] - tol) |
         (df["temperature_2m_mean"] > df["temperature_2m_max"] + tol)).sum())
    if {"rain_sum", "precipitation_sum"} <= set(df.columns):
        out["rain_gt_precip"] = int((df["rain_sum"] > df["precipitation_sum"] + tol).sum())

    # Outliers vs the city's own seasonal climatology. Uses median and MAD per
    # day-of-year (robust): a mean/std z-score would let the outlier inflate its
    # own spread and hide itself.
    doy = df["date"].dt.day_of_year
    g = df.groupby(doy)["temperature_2m_mean"]
    med = g.transform("median")
    mad = (df["temperature_2m_mean"] - med).abs().groupby(doy).transform("median")
    robust_z = 0.6745 * (df["temperature_2m_mean"] - med) / mad.where(mad > 0)
    out["temp_outliers_robust_z_gt_5"] = int((robust_z.abs() > 5).sum())

    # Suspicious "stuck" behaviour: 5 identical consecutive daily means
    same = (df["temperature_2m_mean"].diff() == 0).astype(int)
    out["stuck_runs_5d"] = int((same.rolling(4).sum() == 4).sum())
    return out


def main():
    files = sorted(config.DATA_RAW.glob("*.parquet"))
    if not files:
        sys.exit("No parquet files in data/raw. Run 01_download_data.py first.")
    rows = [analyse_city(pd.read_parquet(f), f.stem) for f in files]
    summary = pd.DataFrame(rows).fillna(0)
    config.REPORTS.mkdir(exist_ok=True)
    out_path = config.REPORTS / "data_quality_summary.csv"
    summary.to_csv(out_path, index=False)
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print(summary.to_string(index=False))
    print("\nSaved ->", out_path)


if __name__ == "__main__":
    main()
