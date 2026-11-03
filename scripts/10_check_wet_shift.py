"""Is the validation period wetter than training? (train + val only; the test split stays locked)"""
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)
path = Path(__file__).resolve().parents[1] / "data" / "processed" / "daily_all_cities.parquet"
df = pd.read_parquet(path)
df = df[df["split"].isin(["train", "val"])].copy()
df["wet"] = df["precipitation_sum"] >= 1.0

by_city = df.pivot_table(index="city", columns="split", values="wet", aggfunc="mean")
by_city["val_minus_train"] = by_city["val"] - by_city["train"]
print("=== Share of wet days (>= 1 mm), per city ===")
print(by_city.round(3).to_string())
print("\nAll cities pooled:", df.groupby("split")["wet"].mean().round(3).to_dict())

print("\n=== Share of wet days by year, all cities pooled ===")
print(df.groupby(df["date"].dt.year)["wet"].mean().round(3).to_string())
