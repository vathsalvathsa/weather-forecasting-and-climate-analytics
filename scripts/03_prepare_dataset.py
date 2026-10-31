"""Step 3.5: build the clean, split-labelled dataset from data/raw.

Output: data/processed/daily_all_cities.parquet and reports/data_card.json
Stops with a clear error if any city's history is not clean.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.data.prepare import build_dataset, data_card


def main():
    meta = pd.read_csv(config.CITY_META_FILE)
    frames = {}
    for name in meta["name"]:
        path = config.DATA_RAW / f"{config.slug(name)}.parquet"
        if path.exists():
            frames[name] = pd.read_parquet(path)
        else:
            print(f"skip {name}: no raw file")
    if not frames:
        sys.exit("No raw files found. Run 01_download_data.py first.")

    df = build_dataset(frames, meta)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    config.REPORTS.mkdir(exist_ok=True)
    out_path = config.DATA_PROCESSED / "daily_all_cities.parquet"
    df.to_parquet(out_path, index=False)

    card = data_card(df)
    (config.REPORTS / "data_card.json").write_text(json.dumps(card, indent=2))

    pd.set_option("display.width", 160)
    print(f"{len(df):,} rows, {df['city'].nunique()} cities, {df.shape[1]} columns")
    print("\nRows per split:")
    print(df.groupby("split").agg(rows=("date", "size"), first=("date", "min"), last=("date", "max")).to_string())
    print("\nSaved ->", out_path)
    print("Saved ->", config.REPORTS / "data_card.json")


if __name__ == "__main__":
    main()
