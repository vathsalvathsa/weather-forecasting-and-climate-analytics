"""Step 3.2: download daily ERA5 history for the sampled cities (resumable).

Examples:
    python scripts/01_download_data.py --pilot
    python scripts/01_download_data.py --cities "Delhi" "Tokyo"
    python scripts/01_download_data.py            # all cities in config.CITIES

Already-downloaded cities are skipped (use --force to redo). If the API quota is
hit, the script stops cleanly: just re-run it later and it continues.
"""
import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.ingest.open_meteo import RateLimitError, fetch_daily


def load_variables():
    if config.ACCEPTED_VARS_FILE.exists():
        return json.loads(config.ACCEPTED_VARS_FILE.read_text())
    print("WARNING: run 00_probe_api.py first; using core variables only.")
    return list(config.CORE_DAILY)


def update_meta(row: dict):
    if config.CITY_META_FILE.exists():
        meta = pd.read_csv(config.CITY_META_FILE)
        meta = meta[meta["name"] != row["name"]]
        meta = pd.concat([meta, pd.DataFrame([row])], ignore_index=True)
    else:
        meta = pd.DataFrame([row])
    meta.to_csv(config.CITY_META_FILE, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--cities", nargs="*")
    ap.add_argument("--start", default=config.START_DATE)
    ap.add_argument("--end", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--wait", action="store_true",
                    help="on an HOURLY rate limit, sleep ~61 min and continue automatically")
    args = ap.parse_args()

    end = args.end or (date.today() - timedelta(days=config.ARCHIVE_LAG_DAYS)).isoformat()
    variables = load_variables()

    if args.cities:
        wanted = {c.lower() for c in args.cities}
        selected = [c for c in config.CITIES if c["name"].lower() in wanted]
    elif args.pilot:
        selected = [c for c in config.CITIES if c["name"] in config.PILOT]
    else:
        selected = config.CITIES

    days = (pd.Timestamp(end) - pd.Timestamp(args.start)).days + 1
    est_units = len(selected) * max(len(variables) / 10, 1) * days / 14
    print(f"{len(selected)} cities, {len(variables)} variables, {days} days each.")
    print(f"Rough API cost estimate: ~{est_units:,.0f} call-units "
          "(free tier is ~10,000/day; the script resumes if it is hit).\n")

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    for i, city in enumerate(selected, 1):
        path = config.DATA_RAW / f"{config.slug(city['name'])}.parquet"
        if path.exists() and not args.force:
            print(f"[{i}/{len(selected)}] skip {city['name']} (already downloaded)")
            continue
        status = "ok"
        while True:
            try:
                df, meta = fetch_daily(city["lat"], city["lon"], args.start, end, variables)
                break
            except RateLimitError as exc:
                if args.wait and "hourly" in str(exc).lower():
                    print(f"Hourly limit reached while fetching {city['name']}. "
                          "Waiting 61 minutes, then retrying (Ctrl+C to stop)...")
                    time.sleep(61 * 60)
                    continue
                print(f"\nRate limit reached ({exc}). Re-run later; finished cities are kept.")
                status = "stop"
                break
            except Exception as exc:
                print(f"[{i}/{len(selected)}] FAILED {city['name']}: {exc}")
                status = "skip"
                break
        if status == "stop":
            break
        if status == "skip":
            continue
        df.to_parquet(path, index=False)
        update_meta({**city, **meta, "rows": len(df), "start": args.start, "end": end})
        print(f"[{i}/{len(selected)}] saved {city['name']}: {len(df)} rows "
              f"(elevation {meta['elevation_m']} m)")
        time.sleep(1)


if __name__ == "__main__":
    main()
