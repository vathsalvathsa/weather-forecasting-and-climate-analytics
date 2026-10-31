"""Step 3.1: check connectivity and which daily variables the API accepts.

Run once. Cheap (one year, one city). Saves data/raw/_accepted_variables.json.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.ingest.open_meteo import fetch_daily

config.DATA_RAW.mkdir(parents=True, exist_ok=True)
TEST = dict(lat=28.61, lon=77.21, start="2023-01-01", end="2023-12-31")

accepted, rejected = [], {}
for var in config.CORE_DAILY + config.OPTIONAL_DAILY:
    try:
        df, _ = fetch_daily(TEST["lat"], TEST["lon"], TEST["start"], TEST["end"], [var])
        non_null = int(df[var].notna().sum())
        if non_null == 0:
            rejected[var] = "accepted but all values empty"
            print(f"EMPTY    {var}")
        else:
            accepted.append(var)
            print(f"OK       {var:32s} non-null days: {non_null}")
    except Exception as exc:  # report, do not hide
        rejected[var] = str(exc)
        print(f"REJECTED {var:32s} {exc}")
    time.sleep(0.4)

config.ACCEPTED_VARS_FILE.write_text(json.dumps(accepted, indent=2))
print("\nAccepted:", len(accepted), "| Rejected:", len(rejected))
print("Saved ->", config.ACCEPTED_VARS_FILE)
