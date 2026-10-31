"""Thin client for the Open-Meteo Historical (ERA5) API.

Data licence: CC BY 4.0, attribution required ("Weather data by Open-Meteo.com").
"""
import time

import pandas as pd
import requests

from src import config


class RateLimitError(Exception):
    """Raised when the API's rate limit (e.g. daily quota) is hit."""


def _get(url: str, params: dict, retries: int = 3, backoff: int = 5) -> dict:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=90)
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(backoff * attempt)
            continue

        if resp.status_code == 200:
            return resp.json()

        try:
            reason = resp.json().get("reason", resp.text[:200])
        except ValueError:
            reason = resp.text[:200]

        if resp.status_code == 429:
            # First 429 may be the per-minute limit: wait once, then give up.
            if attempt == 1:
                time.sleep(65)
                continue
            raise RateLimitError(reason)
        if resp.status_code >= 500 and attempt < retries:
            time.sleep(backoff * attempt)
            continue
        raise RuntimeError(f"HTTP {resp.status_code}: {reason}")
    raise RuntimeError("Request failed after retries")


def fetch_daily(lat: float, lon: float, start: str, end: str, variables: list):
    """Return (DataFrame of daily values with a 'date' column, metadata dict).

    timezone='auto' makes each 'day' a LOCAL calendar day for that location.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(variables),
        "timezone": "auto",
    }
    data = _get(config.ARCHIVE_URL, params)
    df = pd.DataFrame(data["daily"]).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    meta = {
        "grid_latitude": data.get("latitude"),
        "grid_longitude": data.get("longitude"),
        "elevation_m": data.get("elevation"),
        "timezone": data.get("timezone"),
    }
    return df, meta
