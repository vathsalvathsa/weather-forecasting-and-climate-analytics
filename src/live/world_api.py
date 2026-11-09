"""One request for the current weather of many cities (used by the World Map page)."""
import pandas as pd

from src.live.weather_api import FORECAST_URL, WeatherAPIError, _get_json, wmo_description

CURRENT = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code"


def get_world_snapshot(cities: list) -> pd.DataFrame:
    """cities: dicts with name, country, lat, lon. Returns one row per city with current conditions."""
    data = _get_json(FORECAST_URL, {
        "latitude": ",".join(str(c["lat"]) for c in cities),
        "longitude": ",".join(str(c["lon"]) for c in cities),
        "current": CURRENT, "timezone": "auto"}, timeout=40)
    items = data if isinstance(data, list) else [data]          # one place: a dict; several: a list
    if len(items) != len(cities):
        raise WeatherAPIError("The weather service returned an incomplete world snapshot.")
    rows = []
    for c, item in zip(cities, items):
        cur = item.get("current") or {}
        text, icon = wmo_description(cur.get("weather_code"))
        rows.append({"city": c["name"], "country": c.get("country", ""), "lat": c["lat"], "lon": c["lon"],
                     "temp": cur.get("temperature_2m"), "humidity": cur.get("relative_humidity_2m"),
                     "precip": cur.get("precipitation"), "wind": cur.get("wind_speed_10m"),
                     "text": text, "icon": icon})
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["temp"]).reset_index(drop=True)
    if df.empty:
        raise WeatherAPIError("The weather service returned no current conditions.")
    return df
