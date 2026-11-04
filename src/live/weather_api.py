"""Thin, testable client for the live Open-Meteo APIs used by the Streamlit app.

Weather data by Open-Meteo.com (CC BY 4.0). Everything returned here is a
numerical-weather-model FORECAST or analysis from Open-Meteo; none of it comes
from our own machine-learning models, and the app labels it that way.
"""
import pandas as pd
import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

CURRENT_VARS = ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation",
                "weather_code", "cloud_cover", "pressure_msl", "wind_speed_10m", "wind_direction_10m"]
HOURLY_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation_probability", "precipitation",
               "wind_speed_10m", "cloud_cover", "visibility", "uv_index"]
DAILY_VARS = ["weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
              "precipitation_probability_max", "wind_speed_10m_max", "uv_index_max", "sunrise", "sunset"]


class WeatherAPIError(Exception):
    """A problem with the weather service, with a message that is safe to show to users."""


def _get_json(url: str, params: dict, timeout: int = 20) -> dict:
    try:
        resp = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise WeatherAPIError("Could not reach the weather service. Check your internet connection and try again.") from exc
    if resp.status_code == 429:
        raise WeatherAPIError("The weather service is rate-limiting requests right now. Please try again in a minute.")
    if resp.status_code != 200:
        try:
            reason = resp.json().get("reason", "")
        except ValueError:
            reason = ""
        raise WeatherAPIError(f"Weather service error ({resp.status_code}). {reason}".strip())
    try:
        return resp.json()
    except ValueError as exc:
        raise WeatherAPIError("The weather service returned an unreadable response.") from exc


def geocode(query: str, count: int = 5) -> list:
    """Find places by name. Returns [] when nothing matches. Labels are unique."""
    query = (query or "").strip()
    if len(query) < 2:
        return []
    data = _get_json(GEOCODING_URL, {"name": query, "count": count, "language": "en", "format": "json"})
    places = []
    for r in data.get("results") or []:
        label = ", ".join(p for p in (r.get("name"), r.get("admin1"), r.get("country")) if p)
        places.append({"label": label, "name": r.get("name"), "admin1": r.get("admin1"),
                       "country": r.get("country"), "latitude": r["latitude"],
                       "longitude": r["longitude"], "elevation": r.get("elevation"),
                       "timezone": r.get("timezone")})
    seen = {}
    for p in places:                       # same label twice (e.g. two towns): add coordinates
        seen[p["label"]] = seen.get(p["label"], 0) + 1
    for p in places:
        if seen[p["label"]] > 1:
            p["label"] = f"{p['label']} ({p['latitude']:.2f}, {p['longitude']:.2f})"
    return places


def get_forecast(lat: float, lon: float) -> dict:
    """Current conditions, 7 days of hourly values and 7 daily summaries (local time)."""
    data = _get_json(FORECAST_URL, {
        "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 7,
        "current": ",".join(CURRENT_VARS), "hourly": ",".join(HOURLY_VARS), "daily": ",".join(DAILY_VARS),
    })
    if not all(k in data for k in ("current", "hourly", "daily")):
        raise WeatherAPIError("The weather service returned an incomplete forecast.")
    hourly = pd.DataFrame(data["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    daily = pd.DataFrame(data["daily"])
    for col in ("time", "sunrise", "sunset"):
        daily[col] = pd.to_datetime(daily[col])
    return {"current": data["current"], "current_units": data.get("current_units", {}),
            "hourly": hourly, "daily": daily, "timezone": data.get("timezone"),
            "elevation": data.get("elevation")}


def get_air_quality(lat: float, lon: float):
    """Current air quality as a dict, or None when unavailable (never raises)."""
    try:
        data = _get_json(AIR_QUALITY_URL, {
            "latitude": lat, "longitude": lon, "timezone": "auto",
            "current": "us_aqi,pm10,pm2_5,nitrogen_dioxide,ozone"})
    except WeatherAPIError:
        return None
    cur = data.get("current")
    if not cur or cur.get("us_aqi") is None:
        return None
    return cur


def current_hour_values(forecast: dict) -> pd.Series:
    """The hourly row closest to the 'current' time (for UV index and visibility)."""
    hourly = forecast["hourly"]
    now = pd.to_datetime(forecast["current"]["time"])
    return hourly.loc[(hourly["time"] - now).abs().idxmin()]


_WMO = {0: ("Clear sky", "☀️"), 1: ("Mainly clear", "🌤️"), 2: ("Partly cloudy", "⛅"), 3: ("Overcast", "☁️"),
        45: ("Fog", "🌫️"), 48: ("Rime fog", "🌫️"), 51: ("Light drizzle", "🌦️"), 53: ("Drizzle", "🌦️"),
        55: ("Dense drizzle", "🌦️"), 56: ("Freezing drizzle", "🌧️"), 57: ("Dense freezing drizzle", "🌧️"),
        61: ("Light rain", "🌧️"), 63: ("Rain", "🌧️"), 65: ("Heavy rain", "🌧️"),
        66: ("Freezing rain", "🌧️"), 67: ("Heavy freezing rain", "🌧️"), 71: ("Light snow", "🌨️"),
        73: ("Snow", "🌨️"), 75: ("Heavy snow", "❄️"), 77: ("Snow grains", "🌨️"),
        80: ("Light rain showers", "🌦️"), 81: ("Rain showers", "🌦️"), 82: ("Violent rain showers", "⛈️"),
        85: ("Light snow showers", "🌨️"), 86: ("Snow showers", "🌨️"), 95: ("Thunderstorm", "⛈️"),
        96: ("Thunderstorm with hail", "⛈️"), 99: ("Thunderstorm with heavy hail", "⛈️")}


def wmo_description(code) -> tuple:
    """(text, emoji) for a WMO weather code; ('Unknown', '') if missing."""
    if code is None or pd.isna(code):
        return "Unknown", ""
    return _WMO.get(int(code), ("Unknown", ""))


def wind_compass(degrees) -> str:
    if degrees is None or pd.isna(degrees):
        return "n/a"
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return names[int((float(degrees) % 360) / 22.5 + 0.5) % 16]


def aqi_category(us_aqi) -> str:
    if us_aqi is None or pd.isna(us_aqi):
        return "n/a"
    v = float(us_aqi)
    for limit, name in [(50, "Good"), (100, "Moderate"), (150, "Unhealthy for sensitive groups"),
                        (200, "Unhealthy"), (300, "Very unhealthy")]:
        if v <= limit:
            return name
    return "Hazardous"
