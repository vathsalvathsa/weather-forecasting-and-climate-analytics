"""Live ERA5 request for the 'any place' climate view (annual mean temperature, 1991-2025)."""
import pandas as pd

from src.live.weather_api import WeatherAPIError, _get_json

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_annual_mean_temperature(lat: float, lon: float, first_year: int = 1991, last_year: int = 2025) -> pd.DataFrame:
    """Annual mean of daily mean temperature (ERA5, explicit source). Incomplete years are dropped."""
    data = _get_json(ARCHIVE_URL, {
        "latitude": lat, "longitude": lon, "start_date": f"{first_year}-01-01", "end_date": f"{last_year}-12-31",
        "daily": "temperature_2m_mean", "timezone": "auto", "models": "era5"}, timeout=60)
    daily = data.get("daily")
    if not daily or "time" not in daily or "temperature_2m_mean" not in daily:
        raise WeatherAPIError("The climate service returned no temperature history for this place.")
    df = pd.DataFrame(daily).dropna()
    df["time"] = pd.to_datetime(df["time"])
    g = df.groupby(df["time"].dt.year)["temperature_2m_mean"].agg(["mean", "count"])
    g = g[g["count"] >= 360]
    return g["mean"].rename_axis("year").reset_index(name="tmean")
