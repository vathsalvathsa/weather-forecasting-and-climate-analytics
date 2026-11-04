import numpy as np
import pandas as pd
import pytest

from src.live import weather_api as W


class FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code, self._payload = status, payload or {}

    def json(self):
        return self._payload


def patch_get(monkeypatch, resp):
    monkeypatch.setattr(W.requests, "get", lambda *a, **k: resp)


def forecast_payload(n_hours=168):
    times = pd.date_range("2026-11-03", periods=n_hours, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    days = pd.date_range("2026-11-03", periods=7).strftime("%Y-%m-%d").tolist()
    hourly = {"time": times}
    for v in W.HOURLY_VARS:
        hourly[v] = [1.0] * n_hours
    daily = {"time": days, "sunrise": [d + "T06:30" for d in days], "sunset": [d + "T17:30" for d in days]}
    for v in W.DAILY_VARS:
        daily.setdefault(v, [1.0] * 7)
    current = {"time": "2026-11-03T10:15", "temperature_2m": 21.5, "weather_code": 3, "wind_direction_10m": 200}
    return {"current": current, "current_units": {"temperature_2m": "°C"}, "hourly": hourly, "daily": daily,
            "timezone": "Asia/Kolkata", "elevation": 230.0}


def test_geocode_returns_unique_labels(monkeypatch):
    payload = {"results": [
        {"name": "Springfield", "admin1": "Illinois", "country": "USA", "latitude": 39.8, "longitude": -89.6},
        {"name": "Springfield", "admin1": "Illinois", "country": "USA", "latitude": 37.2, "longitude": -93.3},
        {"name": "Delhi", "admin1": "Delhi", "country": "India", "latitude": 28.6, "longitude": 77.2}]}
    patch_get(monkeypatch, FakeResp(200, payload))
    places = W.geocode("Springfield")
    labels = [p["label"] for p in places]
    assert len(set(labels)) == len(labels) == 3
    assert labels[2] == "Delhi, Delhi, India"


def test_geocode_handles_no_results_and_short_queries(monkeypatch):
    patch_get(monkeypatch, FakeResp(200, {"generationtime_ms": 0.3}))        # API omits 'results'
    assert W.geocode("zzzzzz") == []
    assert W.geocode("a") == []
    assert W.geocode("") == []


def test_forecast_parsing(monkeypatch):
    patch_get(monkeypatch, FakeResp(200, forecast_payload()))
    fc = W.get_forecast(28.6, 77.2)
    assert len(fc["hourly"]) == 168 and len(fc["daily"]) == 7
    assert pd.api.types.is_datetime64_any_dtype(fc["hourly"]["time"])
    assert pd.api.types.is_datetime64_any_dtype(fc["daily"]["sunrise"])
    row = W.current_hour_values(fc)
    assert row["time"] == pd.Timestamp("2026-11-03 10:00")                    # nearest hour to 10:15


def test_incomplete_forecast_raises(monkeypatch):
    patch_get(monkeypatch, FakeResp(200, {"current": {}}))
    with pytest.raises(W.WeatherAPIError, match="incomplete"):
        W.get_forecast(1, 2)


def test_rate_limit_and_server_errors_have_friendly_messages(monkeypatch):
    patch_get(monkeypatch, FakeResp(429, {}))
    with pytest.raises(W.WeatherAPIError, match="rate-limiting"):
        W.get_forecast(1, 2)
    patch_get(monkeypatch, FakeResp(500, {"reason": "boom"}))
    with pytest.raises(W.WeatherAPIError, match="500"):
        W.get_forecast(1, 2)


def test_network_failure_is_wrapped(monkeypatch):
    def boom(*a, **k):
        raise W.requests.ConnectionError("offline")
    monkeypatch.setattr(W.requests, "get", boom)
    with pytest.raises(W.WeatherAPIError, match="Could not reach"):
        W.geocode("London")


def test_air_quality_never_raises(monkeypatch):
    patch_get(monkeypatch, FakeResp(500, {}))
    assert W.get_air_quality(1, 2) is None
    patch_get(monkeypatch, FakeResp(200, {"current": {"us_aqi": None}}))
    assert W.get_air_quality(1, 2) is None
    patch_get(monkeypatch, FakeResp(200, {"current": {"us_aqi": 87, "pm2_5": 30.0}}))
    assert W.get_air_quality(1, 2)["us_aqi"] == 87


def test_helpers():
    assert W.wmo_description(63) == ("Rain", "🌧️")
    assert W.wmo_description(None)[0] == "Unknown" and W.wmo_description(12345)[0] == "Unknown"
    assert W.wind_compass(0) == "N" and W.wind_compass(90) == "E" and W.wind_compass(359) == "N"
    assert W.wind_compass(200) == "SSW" and W.wind_compass(np.nan) == "n/a"
    assert [W.aqi_category(v) for v in (10, 75, 120, 180, 250, 400)] == [
        "Good", "Moderate", "Unhealthy for sensitive groups", "Unhealthy", "Very unhealthy", "Hazardous"]
