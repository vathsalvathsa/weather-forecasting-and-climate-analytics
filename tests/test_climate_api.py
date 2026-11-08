import numpy as np
import pandas as pd
import pytest

from src.live import climate_api as CA
from src.live.weather_api import WeatherAPIError


def test_annual_means_and_incomplete_years(monkeypatch):
    days = pd.date_range("1991-01-01", "2025-12-31")
    temp = 20 + 0.03 * (days.year - 1991)                                  # +0.3 per decade, no noise
    payload = {"daily": {"time": days.strftime("%Y-%m-%d").tolist(), "temperature_2m_mean": temp.tolist()}}
    seen = {}
    monkeypatch.setattr(CA, "_get_json", lambda url, params, timeout=20: seen.update(params) or payload)
    out = CA.fetch_annual_mean_temperature(10.0, 20.0)
    assert seen["models"] == "era5" and seen["daily"] == "temperature_2m_mean"
    assert len(out) == 35 and out["year"].iloc[0] == 1991
    assert out["tmean"].iloc[-1] == pytest.approx(20 + 0.03 * 34)
    payload["daily"]["time"] = payload["daily"]["time"][:-100]              # 2025 now incomplete
    payload["daily"]["temperature_2m_mean"] = payload["daily"]["temperature_2m_mean"][:-100]
    assert CA.fetch_annual_mean_temperature(10.0, 20.0)["year"].iloc[-1] == 2024


def test_missing_history_raises(monkeypatch):
    monkeypatch.setattr(CA, "_get_json", lambda url, params, timeout=20: {"daily": {}})
    with pytest.raises(WeatherAPIError, match="no temperature history"):
        CA.fetch_annual_mean_temperature(1.0, 2.0)
