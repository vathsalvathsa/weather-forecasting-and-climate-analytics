import pytest

from src.live import world_api as WA
from src.live.weather_api import WeatherAPIError

CITIES = [{"name": "A", "country": "X", "lat": 1.0, "lon": 2.0}, {"name": "B", "country": "Y", "lat": 3.0, "lon": 4.0}]


def item(temp, code=0):
    return {"current": {"temperature_2m": temp, "relative_humidity_2m": 50, "precipitation": 0.0,
                        "wind_speed_10m": 10.0, "weather_code": code}}


def test_snapshot_parses_a_list_and_sends_all_coordinates(monkeypatch):
    seen = {}
    monkeypatch.setattr(WA, "_get_json", lambda url, params, timeout=20: seen.update(params) or [item(20.0, 3), item(31.5)])
    df = WA.get_world_snapshot(CITIES)
    assert seen["latitude"] == "1.0,3.0" and seen["longitude"] == "2.0,4.0"
    assert list(df["city"]) == ["A", "B"] and df["temp"].tolist() == [20.0, 31.5]
    assert df.loc[0, "text"] == "Overcast"


def test_snapshot_handles_a_single_place_dict_and_count_mismatch(monkeypatch):
    monkeypatch.setattr(WA, "_get_json", lambda url, params, timeout=20: item(5.0))
    assert len(WA.get_world_snapshot(CITIES[:1])) == 1
    with pytest.raises(WeatherAPIError, match="incomplete"):
        WA.get_world_snapshot(CITIES)                          # one result for two cities


def test_snapshot_drops_cities_without_a_temperature_and_fails_if_none(monkeypatch):
    monkeypatch.setattr(WA, "_get_json", lambda url, params, timeout=20: [item(None), item(8.0)])
    assert list(WA.get_world_snapshot(CITIES)["city"]) == ["B"]
    monkeypatch.setattr(WA, "_get_json", lambda url, params, timeout=20: [item(None), item(None)])
    with pytest.raises(WeatherAPIError, match="no current conditions"):
        WA.get_world_snapshot(CITIES)
