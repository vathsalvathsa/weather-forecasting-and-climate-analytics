import numpy as np
import pandas as pd

from app_pages import style as S


def test_gradient_matches_weather_and_time_of_day():
    assert S.sky_gradient(0, True) != S.sky_gradient(0, False)         # clear day vs clear night
    assert S.sky_gradient(95, True) == S.sky_gradient(95, False)       # storms look the same day or night
    assert S.sky_gradient(None, True) == S.sky_gradient(np.nan, False)  # unknown code falls back to the default
    assert S.sky_gradient(63, True) != S.sky_gradient(71, True)        # rain differs from snow


def test_temperature_colours_and_range_bar():
    assert S.temp_color(-5) != S.temp_color(15) != S.temp_color(42)
    assert S.temp_color(None) == S.temp_color(np.nan)
    left, width = S.temp_bar(10, 20, 0, 40)
    assert (left, width) == (25.0, 25.0)
    left, width = S.temp_bar(5, 5, 5, 5)                                # no spread: still visible and inside the bar
    assert 0 <= left <= 100 and 4 <= width <= 100 - left


def test_html_is_escaped_and_complete():
    h = S.hero_html("<script>alert(1)</script>", "☀️", "Clear <b>sky</b>", "21°", "20°", "25°", "15°",
                    ["💧 50% <i>"], S.sky_gradient(0, True))
    assert "<script>" not in h and "&lt;script&gt;" in h and "<b>sky" not in h
    assert h.count('class="chip"') == 1
    stat = S.stat_html("Hottest", "40°C", "<Evil City>")
    assert "<Evil" not in stat and "&lt;Evil City&gt;" in stat


def test_day_cards_and_ticker():
    days = pd.date_range("2026-11-03", periods=7)
    daily = pd.DataFrame({"time": days, "weather_code": [0, 3, 61, 71, 95, 45, np.nan],
                          "temperature_2m_min": np.arange(7.0), "temperature_2m_max": np.arange(7.0) + 8,
                          "precipitation_probability_max": [0, 10, 80, 50, 90, 5, np.nan]})
    html = S.day_cards_html(daily)
    assert html.count('class="day"') == 7 and "n/a" in html            # the missing rain chance shows as n/a
    df = pd.DataFrame({"city": ["Delhi", "<X>"], "temp": [30.2, -3.0], "icon": ["☀️", "❄️"]})
    t = S.ticker_html(df)
    assert t.count('class="tick"') == 4 and "&lt;X&gt;" in t           # items repeated twice for a seamless loop
