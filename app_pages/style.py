"""Visual style: CSS, weather-aware gradients and small HTML builders (pure functions, tested)."""
import html

import pandas as pd

from src.live.weather_api import wmo_description

SOURCE_NOTE = ("Forecast source: Open-Meteo weather-model forecasts. "
               "These are NOT predictions from this project's own machine-learning models.")

CSS = """
<style>
section[data-testid="stSidebar"] {background: linear-gradient(180deg, #0d1b2a 0%, #1b263b 100%);}
div[data-testid="stMetric"] {background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px; padding: 14px 16px; transition: transform .2s ease, box-shadow .2s ease;}
div[data-testid="stMetric"]:hover {transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,.35);}
h1, h2, h3 {letter-spacing: -0.3px;}
@keyframes drift {0% {background-position: 0% 50%;} 50% {background-position: 100% 50%;} 100% {background-position: 0% 50%;}}
@keyframes rise {from {opacity: 0; transform: translateY(10px);} to {opacity: 1; transform: none;}}
@keyframes tick {from {transform: translateX(0);} to {transform: translateX(-50%);}}
.hero {border-radius: 22px; padding: 28px 34px; color: #fff; display: flex; justify-content: space-between;
  align-items: center; flex-wrap: wrap; gap: 16px; margin: 6px 0 18px; background-size: 220% 220%;
  animation: drift 14s ease infinite, rise .6s ease; box-shadow: 0 12px 34px rgba(0,0,0,.4);}
.hero-place {font-size: 1.05rem; opacity: .9; letter-spacing: .4px;}
.hero-cond {font-size: 1.3rem; margin-top: 2px;}
.hero-temp {font-size: 4.6rem; font-weight: 800; line-height: 1.05; margin: 2px 0;}
.hero-sub {opacity: .92;}
.hero-chips {display: flex; flex-wrap: wrap; gap: 8px; max-width: 420px; justify-content: flex-end;}
.chip {background: rgba(255,255,255,.16); border-radius: 999px; padding: 7px 14px; font-size: .92rem;
  backdrop-filter: blur(6px);}
.days {display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; margin: 8px 0 20px;}
.day {background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.08); border-radius: 16px;
  padding: 14px 10px; text-align: center; transition: transform .2s ease, box-shadow .2s ease; animation: rise .5s ease;}
.day:hover {transform: translateY(-4px); box-shadow: 0 10px 26px rgba(0,0,0,.4);}
.day-name {font-weight: 700;} .day-date {opacity: .65; font-size: .8rem;}
.day-icon {font-size: 2rem; margin: 6px 0 2px;} .day-cond {font-size: .8rem; opacity: .8; min-height: 2.1em;}
.day-range {display: flex; align-items: center; justify-content: center; font-size: .9rem; margin: 8px 0;}
.bar {position: relative; flex: 1; height: 6px; border-radius: 3px; background: rgba(255,255,255,.12); margin: 0 8px;}
.fill {position: absolute; top: 0; bottom: 0; border-radius: 3px;}
.day-rain {font-size: .85rem; opacity: .85;}
.ticker {overflow: hidden; white-space: nowrap; border-radius: 12px; background: rgba(255,255,255,.05);
  padding: 10px 0; margin-bottom: 14px; border: 1px solid rgba(255,255,255,.08);}
.ticker-inner {display: inline-block; animation: tick 90s linear infinite;}
.tick {margin: 0 22px;}
.stat {background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.08); border-radius: 16px;
  padding: 14px 16px; animation: rise .5s ease;}
.stat-label {opacity: .7; font-size: .8rem; text-transform: uppercase; letter-spacing: .6px;}
.stat-value {font-size: 1.7rem; font-weight: 800;} .stat-sub {opacity: .85;}
</style>
"""


def inject_css():
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)


def sky_gradient(code, is_day: bool) -> str:
    """A three-stop gradient that matches the weather and time of day (animated by the CSS)."""
    c = None if code is None or pd.isna(code) else int(code)
    if c in (95, 96, 99):
        a, b = "#2b1055", "#7597de"
    elif c in (71, 73, 75, 77, 85, 86):
        a, b = "#5d7a99", "#a8c0d6"
    elif c in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82):
        a, b = "#3a4f7a", "#182848"
    elif c in (45, 48):
        a, b = "#6b7a8f", "#b0bac6"
    elif c in (0, 1):
        a, b = ("#1565c0", "#4fc3f7") if is_day else ("#0f2027", "#2c5364")
    elif c in (2, 3):
        a, b = ("#546e7a", "#263238") if is_day else ("#232526", "#414345")
    else:
        a, b = "#1f3a5f", "#3b6a99"
    return f"linear-gradient(120deg, {a}, {b}, {a})"


def temp_color(t) -> str:
    if t is None or pd.isna(t):
        return "#90a4ae"
    for limit, color in [(0, "#7fb3ff"), (10, "#4fc3f7"), (20, "#81c784"), (30, "#ffd54f"), (38, "#ff9800")]:
        if t < limit:
            return color
    return "#f44336"


def fmt0(value, suffix="") -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.0f}{suffix}"


def temp_bar(lo, hi, week_lo, week_hi):
    """Left offset and width (in %) of a day's temperature range inside the week's range."""
    span = max(week_hi - week_lo, 0.1)
    left = min(max((lo - week_lo) / span * 100, 0), 100)
    width = min(max((hi - lo) / span * 100, 4), 100 - left)
    return left, width


def hero_html(place, icon, text, temp, feels, tmax, tmin, chips, gradient) -> str:
    chips_html = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips)
    return (f'<div class="hero" style="background:{gradient}"><div>'
            f'<div class="hero-place">{html.escape(place)}</div>'
            f'<div class="hero-cond">{icon} {html.escape(text)}</div>'
            f'<div class="hero-temp">{html.escape(temp)}</div>'
            f'<div class="hero-sub">Feels like {html.escape(feels)} &middot; High {html.escape(tmax)} '
            f'&middot; Low {html.escape(tmin)}</div></div><div class="hero-chips">{chips_html}</div></div>')


def day_cards_html(daily: pd.DataFrame) -> str:
    wlo, whi = daily["temperature_2m_min"].min(), daily["temperature_2m_max"].max()
    cards = []
    for _, r in daily.iterrows():
        text, icon = wmo_description(r["weather_code"])
        left, width = temp_bar(r["temperature_2m_min"], r["temperature_2m_max"], wlo, whi)
        c1, c2 = temp_color(r["temperature_2m_min"]), temp_color(r["temperature_2m_max"])
        cards.append(
            f'<div class="day"><div class="day-name">{r["time"].strftime("%a")}</div>'
            f'<div class="day-date">{r["time"].strftime("%d %b")}</div><div class="day-icon">{icon}</div>'
            f'<div class="day-cond">{html.escape(text)}</div><div class="day-range">'
            f'<span>{fmt0(r["temperature_2m_min"], "°")}</span><div class="bar"><div class="fill" '
            f'style="left:{left:.1f}%;width:{width:.1f}%;background:linear-gradient(90deg,{c1},{c2})"></div></div>'
            f'<span>{fmt0(r["temperature_2m_max"], "°")}</span></div>'
            f'<div class="day-rain">💧 {fmt0(r["precipitation_probability_max"], "%")}</div></div>')
    return '<div class="days">' + "".join(cards) + "</div>"


def ticker_html(df: pd.DataFrame) -> str:
    items = "".join(f'<span class="tick">{row.icon} {html.escape(str(row.city))} <b>{row.temp:.0f}°</b></span>'
                    for row in df.itertuples())
    return f'<div class="ticker"><div class="ticker-inner">{items}{items}</div></div>'


def stat_html(label, value, sub) -> str:
    return (f'<div class="stat"><div class="stat-label">{html.escape(label)}</div>'
            f'<div class="stat-value">{html.escape(value)}</div><div class="stat-sub">{html.escape(sub)}</div></div>')


def gauge(value, title, max_value, steps):
    """A semicircle gauge with coloured bands (plotly is imported here so tests need not install it)."""
    import plotly.graph_objects as go
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value), title={"text": title},
        gauge={"axis": {"range": [0, max_value]}, "bar": {"color": "rgba(255,255,255,0.9)", "thickness": 0.25},
               "steps": [{"range": [a, b], "color": c} for a, b, c in steps]}))
    fig.update_layout(height=240, margin={"l": 20, "r": 20, "t": 50, "b": 10}, paper_bgcolor="rgba(0,0,0,0)")
    return fig


AQI_STEPS = [(0, 50, "#2ecc71"), (50, 100, "#f1c40f"), (100, 150, "#e67e22"), (150, 200, "#e74c3c"), (200, 300, "#8e44ad")]
UV_STEPS = [(0, 3, "#2ecc71"), (3, 6, "#f1c40f"), (6, 8, "#e67e22"), (8, 11, "#e74c3c"), (11, 15, "#8e44ad")]
