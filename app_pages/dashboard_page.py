"""Live Dashboard: a weather-aware hero banner, gauges, day cards and interactive charts."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_pages.data import cached_air, cached_forecast, cached_geocode
from app_pages.style import (AQI_STEPS, SOURCE_NOTE, UV_STEPS, day_cards_html, fmt0, gauge, hero_html,
                             sky_gradient)
from src.live.weather_api import (WeatherAPIError, aqi_category, current_hour_values, wind_compass,
                                  wmo_description)

LAYOUT = {"margin": {"l": 10, "r": 10, "t": 40, "b": 10}, "hovermode": "x unified",
          "legend": {"orientation": "h"}, "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"}


def weather_notices(daily: pd.DataFrame) -> list:
    """Simple rule-based notices (thresholds are illustrative, not official warnings)."""
    notes = []
    if (daily["temperature_2m_max"] >= 40).any():
        notes.append("Very hot day(s) forecast (max temperature of 40 °C or more).")
    if (daily["temperature_2m_min"] <= 0).any():
        notes.append("Freezing night(s) forecast (min temperature of 0 °C or less).")
    if (daily["precipitation_sum"] >= 50).any():
        notes.append("Very heavy rain forecast (50 mm or more in a day).")
    if (daily["wind_speed_10m_max"] >= 60).any():
        notes.append("Strong wind forecast (60 km/h or more).")
    if (daily["uv_index_max"] >= 8).any():
        notes.append("Very high UV index forecast (8 or more).")
    return notes


def hourly_figures(fc: dict):
    """Three interactive charts for the next 48 hours."""
    h = fc["hourly"]
    start = pd.to_datetime(fc["current"]["time"]).floor("h")
    h48 = h[h["time"] >= start].head(48).reset_index(drop=True)

    base = float(h48["temperature_2m"].min()) - 2
    temp = go.Figure()
    temp.add_trace(go.Scatter(x=h48["time"], y=[base] * len(h48), mode="lines", line={"width": 0},
                              showlegend=False, hoverinfo="skip"))
    temp.add_trace(go.Scatter(x=h48["time"], y=h48["temperature_2m"], name="Temperature (°C)", mode="lines",
                              line={"shape": "spline", "width": 3, "color": "#ffb74d"},
                              fill="tonexty", fillcolor="rgba(255,183,77,0.18)"))
    temp.add_trace(go.Bar(x=h48["time"], y=h48["precipitation"], name="Precipitation (mm)", yaxis="y2",
                          marker={"color": "rgba(79,195,247,0.65)"}))
    hi, lo = h48["temperature_2m"].idxmax(), h48["temperature_2m"].idxmin()
    temp.add_trace(go.Scatter(x=h48.loc[[hi, lo], "time"], y=h48.loc[[hi, lo], "temperature_2m"], mode="markers+text",
                              text=[f"{h48.loc[hi, 'temperature_2m']:.0f}°", f"{h48.loc[lo, 'temperature_2m']:.0f}°"],
                              textposition="top center", name="Peak / low", marker={"size": 10, "color": "#ff7043"}))
    temp.update_layout(title="Temperature and rain, next 48 hours", yaxis={"title": "°C"},
                       yaxis2={"title": "mm", "overlaying": "y", "side": "right", "showgrid": False}, **LAYOUT)

    air = go.Figure()
    air.add_trace(go.Scatter(x=h48["time"], y=h48["wind_speed_10m"], name="Wind (km/h)", mode="lines",
                             line={"shape": "spline", "color": "#80cbc4"}))
    air.add_trace(go.Scatter(x=h48["time"], y=h48["relative_humidity_2m"], name="Humidity (%)", mode="lines",
                             line={"shape": "spline", "color": "#ce93d8"}))
    air.add_trace(go.Scatter(x=h48["time"], y=h48["cloud_cover"], name="Cloud cover (%)", mode="lines",
                             line={"shape": "spline", "color": "#b0bec5", "dash": "dot"}))
    air.update_layout(title="Wind, humidity and cloud, next 48 hours", **LAYOUT)

    rain = go.Figure(go.Scatter(x=h48["time"], y=h48["precipitation_probability"], mode="lines", fill="tozeroy",
                                line={"shape": "spline", "color": "#4fc3f7"}, name="Chance of precipitation (%)"))
    rain.update_layout(title="Probability of precipitation, next 48 hours", yaxis={"range": [0, 100], "title": "%"},
                       **LAYOUT)
    return temp, air, rain


def render():
    st.title("Live Weather Dashboard")
    query = st.text_input("Search any city or place", value="London")
    try:
        places = cached_geocode(query)
    except WeatherAPIError as exc:
        st.error(str(exc))
        return
    if not places:
        st.warning("No place found. Try a different spelling or add the country.")
        return
    labels = [p["label"] for p in places]
    place = places[labels.index(st.selectbox("Select the location", labels))]
    try:
        with st.spinner("Loading forecast..."):
            fc = cached_forecast(place["latitude"], place["longitude"])
            aq = cached_air(place["latitude"], place["longitude"])
    except WeatherAPIError as exc:
        st.error(str(exc))
        return
    st.info(SOURCE_NOTE)

    cur, today, hour = fc["current"], fc["daily"].iloc[0], current_hour_values(fc)
    now = pd.to_datetime(cur["time"])
    is_day = bool(today["sunrise"] <= now <= today["sunset"])
    text, icon = wmo_description(cur.get("weather_code"))
    chips = [f"💧 Humidity {fmt0(cur.get('relative_humidity_2m'), '%')}",
             f"🌬️ {fmt0(cur.get('wind_speed_10m'), ' km/h')} {wind_compass(cur.get('wind_direction_10m'))}",
             f"☁️ Cloud {fmt0(cur.get('cloud_cover'), '%')}",
             f"🌅 {today['sunrise'].strftime('%H:%M')}  🌇 {today['sunset'].strftime('%H:%M')}",
             f"{'☀️ Day' if is_day else '🌙 Night'} · {fc.get('timezone')}"]
    st.markdown(hero_html(place["label"], icon, text, fmt0(cur.get("temperature_2m"), "°"),
                          fmt0(cur.get("apparent_temperature"), "°"), fmt0(today["temperature_2m_max"], "°"),
                          fmt0(today["temperature_2m_min"], "°"), chips,
                          sky_gradient(cur.get("weather_code"), is_day)), unsafe_allow_html=True)

    notes = weather_notices(fc["daily"])
    if notes:
        st.warning("**Weather notices** (rule-based and illustrative; follow your national weather service for "
                   "official warnings)\n\n" + "\n".join(f"- {n}" for n in notes))

    g1, g2, g3 = st.columns(3)
    uv = hour.get("uv_index")
    if aq is not None and aq.get("us_aqi") is not None:
        g1.plotly_chart(gauge(aq["us_aqi"], "Air quality (US AQI)", 300, AQI_STEPS))
        g1.caption(aqi_category(aq.get("us_aqi")))
    else:
        g1.metric("Air quality (US AQI)", "n/a")
    if uv is not None and not pd.isna(uv):
        g2.plotly_chart(gauge(uv, "UV index", 15, UV_STEPS))
    else:
        g2.metric("UV index", "n/a")
    vis = hour.get("visibility")
    g3.metric("Pressure (sea level)", fmt0(cur.get("pressure_msl"), " hPa"))
    g3.metric("Visibility", fmt0(None if vis is None else vis / 1000.0, " km"))
    g3.metric("Rain (last hour)", fmt0(cur.get("precipitation"), " mm"))

    st.subheader("Next 48 hours")
    temp, air, rain = hourly_figures(fc)
    tabs = st.tabs(["Temperature and rain", "Wind, humidity and cloud", "Chance of rain"])
    with tabs[0]:
        st.plotly_chart(temp)
    with tabs[1]:
        st.plotly_chart(air)
    with tabs[2]:
        st.plotly_chart(rain)

    st.subheader("7-day outlook")
    st.markdown(day_cards_html(fc["daily"]), unsafe_allow_html=True)
    with st.expander("Detailed daily table"):
        d = fc["daily"]
        table = pd.DataFrame({
            "Date": d["time"].dt.strftime("%a %d %b"), "Max °C": d["temperature_2m_max"],
            "Min °C": d["temperature_2m_min"], "Rain mm": d["precipitation_sum"],
            "Rain chance %": d["precipitation_probability_max"], "Wind max km/h": d["wind_speed_10m_max"],
            "UV max": d["uv_index_max"]})
        st.dataframe(table, hide_index=True)
