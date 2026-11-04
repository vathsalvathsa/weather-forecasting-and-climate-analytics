"""Weather Forecasting & Climate Analytics - Streamlit app (part A: live pages).

Run locally:  streamlit run app.py
Weather data by Open-Meteo.com (CC BY 4.0).
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.live.weather_api import (WeatherAPIError, aqi_category, current_hour_values, geocode,
                                  get_air_quality, get_forecast, wind_compass, wmo_description)

PAGES = ["Live Dashboard", "City Comparison", "ML Predictions", "Climate Analytics",
         "Model Performance", "Dataset Information", "About Project", "References"]
SOURCE_NOTE = ("Forecast source: Open-Meteo weather-model forecasts. "
               "These are NOT predictions from this project's own machine-learning models.")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_geocode(query):
    return geocode(query)


@st.cache_data(ttl=600, show_spinner=False)
def cached_forecast(lat, lon):
    return get_forecast(lat, lon)


@st.cache_data(ttl=600, show_spinner=False)
def cached_air(lat, lon):
    return get_air_quality(lat, lon)


def fmt(value, unit="", digits=1):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}{unit}"


# ---------------------------------------------------------------- Live Dashboard
def render_current(place, fc, aq):
    cur = fc["current"]
    hour = current_hour_values(fc)
    today = fc["daily"].iloc[0]
    text, icon = wmo_description(cur.get("weather_code"))
    st.subheader(f"{icon} {place['label']}: {text}")

    c = st.columns(4)
    c[0].metric("Temperature", fmt(cur.get("temperature_2m"), " °C"))
    c[1].metric("Feels like", fmt(cur.get("apparent_temperature"), " °C"))
    c[2].metric("Humidity", fmt(cur.get("relative_humidity_2m"), " %", 0))
    c[3].metric("Pressure (sea level)", fmt(cur.get("pressure_msl"), " hPa", 0))

    c = st.columns(4)
    wind = f"{fmt(cur.get('wind_speed_10m'), ' km/h')} {wind_compass(cur.get('wind_direction_10m'))}"
    c[0].metric("Wind", wind)
    c[1].metric("Cloud cover", fmt(cur.get("cloud_cover"), " %", 0))
    c[2].metric("Rain (last hour)", fmt(cur.get("precipitation"), " mm"))
    vis = hour.get("visibility")
    c[3].metric("Visibility", fmt(None if vis is None else vis / 1000.0, " km"))

    c = st.columns(4)
    c[0].metric("UV index", fmt(hour.get("uv_index"), "", 1))
    c[1].metric("Sunrise", today["sunrise"].strftime("%H:%M"))
    c[2].metric("Sunset", today["sunset"].strftime("%H:%M"))
    if aq is None:
        c[3].metric("Air quality (US AQI)", "n/a")
    else:
        c[3].metric("Air quality (US AQI)", fmt(aq.get("us_aqi"), "", 0), aqi_category(aq.get("us_aqi")),
                    delta_color="off")
    st.caption(f"Local time zone: {fc.get('timezone')}. Observation time: {cur.get('time')}.")


def render_hourly(fc):
    h = fc["hourly"]
    start = pd.to_datetime(fc["current"]["time"]).floor("h")
    h48 = h[h["time"] >= start].head(48)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h48["time"], y=h48["temperature_2m"], name="Temperature (°C)", mode="lines"))
    fig.add_trace(go.Bar(x=h48["time"], y=h48["precipitation"], name="Precipitation (mm)",
                         yaxis="y2", opacity=0.5))
    fig.update_layout(title="Next 48 hours", yaxis={"title": "°C"},
                      yaxis2={"title": "mm", "overlaying": "y", "side": "right", "showgrid": False},
                      legend={"orientation": "h"}, margin={"l": 10, "r": 10, "t": 40, "b": 10})
    st.plotly_chart(fig)
    p = go.Figure()
    p.add_trace(go.Scatter(x=h48["time"], y=h48["precipitation_probability"], mode="lines",
                           name="Chance of precipitation (%)"))
    p.update_layout(title="Probability of precipitation (next 48 hours)", yaxis={"title": "%", "range": [0, 100]},
                    margin={"l": 10, "r": 10, "t": 40, "b": 10})
    st.plotly_chart(p)


def render_daily(fc):
    d = fc["daily"].copy()
    d["conditions"] = [" ".join(reversed(wmo_description(c))) for c in d["weather_code"]]
    table = pd.DataFrame({
        "Date": d["time"].dt.strftime("%a %d %b"), "Conditions": d["conditions"],
        "Max °C": d["temperature_2m_max"], "Min °C": d["temperature_2m_min"],
        "Rain mm": d["precipitation_sum"], "Rain chance %": d["precipitation_probability_max"],
        "Wind max km/h": d["wind_speed_10m_max"], "UV max": d["uv_index_max"]})
    st.dataframe(table, hide_index=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["time"], y=d["temperature_2m_max"], name="Max °C", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=d["time"], y=d["temperature_2m_min"], name="Min °C", mode="lines+markers"))
    fig.update_layout(title="7-day temperature range", legend={"orientation": "h"},
                      margin={"l": 10, "r": 10, "t": 40, "b": 10})
    st.plotly_chart(fig)


def render_alerts(fc):
    """Simple rule-based notices (thresholds are illustrative, not official warnings)."""
    d = fc["daily"]
    notes = []
    if (d["temperature_2m_max"] >= 40).any():
        notes.append("Very hot day(s) forecast (max temperature of 40 °C or more).")
    if (d["temperature_2m_min"] <= 0).any():
        notes.append("Freezing night(s) forecast (min temperature of 0 °C or less).")
    if (d["precipitation_sum"] >= 50).any():
        notes.append("Very heavy rain forecast (50 mm or more in a day).")
    if (d["wind_speed_10m_max"] >= 60).any():
        notes.append("Strong wind forecast (60 km/h or more).")
    if (d["uv_index_max"] >= 8).any():
        notes.append("Very high UV index forecast (8 or more).")
    if notes:
        st.warning("**Weather notices** (rule-based and illustrative; follow your national weather service for official warnings)\n\n"
                   + "\n".join(f"- {n}" for n in notes))


def page_dashboard():
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
    render_current(place, fc, aq)
    render_alerts(fc)
    render_hourly(fc)
    st.subheader("7-day forecast")
    render_daily(fc)


# ---------------------------------------------------------------- City Comparison
def page_comparison():
    st.title("City Comparison")
    st.info(SOURCE_NOTE)
    text = st.text_area("Cities (one per line, up to 5)", value="Delhi\nLondon\nSingapore")
    names = [n.strip() for n in text.splitlines() if n.strip()][:5]
    rows, fig = [], go.Figure()
    for name in names:
        try:
            places = cached_geocode(name)
            if not places:
                st.warning(f"No place found for '{name}'.")
                continue
            p = places[0]
            fc = cached_forecast(p["latitude"], p["longitude"])
        except WeatherAPIError as exc:
            st.error(f"{name}: {exc}")
            continue
        cur, today = fc["current"], fc["daily"].iloc[0]
        rows.append({"City": p["label"], "Now °C": cur.get("temperature_2m"),
                     "Humidity %": cur.get("relative_humidity_2m"),
                     "Wind km/h": cur.get("wind_speed_10m"),
                     "Today max °C": today["temperature_2m_max"], "Today min °C": today["temperature_2m_min"],
                     "Today rain mm": today["precipitation_sum"]})
        fig.add_trace(go.Scatter(x=fc["daily"]["time"], y=fc["daily"]["temperature_2m_max"],
                                 name=p["label"], mode="lines+markers"))
    if not rows:
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    fig.update_layout(title="Forecast daily maximum temperature (°C)", legend={"orientation": "h"},
                      margin={"l": 10, "r": 10, "t": 40, "b": 10})
    st.plotly_chart(fig)


# ---------------------------------------------------------------- Pages under construction
def page_ml():
    st.title("ML Predictions")
    st.info("Under construction: next-day predictions from this project's own trained models "
            "will appear here, clearly separated from the API forecasts.")


def page_climate():
    st.title("Climate Analytics")
    st.info("Under construction: long-term trends, seasonal patterns and anomalies from ERA5 reanalysis. "
            "Short-term weather forecasts and long-term climate analysis will be kept separate.")


def page_performance():
    st.title("Model Performance")
    st.info("Under construction: evaluation results (error, skill vs baselines, calibration) "
            "will be shown here once the final models are trained. No results are shown until then.")


# ---------------------------------------------------------------- Static pages
def page_dataset():
    st.title("Dataset Information")
    st.markdown(
        "**Source:** ERA5 reanalysis (ECMWF/Copernicus) accessed through the Open-Meteo Historical Weather API "
        "(CC BY 4.0; attribution: *Weather data by Open-Meteo.com*).\n\n"
        f"**Training sample:** {len(config.CITIES)} cities are configured; the models are trained on a climatically "
        "diverse subset, with latitude, longitude and elevation as features.\n\n"
        "**Resolution:** daily values per city, local calendar days.\n\n"
        "**Important limitations:** ERA5 is a model-based reanalysis, not a station observation record. "
        "Rain in particular is model-produced and tends to differ from gauge measurements. "
        "A different default data source in the same API changes in 2017, so every request in this project "
        "pins the source to ERA5 to keep the series consistent.")
    st.subheader("Configured cities")
    st.dataframe(pd.DataFrame(config.CITIES)[["name", "country", "region", "lat", "lon"]], hide_index=True)


def page_about():
    st.title("About this project")
    st.markdown(
        "An M.Tech Data Science & Analytics project on weather forecasting and climate analytics.\n\n"
        "**Three kinds of information are kept apart on purpose:**\n"
        "1. *API forecasts* (Open-Meteo): live weather-model output.\n"
        "2. *Our ML predictions*: models trained on ERA5 history, always compared with strong baselines.\n"
        "3. *Climate analytics*: long-term trends and anomalies, which describe the past and are not forecasts.\n\n"
        "No accuracy figure is shown unless it was measured on data the models never saw.")


def page_references():
    st.title("References and sources")
    st.markdown(
        "- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal "
        "Meteorological Society*, 146, 1999-2049. https://doi.org/10.1002/qj.3803\n"
        "- Zippenfennig, P. Open-Meteo.com Weather API. https://open-meteo.com (CC BY 4.0). "
        "https://doi.org/10.5281/zenodo.7970649")


def main():
    st.set_page_config(page_title="Weather & Climate Analytics", page_icon="🌦️", layout="wide")
    st.sidebar.title("Weather & Climate Analytics")
    page = st.sidebar.radio("Navigation", PAGES)
    st.sidebar.caption("Weather data by Open-Meteo.com (CC BY 4.0)")
    {"Live Dashboard": page_dashboard, "City Comparison": page_comparison, "ML Predictions": page_ml,
     "Climate Analytics": page_climate, "Model Performance": page_performance,
     "Dataset Information": page_dataset, "About Project": page_about,
     "References": page_references}[page]()


if __name__ == "__main__":
    main()
