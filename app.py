"""Weather Forecasting & Climate Analytics - Streamlit app.

Run locally:  streamlit run app.py
Weather data by Open-Meteo.com (CC BY 4.0).
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_pages.data import cached_forecast, cached_geocode
from app_pages.style import SOURCE_NOTE, inject_css
from src import config
from src.live.weather_api import WeatherAPIError

PAGES = ["Live Dashboard", "World Map", "City Comparison", "ML Predictions", "Climate Analytics",
         "Model Performance", "Dataset Information", "About Project", "References"]


def page_dashboard():
    from app_pages import dashboard_page
    dashboard_page.render()


def page_world():
    from app_pages import world_page
    world_page.render()


def page_ml():
    from app_pages import ml_page
    ml_page.render()


def page_climate():
    from app_pages import climate_page
    climate_page.render()


def page_performance():
    from app_pages import performance_page
    performance_page.render()


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
                     "Humidity %": cur.get("relative_humidity_2m"), "Wind km/h": cur.get("wind_speed_10m"),
                     "Today max °C": today["temperature_2m_max"], "Today min °C": today["temperature_2m_min"],
                     "Today rain mm": today["precipitation_sum"]})
        fig.add_trace(go.Scatter(x=fc["daily"]["time"], y=fc["daily"]["temperature_2m_max"], name=p["label"],
                                 mode="lines+markers", line={"shape": "spline"}))
    if not rows:
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    fig.update_layout(title="Forecast daily maximum temperature (°C)", legend={"orientation": "h"},
                      hovermode="x unified", margin={"l": 10, "r": 10, "t": 40, "b": 10},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig)


def page_dataset():
    st.title("Dataset Information")
    st.markdown(
        "**Source:** ERA5 reanalysis (ECMWF/Copernicus) accessed through the Open-Meteo Historical Weather API "
        "(CC BY 4.0; attribution: *Weather data by Open-Meteo.com*).\n\n"
        f"**Training sample:** {len(config.CITIES)} cities are configured; the models are trained on a climatically "
        "diverse subset of 10, with a leave-cities-out test of how well they transfer to cities they never saw.\n\n"
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
        "1. *API forecasts* (Open-Meteo): live weather-model output (Live Dashboard, World Map, City Comparison).\n"
        "2. *Our ML predictions*: models trained on ERA5 history, always compared with strong baselines, shown as a "
        "backtest on past days.\n"
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
    inject_css()
    st.sidebar.title("🌦️ Weather & Climate Analytics")
    page = st.sidebar.radio("Navigation", PAGES)
    st.sidebar.caption("Weather data by Open-Meteo.com (CC BY 4.0)")
    {"Live Dashboard": page_dashboard, "World Map": page_world, "City Comparison": page_comparison,
     "ML Predictions": page_ml, "Climate Analytics": page_climate, "Model Performance": page_performance,
     "Dataset Information": page_dataset, "About Project": page_about,
     "References": page_references}[page]()


if __name__ == "__main__":
    main()
