"""World Weather Map: current conditions for many cities at once, with a live ticker and highlights."""
import plotly.graph_objects as go
import streamlit as st

from app_pages.data import cached_world
from app_pages.style import SOURCE_NOTE, stat_html, ticker_html
from src.live.weather_api import WeatherAPIError


def highlights(df):
    """The hottest, coldest, windiest and wettest city right now."""
    hot, cold = df.loc[df["temp"].idxmax()], df.loc[df["temp"].idxmin()]
    wind = df.loc[df["wind"].fillna(0).idxmax()]
    wet = df.loc[df["precip"].fillna(0).idxmax()]
    wet_value = ("No rain right now", "") if (wet["precip"] or 0) <= 0 else (f"{wet['precip']:.1f} mm", wet["city"])
    return [("Hottest now", f"{hot['temp']:.0f}°C", hot["city"]), ("Coldest now", f"{cold['temp']:.0f}°C", cold["city"]),
            ("Windiest now", f"{wind['wind']:.0f} km/h", wind["city"]), ("Wettest now", wet_value[0], wet_value[1])]


def render():
    st.title("World Weather Map")
    st.info(SOURCE_NOTE + " One request gathers all the cities below; the data is cached for 10 minutes.")
    try:
        df = cached_world()
    except WeatherAPIError as exc:
        st.error(str(exc))
        return

    st.markdown(ticker_html(df), unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (label, value, sub) in zip(cols, highlights(df)):
        col.markdown(stat_html(label, value, sub), unsafe_allow_html=True)

    hover = [f"<b>{r.city}</b>, {r.country}<br>{r.icon} {r.text}<br>{r.temp:.1f} °C · humidity "
             f"{(r.humidity if r.humidity == r.humidity else 0):.0f}%<br>wind {(r.wind if r.wind == r.wind else 0):.0f} km/h"
             for r in df.itertuples()]
    size = (df["wind"].fillna(0) * 0.6 + 9).clip(9, 32)
    fig = go.Figure(go.Scattergeo(
        lat=df["lat"], lon=df["lon"], text=hover, hoverinfo="text", mode="markers",
        marker={"size": size, "color": df["temp"], "colorscale": "RdYlBu_r", "cmin": -10, "cmax": 40,
                "colorbar": {"title": "°C"}, "line": {"width": 1, "color": "rgba(255,255,255,0.8)"}, "opacity": 0.92}))
    fig.update_geos(projection_type="natural earth", showland=True, landcolor="#1b2a41", showocean=True,
                    oceancolor="#0b1220", showcountries=True, countrycolor="#33415c", showframe=False,
                    bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=540, margin={"l": 0, "r": 0, "t": 10, "b": 0}, paper_bgcolor="rgba(0,0,0,0)",
                      title="Current temperature (colour) and wind speed (marker size)")
    st.plotly_chart(fig)

    with st.expander("All cities (table)"):
        table = df[["city", "country", "icon", "text", "temp", "humidity", "wind", "precip"]].rename(columns={
            "city": "City", "country": "Country", "icon": "", "text": "Conditions", "temp": "°C",
            "humidity": "Humidity %", "wind": "Wind km/h", "precip": "Rain mm"})
        st.dataframe(table.sort_values("°C", ascending=False), hide_index=True)
