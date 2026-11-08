"""Climate Analytics page: long-term description of the past (ERA5), not forecasts."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.climate.trends import annual_trend
from src.live.climate_api import fetch_annual_mean_temperature
from src.live.weather_api import WeatherAPIError, geocode

CAVEAT = ("Trends are for ERA5, a gridded reanalysis, not station records. The test treats each year as independent, "
          "which makes p-values optimistic. 'Not statistically clear' means the 35-year record cannot separate a trend "
          "from year-to-year variability; it does not prove the climate is stable.")


def _read(name, **kw):
    path = config.ROOT / "app_data" / name
    return pd.read_csv(path, **kw) if path.exists() else None


@st.cache_data(ttl=86400, show_spinner="Requesting 35 years of ERA5 data...")
def cached_annual_temperature(lat, lon):
    return fetch_annual_mean_temperature(lat, lon)


def _trend_figure(years, values, trend, title):
    slope_year = trend["sen_slope_per_decade"] / 10
    intercept = float(np.median(np.asarray(values) - slope_year * np.asarray(years)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=values, mode="lines+markers", name="Annual mean"))
    fig.add_trace(go.Scatter(x=[min(years), max(years)], y=[intercept + slope_year * min(years),
                                                              intercept + slope_year * max(years)],
                             mode="lines", name="Theil-Sen trend", line={"dash": "dash"}))
    fig.update_layout(title=title, yaxis={"title": "°C"}, legend={"orientation": "h"},
                      margin={"l": 10, "r": 10, "t": 40, "b": 10})
    return fig


def render():
    st.title("Climate Analytics")
    st.info("Climate analytics **describe the past** (1991 to 2025, ERA5 reanalysis). They are not forecasts, and they "
            "are kept separate from the weather forecasts and the ML predictions elsewhere in this app.")
    annual, trends = _read("climate_annual.csv"), _read("climate_trends.csv")
    monthly, extremes = _read("climate_monthly.csv", parse_dates=["date"]), _read("extremes_annual.csv")
    if any(x is None for x in (annual, trends, monthly, extremes)):
        st.warning("Climate data files are not available in this deployment.")
        return
    city = st.selectbox("City", sorted(annual["city"].unique()))
    tabs = st.tabs(["Temperature trend", "Seasons and anomalies", "Rain", "Extremes", "Any place (live)"])

    with tabs[0]:
        a = annual[annual["city"] == city]
        t = trends[(trends["city"] == city) & trends["variable"].str.contains("temperature")].iloc[0]
        c = st.columns(3)
        c[0].metric("Warming per decade", f"{t['sen_slope_per_decade']:+.2f} °C")
        c[1].metric("Range (Theil-Sen)", f"{t['sen_low_per_decade']:+.2f} to {t['sen_high_per_decade']:+.2f} °C")
        c[2].metric("Kendall p-value", f"{t['p_value']:.3f}")
        st.plotly_chart(_trend_figure(a["year"], a["tmean"], t.to_dict(), f"Annual mean temperature, {city}"))
        st.caption(CAVEAT)
        allt = trends[trends["variable"].str.contains("temperature")].sort_values("sen_slope_per_decade")
        bar = go.Figure(go.Bar(x=allt["sen_slope_per_decade"], y=allt["city"], orientation="h",
                               error_x={"type": "data", "symmetric": False,
                                        "array": allt["sen_high_per_decade"] - allt["sen_slope_per_decade"],
                                        "arrayminus": allt["sen_slope_per_decade"] - allt["sen_low_per_decade"]}))
        bar.update_layout(title="Warming per decade, all 10 cities (bars show the Theil-Sen range)",
                          xaxis={"title": "°C per decade"}, margin={"l": 10, "r": 10, "t": 40, "b": 10})
        st.plotly_chart(bar)

    with tabs[1]:
        m = monthly[monthly["city"] == city].sort_values("date")
        normal = m[m["year"].between(1991, 2020)].groupby("month")["tmean"].mean()
        seasonal = go.Figure(go.Bar(x=list(normal.index), y=normal.values, name="1991-2020 normal"))
        seasonal.update_layout(title=f"Typical monthly mean temperature, {city} (1991-2020)",
                               xaxis={"title": "month"}, yaxis={"title": "°C"}, margin={"l": 10, "r": 10, "t": 40, "b": 10})
        st.plotly_chart(seasonal)
        roll = m["tmean_anomaly"].rolling(12, min_periods=12).mean()
        an = go.Figure()
        an.add_trace(go.Bar(x=m["date"], y=m["tmean_anomaly"], name="Monthly anomaly", opacity=0.45))
        an.add_trace(go.Scatter(x=m["date"], y=roll, name="12-month average", mode="lines"))
        an.update_layout(title="Temperature anomaly vs the 1991-2020 monthly normal", yaxis={"title": "°C"},
                         legend={"orientation": "h"}, margin={"l": 10, "r": 10, "t": 40, "b": 10})
        st.plotly_chart(an)

    with tabs[2]:
        a = annual[annual["city"] == city]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=a["year"], y=a["precip_mm"], name="Annual rain (mm)"))
        fig.add_trace(go.Scatter(x=a["year"], y=a["wet_days"], name="Wet days (>= 1 mm)", yaxis="y2", mode="lines+markers"))
        fig.update_layout(title=f"Rain, {city}", yaxis={"title": "mm"},
                          yaxis2={"title": "days", "overlaying": "y", "side": "right"},
                          legend={"orientation": "h"}, margin={"l": 10, "r": 10, "t": 40, "b": 10})
        st.plotly_chart(fig)
        rt = trends[(trends["city"] == city) & ~trends["variable"].str.contains("temperature")]
        st.dataframe(rt[["variable", "sen_slope_per_decade", "sen_low_per_decade", "sen_high_per_decade", "p_value"]].round(3),
                     hide_index=True)
        st.caption("ERA5 rainfall is model-produced and can differ from gauge measurements, so rain trends are less "
                   "reliable than temperature trends. " + CAVEAT)

    with tabs[3]:
        e = extremes[extremes["city"] == city]
        hot_thr, rain_thr = e["hot_threshold_c"].iloc[0], e["heavy_rain_threshold_mm"].iloc[0]
        st.markdown(f"Simple, city-relative definitions (not official warning criteria): a **hot day** has a daily maximum "
                    f"above {hot_thr} °C (this city's 95th percentile, 1991-2020); a **heatwave** is 3 or more hot days "
                    f"in a row; a **heavy-rain day** has at least {rain_thr} mm (the 99th percentile of wet days).")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=e["year"], y=e["hot_days"], name="Hot days"))
        fig.add_trace(go.Scatter(x=e["year"], y=e["heatwaves"], name="Heatwave events", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=e["year"], y=e["heavy_rain_days"], name="Heavy-rain days", mode="lines+markers"))
        fig.update_layout(title=f"Extremes per year, {city}", legend={"orientation": "h"},
                          margin={"l": 10, "r": 10, "t": 40, "b": 10})
        st.plotly_chart(fig)

    with tabs[4]:
        st.markdown("Search any place: the app requests its 1991-2025 ERA5 temperature history live and computes the "
                    "same trend. This can fail when the free weather service is busy.")
        query = st.text_input("Place", value="Paris")
        try:
            places = geocode(query)
        except WeatherAPIError as exc:
            st.error(str(exc))
            return
        if not places:
            st.warning("No place found.")
            return
        labels = [p["label"] for p in places]
        place = places[labels.index(st.selectbox("Select the location", labels))]
        try:
            hist = cached_annual_temperature(place["latitude"], place["longitude"])
            trend = annual_trend(hist["year"], hist["tmean"])
        except (WeatherAPIError, ValueError) as exc:
            st.error(f"Could not compute the trend: {exc}")
            return
        c = st.columns(3)
        c[0].metric("Warming per decade", f"{trend['sen_slope_per_decade']:+.2f} °C")
        c[1].metric("Range (Theil-Sen)", f"{trend['sen_low_per_decade']:+.2f} to {trend['sen_high_per_decade']:+.2f} °C")
        c[2].metric("Kendall p-value", f"{trend['p_value']:.3f}")
        st.plotly_chart(_trend_figure(hist["year"], hist["tmean"], trend, f"Annual mean temperature, {place['label']}"))
        st.caption(CAVEAT)
