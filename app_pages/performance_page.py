"""Model Performance page: results read from the saved CSVs (nothing is typed in by hand)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.evaluation import report_tables as RT

NAMES = {"tmean": "Mean temperature", "tmax": "Maximum temperature", "tmin": "Minimum temperature",
         "rh": "Relative humidity", "wind": "Maximum wind speed"}


def _read(name):
    path = config.ROOT / "reports" / name
    return pd.read_csv(path) if path.exists() else None


def render():
    st.title("Model Performance")
    cont, rain, loco = _read("final_test_results.csv"), _read("final_rain_test_results.csv"), _read("leave_cities_out_val.csv")
    if cont is None or rain is None:
        st.warning("Final test results are not available in this deployment.")
        return
    st.info("The models were trained on 1991 to 2020 data and scored **once** on 2021 to 2026, years they never saw. "
            "Data: ERA5 reanalysis for 10 cities. Every number on this page is read from the saved result files.")

    st.subheader("Beating strong baselines")
    st.markdown("A model only matters if it beats simple rules. We compare with **climatology** (the usual value for "
                "that date) and a **damped baseline** (climatology plus today's deviation, fading with lead time). "
                "'Gain' is the percentage reduction in squared error relative to the damped baseline, averaged over "
                "the 10 cities (the fairest single number, since pooled figures favour high-variance cities).")
    gains = RT.city_mean_gain(cont)
    gains["target"] = gains["target"].map(NAMES)
    st.dataframe(gains.round(1), hide_index=True)

    fig = go.Figure()
    for h in (1, 3):
        g = gains[gains["horizon"] == h]
        fig.add_trace(go.Bar(x=g["target"], y=g["mean city gain %"], name=f"{h} day(s) ahead"))
    fig.update_layout(barmode="group", yaxis={"title": "mean gain over damped baseline (%)"},
                      margin={"l": 10, "r": 10, "t": 30, "b": 10}, legend={"orientation": "h"})
    st.plotly_chart(fig)

    st.subheader("Error in physical units (all cities pooled)")
    pooled = RT.pooled_table(cont)
    pooled["target"] = pooled["target"].map(NAMES)
    st.dataframe(pooled.round(3), hide_index=True)

    st.subheader("City by city: mean temperature")
    h = st.radio("Days ahead", [1, 3], horizontal=True)
    series = RT.city_gain(cont, "tmean", h).sort_values()
    bar = go.Figure(go.Bar(x=series.values, y=series.index, orientation="h"))
    bar.update_layout(xaxis={"title": "gain over damped baseline (%)"}, margin={"l": 10, "r": 10, "t": 30, "b": 10})
    st.plotly_chart(bar)

    st.subheader("Rain occurrence (a chance of at least 1 mm)")
    pooled_rain, per_city = RT.rain_tables(rain)
    st.dataframe(pooled_rain.round(3), hide_index=True)
    st.dataframe(per_city.round(3).rename(columns=lambda c: f"{c} day(s) ahead"))
    st.caption("Brier skill above 0 beats the usual chance of rain for that date. Skill falls close to zero by 3 days.")

    st.subheader("Can the models handle a city they never saw?")
    if loco is None:
        st.caption("Leave-cities-out results are not available.")
    else:
        t = RT.loco_summary(loco)
        t["target"] = t["target"].map(NAMES)
        st.dataframe(t.round(1), hide_index=True)
        st.warning("Held-out performance is much weaker than for cities the model trained on, and humidity and wind "
                   "do not transfer at all. So the ML predictions are **validated for the 10 training cities only**. "
                   "(These leave-cities-out numbers were measured on the validation years, before the final test.)")

    st.subheader("Honest limits")
    st.markdown("- ERA5 is a reanalysis (a model product), not station observations.\n"
                "- The models use only a city's own recent history, so skill fades after about 3 days; beyond that, "
                "weather-model forecasts (the Open-Meteo pages) are the right tool.\n"
                "- Design choices (city weighting, dropping location features) were made on validation years; the "
                "test years were used once.")
