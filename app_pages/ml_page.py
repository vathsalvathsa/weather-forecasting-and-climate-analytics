"""ML Predictions page: replays the saved models on past dates (a backtest), never a live forecast."""
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.inference.backtest import error_summary, predict_continuous, predict_rain, rain_summary

LABELS = {"tmean": ("Mean temperature", "°C"), "tmax": ("Maximum temperature", "°C"),
          "tmin": ("Minimum temperature", "°C"), "rh": ("Relative humidity", "%"),
          "wind": ("Maximum wind speed", "km/h"), "rain": ("Rain (will it rain, and how much?)", "mm")}


@st.cache_resource(show_spinner=False)
def load_bundle():
    path = config.ROOT / "models" / "final_models.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_recent():
    path = config.ROOT / "app_data" / "daily_recent.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(show_spinner="Replaying the model...")
def run_continuous(_bundle, city, target, horizon):
    df = load_recent()
    return predict_continuous(_bundle, df[df["city"] == city], target, horizon)


@st.cache_data(show_spinner="Replaying the model...")
def run_rain(_bundle, city, horizon):
    df = load_recent()
    return predict_rain(_bundle, df[df["city"] == city], horizon)


def render():
    st.title("ML Predictions (backtest)")
    bundle, recent = load_bundle(), load_recent()
    if bundle is None or recent is None:
        st.warning("The trained models or their data files are not available in this deployment.")
        return
    st.info("This page replays the project's own trained models on **past** days, from 2021-01-01 onward, "
            "after their training period ended (2020-12-24). It is a backtest on data the models never saw, "
            "**not a live forecast**: the training data (ERA5) arrives days late, so live inputs from the same "
            "source do not exist yet. The models are validated for the cities listed here only.")
    c1, c2, c3 = st.columns(3)
    city = c1.selectbox("City (training cities)", sorted(recent["city"].unique()))
    target = c2.selectbox("What to predict", list(LABELS), format_func=lambda k: LABELS[k][0])
    horizon = c3.radio("Days ahead", [1, 3], horizontal=True)
    if target == "rain":
        render_rain(bundle, city, horizon)
    else:
        render_continuous(bundle, city, target, horizon)


def _window(res, day, days=30):
    return res[(res["date"] >= day - pd.Timedelta(days=days)) & (res["date"] <= day + pd.Timedelta(days=days))]


def render_continuous(bundle, city, target, horizon):
    name, unit = LABELS[target]
    res = run_continuous(bundle, city, target, horizon)
    if res.empty:
        st.warning("No backtest rows are available for this choice.")
        return
    first, last = res["date"].min().date(), res["date"].max().date()
    day = pd.Timestamp(st.date_input("Issue date (the forecast is made at the end of this day)",
                                     value=last, min_value=first, max_value=last))
    win = _window(res, day)

    st.subheader(f"{name} in {city}, {horizon} day(s) ahead: model vs baselines")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=win["target_date"], y=win["actual"], name="Actual (ERA5)", mode="lines"))
    fig.add_trace(go.Scatter(x=win["target_date"], y=win["model"], name="ML model", mode="lines"))
    fig.add_trace(go.Scatter(x=win["target_date"], y=win["climatology"], name="Climatology (usual value)",
                             mode="lines", line={"dash": "dot"}))
    sel = res[res["date"] == day]
    if not sel.empty:
        fig.add_trace(go.Scatter(x=sel["target_date"], y=sel["model"], name="Selected day", mode="markers",
                                 marker={"size": 11}))
    fig.update_layout(yaxis={"title": unit}, legend={"orientation": "h"}, margin={"l": 10, "r": 10, "t": 30, "b": 10})
    st.plotly_chart(fig)

    if not sel.empty:
        r = sel.iloc[0]
        c = st.columns(4)
        c[0].metric("Predicted by the model", f"{r['model']:.1f} {unit}")
        c[1].metric("Actual", f"{r['actual']:.1f} {unit}")
        c[2].metric("Error", f"{r['model'] - r['actual']:+.1f} {unit}")
        c[3].metric("Climatology says", f"{r['climatology']:.1f} {unit}")

    st.subheader("How good is it? (whole backtest period)")
    summary = error_summary(res)
    st.dataframe(summary.round(3), hide_index=True)
    st.caption("MAE and RMSE are in the units above (lower is better). Skill vs climatology: 1 is perfect, 0 means "
               "no better than the usual value for that date. Persistence means 'tomorrow equals today'.")
    st.caption("Windows are not cherry-picked: the table covers every day from 2021-01-01 to the last available day.")


def render_rain(bundle, city, horizon):
    res = run_rain(bundle, city, horizon)
    if res.empty:
        st.warning("No backtest rows are available for this choice.")
        return
    first, last = res["date"].min().date(), res["date"].max().date()
    day = pd.Timestamp(st.date_input("Issue date (the forecast is made at the end of this day)",
                                     value=last, min_value=first, max_value=last))
    win = _window(res, day)
    st.subheader(f"Rain in {city}, {horizon} day(s) ahead")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=win["target_date"], y=win["amount_actual"], name="Actual rain (mm)", opacity=0.5))
    fig.add_trace(go.Scatter(x=win["target_date"], y=win["p_model"] * 100, name="Model: chance of rain (%)",
                             yaxis="y2", mode="lines"))
    fig.add_trace(go.Scatter(x=win["target_date"], y=win["p_climatology"] * 100, name="Climatology chance (%)",
                             yaxis="y2", mode="lines", line={"dash": "dot"}))
    fig.update_layout(yaxis={"title": "mm"}, yaxis2={"title": "%", "overlaying": "y", "side": "right", "range": [0, 100]},
                      legend={"orientation": "h"}, margin={"l": 10, "r": 10, "t": 30, "b": 10})
    st.plotly_chart(fig)
    sel = res[res["date"] == day]
    if not sel.empty:
        r = sel.iloc[0]
        c = st.columns(4)
        c[0].metric("Model: chance of >= 1 mm", f"{r['p_model'] * 100:.0f} %")
        c[1].metric("Climatology chance", f"{r['p_climatology'] * 100:.0f} %")
        c[2].metric("Expected rain if it rains", f"{r['amount_if_wet']:.1f} mm")
        c[3].metric("Actual rain", f"{r['amount_actual']:.1f} mm")
    s = rain_summary(res)
    st.subheader("How good is it? (whole backtest period)")
    st.dataframe(pd.DataFrame([{"Brier score, model": s["brier_model"], "Brier score, climatology": s["brier_climatology"],
                                "Brier skill vs climatology": s["brier_skill"], "wet days": s["wet_days_actual"],
                                "days": s["days"]}]).round(3), hide_index=True)
    st.caption("Brier score measures probability forecasts (lower is better). Skill above 0 means better than the "
               "usual chance of rain for that date. Beyond about 1 to 2 days, rain is not predictable from a "
               "city's own recent history: expect skill near zero at 3 days.")
