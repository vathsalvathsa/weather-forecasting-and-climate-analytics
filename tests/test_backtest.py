import numpy as np
import pandas as pd
import pytest

from src.features import build as F
from src.features import rain as R
from src.inference import backtest as BT
from test_features import make_full_df


def make_bundle(df):
    pytest.importorskip("sklearn")
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    clims = F.fit_climatologies(df)
    feats = F.build_features(df, clims)
    m, cols = F.make_supervised(df, feats, clims, "tmean", 1)
    tr = m[m["split"] == "train"]
    reg = HistGradientBoostingRegressor(max_iter=30, random_state=0).fit(tr[cols], tr["y_anom"])
    wet_clim = R.fit_wet_climatology(df)
    mr, rcols = R.make_supervised_rain(df, feats, wet_clim, 1)
    trr = mr[mr["split"] == "train"]
    clf = HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(trr[rcols], trr["y_true"])
    wet = trr[trr["y_true"] == 1]
    amt = HistGradientBoostingRegressor(loss="gamma", max_iter=30, random_state=0).fit(wet[rcols], wet["amount_true"])
    return {"climatologies": clims, "wet_climatology": wet_clim, "reg_models": {"tmean_h1": reg},
            "columns": {"tmean_h1": cols, "rain_h1": rcols}, "rain_clf": {"h1": clf}, "rain_amount": {"h1": amt}}


def test_backtest_is_aligned_out_of_sample_and_finite():
    df = make_full_df(n_cities=2)
    bundle = make_bundle(df)
    city_df = df[df["city"] == "C0"]
    out = BT.predict_continuous(bundle, city_df, "tmean", 1)
    assert (out["date"] >= BT.TEST_START).all() and len(out) > 500
    assert np.isfinite(out[["actual", "model", "climatology", "persistence"]].to_numpy()).all()
    lookup = city_df.set_index("date")["temperature_2m_mean"]
    assert np.allclose(out["actual"], lookup.reindex(out["target_date"]).to_numpy())   # truth is the day after
    assert np.allclose(out["persistence"], lookup.reindex(out["date"]).to_numpy())     # persistence = today


def test_backtest_predictions_never_use_the_future():
    df = make_full_df(n_cities=2)
    bundle = make_bundle(df)
    city_df = df[df["city"] == "C1"]
    a = BT.predict_continuous(bundle, city_df, "tmean", 1)
    corrupted = city_df.copy()
    future = corrupted["date"] >= "2022-06-30"
    for col in list(F.ANOM.values()) + ["precipitation_sum"]:
        corrupted.loc[future, col] += 50.0
    b = BT.predict_continuous(bundle, corrupted, "tmean", 1)
    early = a["date"] < pd.Timestamp("2022-06-29")
    assert early.sum() > 100
    assert np.allclose(a.loc[early, "model"].to_numpy(), b.loc[early, "model"].to_numpy())


def test_rain_backtest_outputs_are_consistent():
    df = make_full_df(n_cities=2)
    bundle = make_bundle(df)
    out = BT.predict_rain(bundle, df[df["city"] == "C0"], 1)
    assert out["p_model"].between(0, 1).all() and out["p_climatology"].between(0, 1).all()
    assert np.allclose(out["expected_mm"], out["p_model"] * out["amount_if_wet"])
    assert ((out["amount_actual"] >= R.WET_MM).astype(float) == out["wet_actual"]).all()


def test_error_and_rain_summaries():
    res = pd.DataFrame({"actual": [1.0, 2.0, 3.0], "model": [1.0, 2.0, 3.0],
                        "climatology": [2.0, 2.0, 2.0], "persistence": [0.0, 1.0, 2.0]})
    e = BT.error_summary(res).set_index("forecast")
    assert e.loc["model", "MAE"] == 0 and e.loc["model", "skill_vs_climatology"] == 1.0
    assert e.loc["climatology", "skill_vs_climatology"] == 0.0
    assert e.loc["persistence", "MAE"] == 1.0
    r = pd.DataFrame({"wet_actual": [1.0, 0.0, 1.0, 0.0], "p_model": [1.0, 0.0, 1.0, 0.0],
                      "p_climatology": [0.5] * 4})
    s = BT.rain_summary(r)
    assert s["brier_model"] == 0 and s["brier_skill"] == 1.0 and s["wet_days_actual"] == 2
