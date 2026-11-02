import numpy as np
import pandas as pd
import pytest

from src.evaluation import calibration as C
from src.features import build as F
from src.features import rain as R
from test_features import make_full_df


def _setup(df, h=1):
    clims = F.fit_climatologies(df)
    feats = F.build_features(df, clims)
    wet_clim = R.fit_wet_climatology(df)
    return R.make_supervised_rain(df, feats, wet_clim, h)


def test_wet_labels_and_amounts_are_consistent():
    df = make_full_df()
    m, cols = _setup(df, h=2)
    assert set(m["y_true"].unique()) <= {0.0, 1.0}
    assert (m.loc[m["y_true"] == 1, "amount_true"] >= R.WET_MM).all()
    assert (m.loc[m["y_true"] == 0, "amount_true"] < R.WET_MM).all()
    assert not m[cols].isna().any().any()
    assert (m["split"] != "gap").all()
    assert m["clim_wet_tgt"].between(0, 1).all()


def test_target_is_the_day_h_ahead():
    df = make_full_df(n_cities=1)
    m, _ = _setup(df, h=2)
    lookup = df.set_index("date")["precipitation_sum"]
    expected = lookup.reindex(m["target_date"]).to_numpy()
    assert np.allclose(m["amount_true"].to_numpy(), expected)


def test_wet_climatology_ignores_validation_and_test_rain():
    df = make_full_df()
    a = R.fit_wet_climatology(df).table_.to_numpy()
    df2 = df.copy()
    held_out = df2["split"].isin(["val", "test"])
    df2.loc[held_out, "precipitation_sum"] = 99.0
    b = R.fit_wet_climatology(df2).table_.to_numpy()
    assert np.allclose(a, b)


def test_reliability_for_calibrated_and_miscalibrated_forecasts():
    rng = np.random.default_rng(0)
    p = rng.random(50000)
    y = (rng.random(50000) < p).astype(float)                 # perfectly calibrated
    assert C.expected_calibration_error(y, p) < 0.02
    y2 = (rng.random(50000) < 0.3).astype(float)
    assert C.expected_calibration_error(y2, np.full(50000, 0.9)) == pytest.approx(0.6, abs=0.02)
    t = C.reliability_table(y, p)
    assert t["n"].sum() == 50000


def test_classifier_beats_climatology_when_rain_depends_on_humidity():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    df = make_full_df()
    rng = np.random.default_rng(1)
    for c in df["city"].unique():                              # tomorrow's rain follows today's humidity
        idx = df.index[df["city"] == c]
        rh = df.loc[idx, "relative_humidity_2m_mean"].to_numpy()
        z = (rh - rh.mean()) / rh.std()
        p_next = 1 / (1 + np.exp(-(2.0 * z - 1.0)))
        wet_next = rng.random(len(idx) - 1) < p_next[:-1]
        amount = rng.gamma(2.0, 6.0, len(idx) - 1) + 1.0
        precip = np.zeros(len(idx))
        precip[1:] = np.where(wet_next, amount, 0.0)
        df.loc[idx, "precipitation_sum"] = precip
    m, cols = _setup(df, h=1)
    tr, ev = m[m["split"] == "train"], m[m["split"] == "val"]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(tr[cols], tr["y_true"])
    p = model.predict_proba(ev[cols])[:, 1]
    brier_model = np.mean((ev["y_true"] - p) ** 2)
    brier_clim = np.mean((ev["y_true"] - ev["clim_wet_tgt"]) ** 2)
    assert brier_model < 0.9 * brier_clim
