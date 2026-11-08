"""Replay the saved FINAL models on past dates (the backtest explorer in the app).

Only dates from 2021-01-01 onward are returned: the models were trained through 2020-12-24,
so these days were never seen in training. Features for an issue day use only data up to
that day (covered by tests).
"""
import numpy as np
import pandas as pd

from src.features.build import ANOM, build_features
from src.features.rain import WET_MM

TEST_START = pd.Timestamp("2021-01-01")


def _targets(city_df, col, horizon):
    d = city_df.sort_values("date").reset_index(drop=True)
    return pd.DataFrame({"date": d["date"], "target_date": d["date"].shift(-horizon),
                         "y_now": d[col], "y_true": d[col].shift(-horizon)}).dropna(subset=["y_true"])


def predict_continuous(bundle, city_df, target: str, horizon: int) -> pd.DataFrame:
    """Model, climatology and persistence forecasts for one city, one target, one horizon."""
    city = city_df["city"].iloc[0]
    raw, clims, key = ANOM[target], bundle["climatologies"], f"{target}_h{horizon}"
    feats = build_features(city_df, clims).drop(columns=["city"])
    s = _targets(city_df, raw, horizon)
    s["clim_tgt"] = clims[raw].predict(pd.Series([city] * len(s)), s["target_date"])
    m = s.merge(feats, on="date", how="left", validate="one_to_one")
    cols = bundle["columns"][key]
    m = m.dropna(subset=cols)
    m = m[m["date"] >= TEST_START].reset_index(drop=True)
    anomaly = bundle["reg_models"][key].predict(m[cols])
    return pd.DataFrame({"date": m["date"], "target_date": m["target_date"], "actual": m["y_true"],
                         "model": m["clim_tgt"] + anomaly, "climatology": m["clim_tgt"],
                         "persistence": m["y_now"]})


def predict_rain(bundle, city_df, horizon: int) -> pd.DataFrame:
    """Probability of a wet day (>= 1 mm) and the expected amount, with the climatological chance."""
    city = city_df["city"].iloc[0]
    clims, wet_clim = bundle["climatologies"], bundle["wet_climatology"]
    feats = build_features(city_df, clims).drop(columns=["city"])
    d = city_df.sort_values("date").reset_index(drop=True)
    s = pd.DataFrame({"date": d["date"], "target_date": d["date"].shift(-horizon),
                      "amount_actual": d["precipitation_sum"].shift(-horizon)}).dropna(subset=["amount_actual"])
    s["clim_wet_tgt"] = wet_clim.predict(pd.Series([city] * len(s)), s["target_date"])
    m = s.merge(feats, on="date", how="left", validate="one_to_one")
    cols = bundle["columns"][f"rain_h{horizon}"]
    m = m.dropna(subset=cols)
    m = m[m["date"] >= TEST_START].reset_index(drop=True)
    p = bundle["rain_clf"][f"h{horizon}"].predict_proba(m[cols])[:, 1]
    amount = bundle["rain_amount"][f"h{horizon}"].predict(m[cols])
    return pd.DataFrame({"date": m["date"], "target_date": m["target_date"],
                         "wet_actual": (m["amount_actual"] >= WET_MM).astype(float),
                         "amount_actual": m["amount_actual"], "p_model": p, "p_climatology": m["clim_wet_tgt"],
                         "amount_if_wet": amount, "expected_mm": p * amount})


def error_summary(res: pd.DataFrame) -> pd.DataFrame:
    """MAE and RMSE of model / climatology / persistence, plus MSE skill vs climatology."""
    y = res["actual"].to_numpy(float)
    rows = []
    mse_clim = np.mean((y - res["climatology"].to_numpy(float)) ** 2)
    for name in ("model", "climatology", "persistence"):
        err = y - res[name].to_numpy(float)
        mse = np.mean(err ** 2)
        rows.append({"forecast": name, "MAE": np.mean(np.abs(err)), "RMSE": np.sqrt(mse),
                     "skill_vs_climatology": 1 - mse / mse_clim if mse_clim > 0 else np.nan})
    return pd.DataFrame(rows)


def rain_summary(res: pd.DataFrame) -> dict:
    y = res["wet_actual"].to_numpy(float)
    b_model = float(np.mean((res["p_model"].to_numpy(float) - y) ** 2))
    b_clim = float(np.mean((res["p_climatology"].to_numpy(float) - y) ** 2))
    return {"brier_model": b_model, "brier_climatology": b_clim,
            "brier_skill": 1 - b_model / b_clim if b_clim > 0 else float("nan"),
            "wet_days_actual": int(y.sum()), "days": int(len(y))}
