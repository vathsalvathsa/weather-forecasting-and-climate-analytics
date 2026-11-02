"""Supervised tables for rain forecasts.

Stage 1 target: wet day at t+h (precipitation >= 1 mm)   -> column y_true (0/1)
Stage 2 target: precipitation amount at t+h               -> column amount_true (mm)
The wet-day climatology is fitted on TRAIN rows only and is also a feature.
"""
import pandas as pd

from src.models.baselines import Climatology, make_samples

WET_MM = 1.0


def _with_wet(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(wet=(df["precipitation_sum"] >= WET_MM).astype(float))


def fit_wet_climatology(df: pd.DataFrame) -> Climatology:
    d = _with_wet(df)
    return Climatology().fit(d[d["split"] == "train"], "wet")


def make_supervised_rain(df: pd.DataFrame, feats: pd.DataFrame, wet_clim: Climatology, horizon: int):
    d = _with_wet(df)
    s = make_samples(d, "wet", horizon)                                  # y_true = wet at t+h
    amt = make_samples(d, "precipitation_sum", horizon)[["city", "date", "y_true"]]
    amt = amt.rename(columns={"y_true": "amount_true"})
    s = s.merge(amt, on=["city", "date"], validate="one_to_one")
    s["clim_wet_tgt"] = wet_clim.predict(s["city"], s["target_date"])
    m = s.merge(feats, on=["city", "date"], how="left", validate="one_to_one")
    cols = [c for c in feats.columns if c not in ("city", "date")] + ["clim_wet_tgt"]
    m = m.dropna(subset=cols).reset_index(drop=True)
    return m, cols
