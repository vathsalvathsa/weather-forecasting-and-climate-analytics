"""Step 4.4: can the model forecast a city it has NEVER been trained on?

For each of the 10 cities: train gradient boosting on the other 9 cities (train split only)
and score the held-out city on the validation split. Compared with a model that did see
the city. Each city keeps its OWN climatology (an app user's new city would get one from
its history); only the learned model is blind to it. Run with and without lat/lon/elevation.

    python scripts/12_leave_cities_out.py

Rule fixed in advance: keep the location features only if they raise the average
held-out-city gain by at least 1 percentage point.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src import config
from src.evaluation.evaluate import summarise
from src.features.build import ANOM, LOCATION_COLS, build_features, fit_climatologies, make_supervised
from src.models.baselines import baseline_predictions
from src.models.weights import city_weights

TARGETS = ["tmean", "tmax", "tmin", "rh", "wind"]
JOBS = [(t, 1) for t in TARGETS] + [("tmean", 3)]
VARIANTS = ["insample", "loco_loc", "loco_noloc"]


def new_hgb():
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=False, random_state=42)


def gain_over_damped(skill: pd.DataFrame, models: list) -> pd.DataFrame:
    return pd.DataFrame({m: 100 * (1 - (1 - skill[m]) / (1 - skill["damped_anomaly"])) for m in models})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--final", action="store_true", help="required to touch the test split")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        sys.exit("The test split is reserved for the final evaluation. Use --split val.")

    df = pd.read_parquet(config.DATA_PROCESSED / "daily_all_cities.parquet")
    clims = fit_climatologies(df)
    feats = build_features(df, clims)

    results = []
    for target, h in JOBS:
        t0 = time.time()
        raw = ANOM[target]
        m, cols = make_supervised(df, feats, clims, target, h)
        cols_nl = [c for c in cols if c not in LOCATION_COLS]
        tr = m[m["split"] == "train"]
        ev = m[m["split"] == args.split]
        base = baseline_predictions(df, raw, h)[["city", "date", "pred_climatology", "pred_damped_anomaly"]]
        ev = ev.merge(base, on=["city", "date"], how="left", validate="one_to_one").copy()
        clim = ev["clim_tgt"].to_numpy()

        full = new_hgb().fit(tr[cols], tr["y_anom"], sample_weight=city_weights(tr))
        ev["pred_insample"] = clim + full.predict(ev[cols])
        ev["pred_loco_loc"] = np.nan
        ev["pred_loco_noloc"] = np.nan
        for city in ev["city"].unique():
            tr_c = tr[tr["city"] != city]
            idx = (ev["city"] == city).to_numpy()
            w = city_weights(tr_c)
            ev.loc[idx, "pred_loco_loc"] = clim[idx] + new_hgb().fit(
                tr_c[cols], tr_c["y_anom"], sample_weight=w).predict(ev.loc[idx, cols])
            ev.loc[idx, "pred_loco_noloc"] = clim[idx] + new_hgb().fit(
                tr_c[cols_nl], tr_c["y_anom"], sample_weight=w).predict(ev.loc[idx, cols_nl])

        res = summarise(ev, ["pred_climatology", "pred_damped_anomaly"] + [f"pred_{v}" for v in VARIANTS], "continuous")
        res.insert(0, "target", target)
        res.insert(1, "horizon", h)
        results.append(res)
        print(f"done {target} h={h} ({time.time() - t0:.0f}s)")

    out = pd.concat(results, ignore_index=True)
    config.REPORTS.mkdir(exist_ok=True)
    out.to_csv(config.REPORTS / f"leave_cities_out_{args.split}.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    rows = []
    for (target, h), g in out[out["city"] != "ALL"].groupby(["target", "horizon"]):
        gains = gain_over_damped(g.pivot(index="city", columns="model", values="mse_skill_vs_clim"), VARIANTS)
        row = {"target": target, "horizon": h}
        for v in VARIANTS:
            row[f"{v}_mean_gain"] = gains[v].mean()
            row[f"{v}_cities_below_0"] = int((gains[v] < 0).sum())
        rows.append(row)
    print(f"\n=== Average over the 10 cities of % MSE improvement over each city's damped baseline, split = {args.split} ===")
    print("(insample = model saw the city; loco = city held out; loc/noloc = with/without lat, lon, elevation)")
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    c = out[(out["target"] == "tmean") & (out["horizon"] == 1) & (out["city"] != "ALL")]
    print("\n=== Per city, tmean h=1: % improvement over damped baseline ===")
    print(gain_over_damped(c.pivot(index="city", columns="model", values="mse_skill_vs_clim"), VARIANTS).round(1).to_string())
    print("\nSaved ->", config.REPORTS / f"leave_cities_out_{args.split}.csv")


if __name__ == "__main__":
    main()
