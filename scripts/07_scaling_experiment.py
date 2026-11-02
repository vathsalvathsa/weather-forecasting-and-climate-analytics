"""Step 4.3a: why do Singapore / Nairobi / Mumbai lose to the baseline?

Compares three ways of training gradient boosting (same features, train split only):
  hgb       one global model                         (what we ran before)
  hgb_w     one global model, cities weighted 1/variance
  hgb_city  a separate model per city                (diagnostic ceiling: cannot serve unseen cities)

    python scripts/07_scaling_experiment.py

The test split is locked behind --final, as in the other scripts.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src import config
from src.evaluation.evaluate import summarise
from src.features.build import ANOM, build_features, fit_climatologies, make_supervised
from src.models.baselines import baseline_predictions
from src.models.weights import city_weights

TARGETS = ["tmean", "tmax", "tmin", "rh", "wind"]
HORIZONS = [1, 3]
VARIANTS = ["hgb", "hgb_w", "hgb_city"]


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
    for target in TARGETS:
        raw = ANOM[target]
        for h in HORIZONS:
            t0 = time.time()
            m, cols = make_supervised(df, feats, clims, target, h)
            tr = m[m["split"] == "train"]
            ev = m[m["split"] == args.split]
            base = baseline_predictions(df, raw, h)[["city", "date", "pred_climatology", "pred_damped_anomaly"]]
            ev = ev.merge(base, on=["city", "date"], how="left", validate="one_to_one").copy()
            clim = ev["clim_tgt"].to_numpy()

            ev["pred_hgb"] = clim + new_hgb().fit(tr[cols], tr["y_anom"]).predict(ev[cols])
            ev["pred_hgb_w"] = clim + new_hgb().fit(tr[cols], tr["y_anom"],
                                                    sample_weight=city_weights(tr)).predict(ev[cols])
            ev["pred_hgb_city"] = float("nan")
            for city in ev["city"].unique():
                tr_c = tr[tr["city"] == city]
                idx = ev["city"] == city
                ev.loc[idx, "pred_hgb_city"] = (ev.loc[idx, "clim_tgt"].to_numpy()
                                                + new_hgb().fit(tr_c[cols], tr_c["y_anom"]).predict(ev.loc[idx, cols]))

            res = summarise(ev, ["pred_climatology", "pred_damped_anomaly"] + [f"pred_{v}" for v in VARIANTS], "continuous")
            res.insert(0, "target", target)
            res.insert(1, "horizon", h)
            results.append(res)
            print(f"done {target} h={h} ({time.time() - t0:.0f}s)")

    out = pd.concat(results, ignore_index=True)
    config.REPORTS.mkdir(exist_ok=True)
    out.to_csv(config.REPORTS / f"scaling_experiment_{args.split}.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    a = out[out["city"] == "ALL"].pivot(index=["target", "horizon"], columns="model", values="mse_skill_vs_clim")
    print(f"\n=== % MSE improvement over damped baseline, all cities pooled, split = {args.split} ===")
    print(gain_over_damped(a, VARIANTS).round(1).to_string())

    for target, h in [("tmean", 1), ("tmean", 3)]:
        c = out[(out["target"] == target) & (out["horizon"] == h) & (out["city"] != "ALL")]
        s = c.pivot(index="city", columns="model", values="mse_skill_vs_clim")
        print(f"\n=== Per city: % improvement over damped baseline, {target} h={h} ===")
        print(gain_over_damped(s, VARIANTS).round(1).to_string())
    print("\nSaved ->", config.REPORTS / f"scaling_experiment_{args.split}.csv")


if __name__ == "__main__":
    main()
