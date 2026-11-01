"""Step 4.2: first ML models (Ridge, gradient boosting) vs the baselines.

    python scripts/05_train_first_models.py                 # train on train, score on val
    python scripts/05_train_first_models.py --no-location   # drop lat/lon/elevation

Models are fitted on the TRAIN split only. The test split is locked behind --final.
No hyperparameter tuning yet: these are deliberately plain settings.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.evaluation.evaluate import summarise
from src.features.build import ANOM, LOCATION_COLS, build_features, fit_climatologies, make_supervised
from src.models.baselines import baseline_predictions

TARGETS = ["tmean", "tmax", "tmin", "rh", "wind"]
HORIZONS = [1, 3]


def make_models():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "hgb": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
            l2_regularization=1.0, early_stopping=False, random_state=42),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--final", action="store_true", help="required to touch the test split")
    ap.add_argument("--no-location", action="store_true", help="drop lat/lon/elevation features")
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
            if args.no_location:
                cols = [c for c in cols if c not in LOCATION_COLS]
            tr = m[m["split"] == "train"]
            ev = m[m["split"] == args.split]
            base = baseline_predictions(df, raw, h)[["city", "date", "pred_climatology", "pred_damped_anomaly"]]
            ev = ev.merge(base, on=["city", "date"], how="left", validate="one_to_one").copy()

            preds = ["pred_climatology", "pred_damped_anomaly"]
            for name, model in make_models().items():
                model.fit(tr[cols], tr["y_anom"])
                ev[f"pred_{name}"] = ev["clim_tgt"].to_numpy() + model.predict(ev[cols])
                preds.append(f"pred_{name}")

            res = summarise(ev, preds, "continuous")
            res.insert(0, "target", target)
            res.insert(1, "horizon", h)
            results.append(res)
            print(f"done {target} h={h}: {len(tr):,} train rows, {len(cols)} features, {time.time() - t0:.0f}s")

    out = pd.concat(results, ignore_index=True)
    tag = f"{args.split}{'_nolocation' if args.no_location else ''}"
    config.REPORTS.mkdir(exist_ok=True)
    out.to_csv(config.REPORTS / f"models_v1_{tag}.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    a = out[out["city"] == "ALL"]
    print(f"\n=== MAE, all cities pooled, split = {args.split} ===")
    print(a.pivot(index=["target", "horizon"], columns="model", values="mae")
          [["climatology", "damped_anomaly", "ridge", "hgb"]].round(2).to_string())

    sk = a.pivot(index=["target", "horizon"], columns="model", values="mse_skill_vs_clim")
    gain = pd.DataFrame({m: 100 * (1 - (1 - sk[m]) / (1 - sk["damped_anomaly"])) for m in ["ridge", "hgb"]})
    print("\n=== MSE improvement over the damped-anomaly baseline, % (positive = ML better) ===")
    print(gain.round(1).to_string())

    for target, h in [("tmean", 1), ("tmean", 3)]:
        c = out[(out["target"] == target) & (out["horizon"] == h) & (out["city"] != "ALL")]
        sk_c = c.pivot(index="city", columns="model", values="mse_skill_vs_clim")
        g_c = pd.DataFrame({m: 100 * (1 - (1 - sk_c[m]) / (1 - sk_c["damped_anomaly"])) for m in ["ridge", "hgb"]})
        print(f"\n=== Per city: % MSE improvement over damped baseline, {target} h={h} ===")
        print(g_c.round(1).to_string())
    print("\nSaved ->", config.REPORTS / f"models_v1_{tag}.csv")


if __name__ == "__main__":
    main()
