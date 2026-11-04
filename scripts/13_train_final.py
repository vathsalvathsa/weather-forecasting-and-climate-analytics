"""Step 4.5: train the FINAL models and score them ONCE on the test split.

    python scripts/13_train_final.py                          # dry run: fit on train, score on val, saves nothing
    python scripts/13_train_final.py --final                  # fit on train+val, score ONCE on test, save to models/
    python scripts/13_train_final.py --final --no-location    # same without lat/lon/elevation

Recipe (chosen on validation data only): gradient boosting on climatology anomalies, cities
weighted by 1/variance; rain = probability model + gamma amount model.
Outputs with --final: models/final_models.joblib, models/model_card.json,
reports/final_test_results.csv, reports/final_rain_test_results.csv
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src import config
from src.data.final_split import final_training_split
from src.evaluation.evaluate import summarise
from src.features.build import ANOM, LOCATION_COLS, build_features, fit_climatologies, make_supervised
from src.features.rain import fit_wet_climatology, make_supervised_rain
from src.models.baselines import baseline_predictions
from src.models.weights import city_weights

TARGETS = ["tmean", "tmax", "tmin", "rh", "wind"]
HORIZONS = [1, 3]


def new_reg():
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=False, random_state=42)


def new_clf():
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
        l2_regularization=1.0, early_stopping=False, random_state=42)


def new_amount():
    return HistGradientBoostingRegressor(
        loss="gamma", max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100,
        l2_regularization=1.0, early_stopping=False, random_state=42)


def gain(skill_model, skill_ref):
    return 100 * (1 - (1 - skill_model) / (1 - skill_ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="train on train+val, score once on test, save models")
    ap.add_argument("--no-location", action="store_true", help="drop lat/lon/elevation features")
    args = ap.parse_args()

    df = pd.read_parquet(config.DATA_PROCESSED / "daily_all_cities.parquet")
    if args.final:
        df = final_training_split(df)
    eval_split = "test" if args.final else "val"
    print(f"Fitting on split == 'train'; scoring on split == '{eval_split}'"
          f"{' (FINAL: this is the one look at the test split)' if args.final else ' (dry run)'}")

    clims = fit_climatologies(df)
    feats = build_features(df, clims)
    wet_clim = fit_wet_climatology(df)

    reg_models, reg_cols, cont = {}, {}, []
    for target in TARGETS:
        raw = ANOM[target]
        for h in HORIZONS:
            m, cols = make_supervised(df, feats, clims, target, h)
            if args.no_location:
                cols = [c for c in cols if c not in LOCATION_COLS]
            tr, ev = m[m["split"] == "train"], m[m["split"] == eval_split]
            model = new_reg().fit(tr[cols], tr["y_anom"], sample_weight=city_weights(tr))
            base = baseline_predictions(df, raw, h)[["city", "date", "pred_climatology", "pred_damped_anomaly"]]
            ev = ev.merge(base, on=["city", "date"], how="left", validate="one_to_one").copy()
            ev["pred_final"] = ev["clim_tgt"].to_numpy() + model.predict(ev[cols])
            res = summarise(ev, ["pred_climatology", "pred_damped_anomaly", "pred_final"], "continuous")
            res.insert(0, "target", target)
            res.insert(1, "horizon", h)
            cont.append(res)
            reg_models[f"{target}_h{h}"], reg_cols[f"{target}_h{h}"] = model, cols
            print(f"trained {target} h={h}")

    rain_clf, rain_amt, rain = {}, {}, []
    for h in HORIZONS:
        m, cols = make_supervised_rain(df, feats, wet_clim, h)
        if args.no_location:
            cols = [c for c in cols if c not in LOCATION_COLS]
        tr, ev = m[m["split"] == "train"], m[m["split"] == eval_split].copy()
        clf = new_clf().fit(tr[cols], tr["y_true"])
        wet_tr = tr[tr["y_true"] == 1]
        amount = new_amount().fit(wet_tr[cols], wet_tr["amount_true"])
        ev["pred_climatology"] = ev["clim_wet_tgt"]
        ev["pred_final"] = clf.predict_proba(ev[cols])[:, 1]
        res = summarise(ev, ["pred_climatology", "pred_final"], "binary")
        res.insert(0, "horizon", h)
        rain.append(res)
        rain_clf[f"h{h}"], rain_amt[f"h{h}"], reg_cols[f"rain_h{h}"] = clf, amount, cols
        print(f"trained rain h={h}")

    cont, rain = pd.concat(cont, ignore_index=True), pd.concat(rain, ignore_index=True)
    pd.set_option("display.width", 200, "display.max_columns", 20)

    a = cont[cont["city"] == "ALL"].pivot(index=["target", "horizon"], columns="model", values="mse_skill_vs_clim")
    mae = cont[cont["city"] == "ALL"].pivot(index=["target", "horizon"], columns="model", values="mae")
    table = pd.DataFrame({"MAE_climatology": mae["climatology"], "MAE_damped": mae["damped_anomaly"],
                          "MAE_final": mae["final"],
                          "gain_over_damped_%": gain(a["final"], a["damped_anomaly"]),
                          "skill_vs_climatology": a["final"]})
    print(f"\n=== FINAL models, split = {eval_split}, all cities pooled ===")
    print(table.round(3).to_string())

    per_city = cont[(cont["target"] == "tmean") & (cont["horizon"] == 1) & (cont["city"] != "ALL")]
    s = per_city.pivot(index="city", columns="model", values="mse_skill_vs_clim")
    print("\n=== Per city, tmean h=1: % improvement over damped baseline ===")
    print(gain(s["final"], s["damped_anomaly"]).round(1).to_string())

    r_all = rain[rain["city"] == "ALL"].pivot(index="horizon", columns="model", values="bss_vs_clim")
    print("\n=== Rain occurrence: Brier skill vs climatology (pooled) ===")
    print(r_all[["final"]].round(3).to_string())
    r_city = rain[(rain["city"] != "ALL") & (rain["model"] == "final")].pivot(index="city", columns="horizon", values="bss_vs_clim")
    print("\n=== Rain occurrence: Brier skill vs climatology, per city ===")
    print(r_city.round(3).to_string())

    if not args.final:
        print("\nDry run only: nothing saved. Use --final when the recipe is settled.")
        return

    models_dir = config.ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    bundle = {"version": 1, "created": str(date.today()), "sklearn": sklearn.__version__,
              "location_features": not args.no_location, "trained_through": "2020-12-24",
              "reg_models": reg_models, "rain_clf": rain_clf, "rain_amount": rain_amt,
              "columns": reg_cols, "climatologies": clims, "wet_climatology": wet_clim}
    joblib.dump(bundle, models_dir / "final_models.joblib", compress=3)
    card = {"created": bundle["created"], "sklearn": sklearn.__version__,
            "data": "ERA5 reanalysis via Open-Meteo (models=era5), daily, 10 cities, 1991-2026",
            "trained_on": "1991-01-01 to 2020-12-24 (train + validation)",
            "scored_once_on": "2021-01-01 to 2026-10 (test)",
            "location_features": bundle["location_features"],
            "pooled_test_results": json.loads(table.round(4).reset_index().to_json(orient="records")),
            "limitations": ["ERA5 is a reanalysis, not station data",
                            "local history only: skill fades after about 3 days",
                            "inputs at prediction time must come from the same source as training"]}
    (models_dir / "model_card.json").write_text(json.dumps(card, indent=2))
    config.REPORTS.mkdir(exist_ok=True)
    cont.to_csv(config.REPORTS / "final_test_results.csv", index=False)
    rain.to_csv(config.REPORTS / "final_rain_test_results.csv", index=False)
    print("\nSaved -> models/final_models.joblib, models/model_card.json, reports/final_test_results.csv")


if __name__ == "__main__":
    main()
