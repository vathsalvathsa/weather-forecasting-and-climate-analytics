"""Step 4.3b: rain models. Stage 1: will it rain (>= 1 mm)? Stage 2: how much, if it does?

    python scripts/09_rain_models.py

Fitted on the TRAIN split only; the test split is locked behind --final.
Probabilities are judged by Brier skill vs the wet-day climatology (per city, not only
pooled: pooled ROC-AUC is inflated because climatology already separates wet from dry cities).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.evaluation.calibration import expected_calibration_error, reliability_table
from src.evaluation.evaluate import summarise
from src.features.build import build_features, fit_climatologies
from src.features.rain import fit_wet_climatology, make_supervised_rain
from src.models.baselines import baseline_predictions

HORIZONS = [1, 3]


def make_logit():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))


def make_clf():
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
        l2_regularization=1.0, early_stopping=False, random_state=42)


def make_amount():
    return HistGradientBoostingRegressor(
        loss="gamma", max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100,
        l2_regularization=1.0, early_stopping=False, random_state=42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--final", action="store_true", help="required to touch the test split")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        sys.exit("The test split is reserved for the final evaluation. Use --split val.")

    df = pd.read_parquet(config.DATA_PROCESSED / "daily_all_cities.parquet")
    feats = build_features(df, fit_climatologies(df))
    wet_clim = fit_wet_climatology(df)

    occ, amt_res, reliab = [], [], {}
    for h in HORIZONS:
        t0 = time.time()
        m, cols = make_supervised_rain(df, feats, wet_clim, h)
        tr = m[m["split"] == "train"]
        ev = m[m["split"] == args.split].copy()

        # Stage 1: probability of a wet day
        ev["pred_climatology"] = ev["clim_wet_tgt"]
        ev["pred_logit"] = make_logit().fit(tr[cols], tr["y_true"]).predict_proba(ev[cols])[:, 1]
        ev["pred_hgb_clf"] = make_clf().fit(tr[cols], tr["y_true"]).predict_proba(ev[cols])[:, 1]
        res = summarise(ev, ["pred_climatology", "pred_logit", "pred_hgb_clf"], "binary")
        res.insert(0, "horizon", h)
        occ.append(res)
        reliab[h] = reliability_table(ev["y_true"], ev["pred_hgb_clf"])
        ece = expected_calibration_error(ev["y_true"], ev["pred_hgb_clf"])

        # Stage 2: amount on wet days (gamma loss), then expected precipitation = P(wet) * E[amount | wet]
        wet_tr = tr[tr["y_true"] == 1]
        amount = make_amount().fit(wet_tr[cols], wet_tr["amount_true"])
        ev["pred_expected_precip"] = ev["pred_hgb_clf"] * amount.predict(ev[cols])
        base = baseline_predictions(df, "precipitation_sum", h)[["city", "date", "pred_climatology", "pred_damped_anomaly"]]
        base = base.rename(columns={"pred_climatology": "pred_precip_clim", "pred_damped_anomaly": "pred_precip_damped"})
        ev = ev.merge(base, on=["city", "date"], how="left", validate="one_to_one")
        ev_amt = ev.assign(y_true=ev["amount_true"])
        res2 = summarise(ev_amt, ["pred_precip_clim", "pred_precip_damped", "pred_expected_precip"],
                         "continuous", ref_col="pred_precip_clim")
        res2.insert(0, "horizon", h)
        amt_res.append(res2)
        print(f"done h={h}: {len(tr):,} train rows, {len(cols)} features, ECE(hgb)={ece:.3f}, {time.time() - t0:.0f}s")

    occ, amt_res = pd.concat(occ, ignore_index=True), pd.concat(amt_res, ignore_index=True)
    config.REPORTS.mkdir(exist_ok=True)
    occ.to_csv(config.REPORTS / f"rain_occurrence_{args.split}.csv", index=False)
    amt_res.to_csv(config.REPORTS / f"rain_amount_{args.split}.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    a = occ[occ["city"] == "ALL"]
    for metric, title in [("brier", "Brier score (lower is better)"),
                          ("bss_vs_clim", "Brier skill vs climatology (higher is better)"),
                          ("roc_auc", "ROC-AUC, pooled (inflated by between-city differences)"),
                          ("pr_auc", "PR-AUC, pooled")]:
        print(f"\n=== Rain occurrence: {title}, split = {args.split} ===")
        print(a.pivot(index="horizon", columns="model", values=metric)[["climatology", "logit", "hgb_clf"]].round(4).to_string())

    pc = occ[(occ["city"] != "ALL") & (occ["model"] == "hgb_clf")]
    print("\n=== Per city, gradient boosting: Brier skill vs climatology ===")
    print(pc.pivot(index="city", columns="horizon", values="bss_vs_clim").round(3).to_string())
    print("\n=== Per city, gradient boosting: ROC-AUC (within city) ===")
    print(pc.pivot(index="city", columns="horizon", values="roc_auc").round(3).to_string())

    print("\n=== Reliability, gradient boosting, h=1: forecast probability vs observed frequency ===")
    print(reliab[1].round(3).to_string(index=False))

    b = amt_res[amt_res["city"] == "ALL"]
    print("\n=== Precipitation amount (mm): expected value from the two-stage model ===")
    print(b.pivot(index="horizon", columns="model", values="mae")[["precip_clim", "precip_damped", "expected_precip"]].round(3).to_string())
    print("(MAE above; MSE skill vs climatology below, higher is better)")
    print(b.pivot(index="horizon", columns="model", values="mse_skill_vs_clim")[["precip_damped", "expected_precip"]].round(3).to_string())
    print("\nSaved ->", config.REPORTS / f"rain_occurrence_{args.split}.csv", "and rain_amount_*.csv")


if __name__ == "__main__":
    main()
