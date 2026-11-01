"""Step 4.1: score the baseline forecasts. Development uses train/val only.

    python scripts/04_run_baselines.py              # validation split (default)
    python scripts/04_run_baselines.py --split train

The TEST split is reserved for one final evaluation of the chosen models. It is
locked here behind --final so that it cannot be peeked at by accident.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.evaluation.evaluate import summarise
from src.models.baselines import baseline_predictions

CONTINUOUS = {
    "tmax": "temperature_2m_max", "tmin": "temperature_2m_min",
    "tmean": "temperature_2m_mean", "humidity": "relative_humidity_2m_mean",
    "wind_max": "wind_speed_10m_max", "precip_mm": "precipitation_sum",
}
HORIZONS = [1, 3, 7]
CONT_PREDS = ["pred_persistence", "pred_climatology", "pred_anomaly_persistence", "pred_damped_anomaly"]
RAIN_PREDS = ["pred_persistence", "pred_climatology"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--final", action="store_true", help="required to touch the test split")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        sys.exit("The test split is reserved for the final evaluation. Use --split val.")

    df = pd.read_parquet(config.DATA_PROCESSED / "daily_all_cities.parquet")
    df["wet"] = (df["precipitation_sum"] >= 1.0).astype(float)   # wet day: >= 1 mm

    cont, rain = [], []
    for name, col in CONTINUOUS.items():
        for h in HORIZONS:
            s = baseline_predictions(df, col, h)
            s = s[s["split"] == args.split]
            res = summarise(s, CONT_PREDS, "continuous")
            res.insert(0, "target", name)
            res.insert(1, "horizon", h)
            cont.append(res)
    for h in HORIZONS:
        s = baseline_predictions(df, "wet", h, kinds=("persistence", "climatology"))
        s = s[s["split"] == args.split]
        res = summarise(s, RAIN_PREDS, "binary")
        res.insert(0, "target", "rain_occurrence")
        res.insert(1, "horizon", h)
        rain.append(res)

    cont, rain = pd.concat(cont), pd.concat(rain)
    config.REPORTS.mkdir(exist_ok=True)
    out = pd.concat([cont, rain], ignore_index=True)
    out.to_csv(config.REPORTS / f"baselines_{args.split}.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    c_all = cont[cont["city"] == "ALL"]
    print(f"\n=== MAE, all cities pooled, split = {args.split} ===")
    print(c_all.pivot(index=["target", "horizon"], columns="model", values="mae").round(2).to_string())
    print("\n=== MSE skill vs climatology (0 = same as climatology, higher = better) ===")
    print(c_all.pivot(index=["target", "horizon"], columns="model", values="mse_skill_vs_clim").round(3).to_string())
    r_all = rain[rain["city"] == "ALL"]
    print("\n=== Rain occurrence (>= 1 mm): Brier score (lower is better) ===")
    print(r_all.pivot(index="horizon", columns="model", values="brier").round(4).to_string())
    print("\n=== Rain occurrence: ROC-AUC / PR-AUC ===")
    print(r_all.pivot(index="horizon", columns="model", values=["roc_auc", "pr_auc"]).round(3).to_string())
    print("\nSaved ->", config.REPORTS / f"baselines_{args.split}.csv")


if __name__ == "__main__":
    main()
