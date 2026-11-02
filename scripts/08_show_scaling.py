"""Reprint the scaling-experiment tables from the saved CSV (no retraining)."""
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200, "display.max_columns", 20)
out = pd.read_csv(Path(__file__).resolve().parents[1] / "reports" / "scaling_experiment_val.csv")
V = ["hgb", "hgb_w", "hgb_city"]


def gain(skill):
    return pd.DataFrame({m: 100 * (1 - (1 - skill[m]) / (1 - skill["damped_anomaly"])) for m in V})


a = out[out["city"] == "ALL"].pivot(index=["target", "horizon"], columns="model", values="mse_skill_vs_clim")
print("=== % MSE improvement over damped baseline, all cities pooled ===")
print(gain(a).round(1).to_string())
for t, h in [("tmean", 1), ("tmean", 3)]:
    c = out[(out["target"] == t) & (out["horizon"] == h) & (out["city"] != "ALL")]
    s = c.pivot(index="city", columns="model", values="mse_skill_vs_clim")
    print(f"\n=== Per city: {t} h={h} ===")
    print(gain(s).round(1).to_string())
