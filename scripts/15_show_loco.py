"""Reprint the leave-cities-out summary from the saved CSV (no retraining)."""
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200, "display.max_columns", 20)
path = Path(__file__).resolve().parents[1] / "reports" / "leave_cities_out_val.csv"
if not path.exists():
    raise SystemExit("reports/leave_cities_out_val.csv not found: run scripts/12_leave_cities_out.py first")
out = pd.read_csv(path)
V = ["insample", "loco_loc", "loco_noloc"]


def gains(skill):
    return pd.DataFrame({m: 100 * (1 - (1 - skill[m]) / (1 - skill["damped_anomaly"])) for m in V})


rows = []
for (target, h), g in out[out["city"] != "ALL"].groupby(["target", "horizon"]):
    gn = gains(g.pivot(index="city", columns="model", values="mse_skill_vs_clim"))
    row = {"target": target, "horizon": h}
    for v in V:
        row[f"{v}_mean_gain"] = gn[v].mean()
        row[f"{v}_cities_below_0"] = int((gn[v] < 0).sum())
    rows.append(row)
summary = pd.DataFrame(rows)
print("=== Average over cities of % MSE improvement over each city's damped baseline ===")
print(summary.round(1).to_string(index=False))
print("\nAverage over all six jobs: with location %.1f | without location %.1f | difference %.1f points" % (
    summary["loco_loc_mean_gain"].mean(), summary["loco_noloc_mean_gain"].mean(),
    summary["loco_loc_mean_gain"].mean() - summary["loco_noloc_mean_gain"].mean()))
c = out[(out["target"] == "tmean") & (out["horizon"] == 1) & (out["city"] != "ALL")]
print("\n=== Per city, tmean h=1 ===")
print(gains(c.pivot(index="city", columns="model", values="mse_skill_vs_clim")).round(1).to_string())
