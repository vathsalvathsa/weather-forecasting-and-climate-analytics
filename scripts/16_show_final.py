"""Reprint the FINAL test results from the saved CSVs. Reads files only: it does not touch the test data."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
pd.set_option("display.width", 200, "display.max_columns", 20)
cont_path = ROOT / "reports" / "final_test_results.csv"
rain_path = ROOT / "reports" / "final_rain_test_results.csv"
if not cont_path.exists() or not rain_path.exists():
    raise SystemExit("Final results not found: the --final run has not completed (see models/ and reports/).")

cont, rain = pd.read_csv(cont_path), pd.read_csv(rain_path)


def gain(model, ref):
    return 100 * (1 - (1 - model) / (1 - ref))


a = cont[cont["city"] == "ALL"]
skill = a.pivot(index=["target", "horizon"], columns="model", values="mse_skill_vs_clim")
mae = a.pivot(index=["target", "horizon"], columns="model", values="mae")
pooled = pd.DataFrame({"MAE_climatology": mae["climatology"], "MAE_damped": mae["damped_anomaly"],
                       "MAE_final": mae["final"], "gain_over_damped_%": gain(skill["final"], skill["damped_anomaly"]),
                       "skill_vs_climatology": skill["final"]})
print("=== FINAL TEST (2021 to 2026), all cities pooled ===")
print(pooled.round(3).to_string())

c = cont[cont["city"] != "ALL"]
rows = []
for (target, h), g in c.groupby(["target", "horizon"]):
    s = g.pivot(index="city", columns="model", values="mse_skill_vs_clim")
    gn = gain(s["final"], s["damped_anomaly"])
    rows.append({"target": target, "horizon": h, "mean_city_gain_%": gn.mean(), "min_city_gain_%": gn.min(),
                 "cities_below_0": int((gn < 0).sum())})
print("\n=== Average over the 10 cities (the main number to report) ===")
print(pd.DataFrame(rows).round(1).to_string(index=False))

for h in (1, 3):
    g = c[(c["target"] == "tmean") & (c["horizon"] == h)].pivot(index="city", columns="model", values="mse_skill_vs_clim")
    print(f"\n=== Per city, mean temperature, h={h}: % improvement over damped baseline ===")
    print(gain(g["final"], g["damped_anomaly"]).round(1).to_string())

r = rain[rain["city"] == "ALL"].pivot(index="horizon", columns="model", values="bss_vs_clim")
print("\n=== Rain occurrence: Brier skill vs climatology (pooled) ===")
print(r[["final"]].round(3).to_string())
rc = rain[(rain["city"] != "ALL") & (rain["model"] == "final")].pivot(index="city", columns="horizon", values="bss_vs_clim")
print("\n=== Rain occurrence, per city ===")
print(rc.round(3).to_string())

m = ROOT / "models" / "final_models.joblib"
print("\nSaved model file:", f"{m.stat().st_size / 1e6:.1f} MB" if m.exists() else "MISSING")
