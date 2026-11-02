"""Compare model gains with vs without location features (reads saved CSVs only)."""
from pathlib import Path

import pandas as pd

REPORTS = Path(__file__).resolve().parents[1] / "reports"
pd.set_option("display.width", 200)


def gains(path):
    out = pd.read_csv(path)
    a = out[out["city"] == "ALL"]
    sk = a.pivot(index=["target", "horizon"], columns="model", values="mse_skill_vs_clim")
    return pd.DataFrame({m: 100 * (1 - (1 - sk[m]) / (1 - sk["damped_anomaly"])) for m in ["ridge", "hgb"]})


def city_gain(path, target="tmean", horizon=1):
    out = pd.read_csv(path)
    c = out[(out["target"] == target) & (out["horizon"] == horizon) & (out["city"] != "ALL")]
    s = c.pivot(index="city", columns="model", values="mse_skill_vs_clim")
    return 100 * (1 - (1 - s["hgb"]) / (1 - s["damped_anomaly"]))


with_loc = REPORTS / "models_v1_val.csv"
no_loc = REPORTS / "models_v1_val_nolocation.csv"

print("=== % MSE improvement over damped baseline ===")
print(pd.concat({"with_location": gains(with_loc), "no_location": gains(no_loc)}, axis=1).round(1).to_string())
print("\n=== hgb, tmean h=1, per city ===")
print(pd.DataFrame({"with_location": city_gain(with_loc), "no_location": city_gain(no_loc)}).round(1).to_string())
