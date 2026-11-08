import numpy as np
import pandas as pd
import pytest

from src.evaluation import report_tables as RT


def make_cont():
    rows = []
    for city, f in [("ALL", 0.6), ("A", 0.5), ("B", 0.7)]:
        for model, skill, mae in [("climatology", 0.0, 2.0), ("damped_anomaly", 0.4, 1.5), ("final", f, 1.2)]:
            rows.append({"target": "tmean", "horizon": 1, "city": city, "model": model,
                         "mse_skill_vs_clim": skill, "mae": mae})
    return pd.DataFrame(rows)


def test_gain_over_reference():
    assert RT.gain_over_ref(0.4, 0.4) == pytest.approx(0.0)
    assert RT.gain_over_ref(0.7, 0.4) == pytest.approx(50.0)       # error 0.3 vs 0.6
    assert RT.gain_over_ref(0.2, 0.4) < 0


def test_pooled_city_tables():
    cont = make_cont()
    p = RT.pooled_table(cont)
    assert list(p["MAE final model"]) == [1.2]
    assert p["gain over damped baseline %"].iloc[0] == pytest.approx(100 * (1 - 0.4 / 0.6))
    g = RT.city_mean_gain(cont)
    assert g["cities below 0"].iloc[0] == 0
    assert g["mean city gain %"].iloc[0] == pytest.approx(np.mean([100 * (1 - 0.5 / 0.6), 100 * (1 - 0.3 / 0.6)]))
    s = RT.city_gain(cont, "tmean", 1)
    assert set(s.index) == {"A", "B"} and s["B"] > s["A"]


def test_rain_and_loco_tables():
    rain = pd.DataFrame({"horizon": [1, 1, 1], "city": ["ALL", "A", "A"], "model": ["final", "final", "climatology"],
                         "brier": [0.1, 0.1, 0.2], "bss_vs_clim": [0.2, 0.25, 0.0]})
    pooled, per_city = RT.rain_tables(rain)
    assert pooled["Brier skill vs climatology"].iloc[0] == 0.2
    assert per_city.loc["A", 1] == 0.25
    loco = pd.DataFrame([{"target": "tmean", "horizon": 1, "city": c, "model": m, "mse_skill_vs_clim": v}
                         for c in ("A", "B")
                         for m, v in [("damped_anomaly", 0.4), ("insample", 0.6), ("loco_loc", 0.5), ("loco_noloc", 0.55)]])
    out = RT.loco_summary(loco)
    assert out["city seen in training %"].iloc[0] > out["city held out, no location %"].iloc[0] > out["city held out, with location %"].iloc[0]
