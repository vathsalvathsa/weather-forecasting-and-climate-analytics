"""Tables for the Model Performance page, computed from the saved result CSVs."""
import pandas as pd


def gain_over_ref(model_skill, ref_skill):
    """% reduction in squared error relative to a reference forecast, from skill scores vs climatology."""
    return 100 * (1 - (1 - model_skill) / (1 - ref_skill))


def pooled_table(cont: pd.DataFrame) -> pd.DataFrame:
    a = cont[cont["city"] == "ALL"]
    skill = a.pivot(index=["target", "horizon"], columns="model", values="mse_skill_vs_clim")
    mae = a.pivot(index=["target", "horizon"], columns="model", values="mae")
    return pd.DataFrame({"MAE climatology": mae["climatology"], "MAE damped baseline": mae["damped_anomaly"],
                         "MAE final model": mae["final"],
                         "gain over damped baseline %": gain_over_ref(skill["final"], skill["damped_anomaly"]),
                         "skill vs climatology": skill["final"]}).reset_index()


def city_mean_gain(cont: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, h), g in cont[cont["city"] != "ALL"].groupby(["target", "horizon"]):
        s = g.pivot(index="city", columns="model", values="mse_skill_vs_clim")
        gn = gain_over_ref(s["final"], s["damped_anomaly"])
        rows.append({"target": target, "horizon": h, "mean city gain %": gn.mean(),
                     "worst city gain %": gn.min(), "cities below 0": int((gn < 0).sum())})
    return pd.DataFrame(rows)


def city_gain(cont: pd.DataFrame, target: str, horizon: int) -> pd.Series:
    g = cont[(cont["target"] == target) & (cont["horizon"] == horizon) & (cont["city"] != "ALL")]
    s = g.pivot(index="city", columns="model", values="mse_skill_vs_clim")
    return gain_over_ref(s["final"], s["damped_anomaly"]).rename("gain over damped baseline %")


def rain_tables(rain: pd.DataFrame):
    pooled = rain[(rain["city"] == "ALL") & (rain["model"] == "final")][["horizon", "brier", "bss_vs_clim"]]
    per_city = (rain[(rain["city"] != "ALL") & (rain["model"] == "final")]
                .pivot(index="city", columns="horizon", values="bss_vs_clim"))
    return pooled.rename(columns={"bss_vs_clim": "Brier skill vs climatology"}), per_city


def loco_summary(loco: pd.DataFrame) -> pd.DataFrame:
    variants = ["insample", "loco_loc", "loco_noloc"]
    rows = []
    for (target, h), g in loco[loco["city"] != "ALL"].groupby(["target", "horizon"]):
        s = g.pivot(index="city", columns="model", values="mse_skill_vs_clim")
        row = {"target": target, "horizon": h}
        for v in variants:
            row[v] = gain_over_ref(s[v], s["damped_anomaly"]).mean()
        rows.append(row)
    return pd.DataFrame(rows).rename(columns={
        "insample": "city seen in training %", "loco_loc": "city held out, with location %",
        "loco_noloc": "city held out, no location %"})
