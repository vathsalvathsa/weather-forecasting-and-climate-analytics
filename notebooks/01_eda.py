# %% [markdown]
# # 01 - Exploratory Data Analysis (pilot cities)
# Open in VS Code and click **Run Cell** on each `# %%` block, or run the whole
# file with `python notebooks/01_eda.py` (figures are saved to `reports/figures/`).
#
# **Rules for this notebook**
# - EDA is DESCRIPTIVE. Nothing computed here may be used as a model feature unless
#   it is recomputed from the TRAINING period only (otherwise: data leakage).
# - Annual statistics use complete years only (1991-2025); 2026 is partial.
# - Data: ERA5 reanalysis via Open-Meteo (CC BY 4.0). It is a model product, not
#   a station observation.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src import config

sns.set_theme(style="whitegrid", context="notebook")
FIG_DIR = config.REPORTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def finish(fig, name):
    """Save the figure; show it in Jupyter/VS Code, close it in a plain script."""
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    try:
        get_ipython()  # only defined inside Jupyter / VS Code interactive
        plt.show()
    except NameError:
        plt.close(fig)


# %% [markdown]
# ## 1. Load all downloaded cities into one table

# %%
meta = pd.read_csv(config.CITY_META_FILE)
meta["slug"] = meta["name"].map(config.slug)
slug_to_name = dict(zip(meta["slug"], meta["name"]))

frames = []
for f in sorted(config.DATA_RAW.glob("*.parquet")):
    d = pd.read_parquet(f)
    d["city"] = slug_to_name.get(f.stem, f.stem)
    frames.append(d)
df = pd.concat(frames, ignore_index=True).sort_values(["city", "date"]).reset_index(drop=True)
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["doy"] = df["date"].dt.day_of_year

print("cities:", df["city"].nunique(), "| rows:", len(df))
print("period:", df["date"].min().date(), "->", df["date"].max().date())
meta[["name", "country", "lat", "lon", "elevation_m", "region"]]

# %% [markdown]
# ## 2. Variable summary (units are those returned by the API)

# %%
VARS = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "precipitation_sum", "wind_speed_10m_max", "relative_humidity_2m_mean",
        "surface_pressure_mean", "cloud_cover_mean", "dew_point_2m_mean",
        "shortwave_radiation_sum", "et0_fao_evapotranspiration"]
df[VARS].describe().T.round(2)

# %% [markdown]
# ## 3. Per-city climate summary
# Wet day = precipitation >= 1 mm (the common WMO convention). Annual totals use
# complete years only.

# %%
full = df[df["year"].between(1991, 2025)]
annual = full.groupby(["city", "year"]).agg(
    annual_tmean=("temperature_2m_mean", "mean"),
    annual_precip=("precipitation_sum", "sum")).reset_index()

summary = full.groupby("city").agg(
    tmean=("temperature_2m_mean", "mean"),
    tmax_p99=("temperature_2m_max", lambda s: s.quantile(0.99)),
    tmin_p01=("temperature_2m_min", lambda s: s.quantile(0.01)),
    humidity=("relative_humidity_2m_mean", "mean"),
    wet_day_frac=("precipitation_sum", lambda s: (s >= 1).mean()),
).join(annual.groupby("city")["annual_precip"].mean().rename("annual_precip_mm"))
summary = summary.join(meta.set_index("name")[["lat", "lon", "elevation_m"]]).round(2)
summary.to_csv(config.REPORTS / "eda_city_summary.csv")
summary.sort_values("tmean")

# %% [markdown]
# ## 4. Seasonal cycle of temperature
# Shows why `month` / `day-of-year` features (sin/cos) are needed, and that the
# Southern Hemisphere is phase-shifted.

# %%
seasonal = full.pivot_table(index="month", columns="city", values="temperature_2m_mean", aggfunc="mean")
fig, ax = plt.subplots(figsize=(10, 5))
seasonal.plot(ax=ax, marker="o", ms=3)
ax.set(xlabel="Month", ylabel="Mean temperature (deg C)", title="Seasonal cycle of daily mean temperature, 1991-2025")
ax.legend(ncol=2, fontsize=8)
finish(fig, "seasonal_cycle_temperature.png")

# %% [markdown]
# ## 5. Rainfall is zero-inflated and heavy-tailed
# Most days have little or no rain, a few days have very large totals. This is why
# rainfall is modelled in two stages (rain / no rain, then amount) and why MAE on
# all days can look deceptively good.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
dry = full.groupby("city")["precipitation_sum"].apply(lambda s: (s < 1).mean()).sort_values()
sns.barplot(x=dry.values, y=dry.index, ax=axes[0], color="steelblue")
axes[0].set(xlabel="Fraction of dry days (< 1 mm)", ylabel="", title="Dry-day fraction by city")
wet = full.loc[full["precipitation_sum"] >= 1, "precipitation_sum"]
axes[1].hist(wet, bins=np.logspace(0, np.log10(max(wet.max(), 10)), 50), color="steelblue")
axes[1].set(xscale="log", yscale="log", xlabel="Wet-day precipitation (mm, log)", ylabel="Days (log)",
            title="Wet-day amounts, all cities pooled")
finish(fig, "rainfall_distribution.png")

# %% [markdown]
# ## 6. Relationships between variables
# Each variable is z-scored WITHIN its city first, so the heatmap shows how
# variables move together locally, not differences between climates.

# %%
CORR_VARS = [v for v in VARS if v != "temperature_2m_mean"] + ["temperature_2m_mean"]
z = full.groupby("city")[CORR_VARS].transform(lambda s: (s - s.mean()) / s.std())
corr = z.corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Within-city standardised correlations (daily)")
finish(fig, "correlation_heatmap.png")

# %% [markdown]
# ## 7. Persistence: how predictable is tomorrow from today?
# Autocorrelation of the temperature ANOMALY (day minus the day-of-year average).
# High lag-1 autocorrelation means a persistence baseline is hard to beat for
# temperature; rain occurrence is far less persistent. (Anomalies here use the full
# period - fine for EDA, NOT for modelling.)

# %%
df["tmean_anom"] = df["temperature_2m_mean"] - df.groupby(["city", "doy"])["temperature_2m_mean"].transform("mean")
df["rain_flag"] = (df["precipitation_sum"] >= 1).astype(float)
lags = range(1, 31)
acf_t = pd.DataFrame({c: [g["tmean_anom"].autocorr(l) for l in lags] for c, g in df.groupby("city")}, index=list(lags))
acf_r = pd.DataFrame({c: [g["rain_flag"].autocorr(l) for l in lags] for c, g in df.groupby("city")}, index=list(lags))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
acf_t.plot(ax=axes[0], legend=False); axes[0].set(title="Temperature anomaly autocorrelation", xlabel="Lag (days)", ylabel="Autocorrelation")
acf_r.plot(ax=axes[1]); axes[1].set(title="Rain-day autocorrelation", xlabel="Lag (days)")
axes[1].legend(ncol=2, fontsize=7)
finish(fig, "autocorrelation.png")
print("Lag-1 autocorrelation, temperature anomaly:\n", acf_t.loc[1].round(2).sort_values().to_string())

# %% [markdown]
# ## 8. Long-term temperature trend (preview)
# Simple linear trend of annual mean temperature, 1991-2025. This is a PREVIEW:
# the climate module will use Mann-Kendall / Sen's slope, which do not assume
# normally distributed residuals. ERA5 is also not a station record.

# %%
rows = []
for city, g in annual.groupby("city"):
    res = stats.linregress(g["year"], g["annual_tmean"])
    rows.append({"city": city, "trend_C_per_decade": res.slope * 10, "p_value": res.pvalue})
trend = pd.DataFrame(rows).set_index("city").round(3).sort_values("trend_C_per_decade")
trend.to_csv(config.REPORTS / "eda_trend_preview.csv")

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=trend["trend_C_per_decade"], y=trend.index, ax=ax, color="indianred")
ax.set(xlabel="deg C per decade (OLS, 1991-2025)", ylabel="", title="Annual mean temperature trend (preview)")
finish(fig, "temperature_trend_preview.png")
trend

# %% [markdown]
# ## 9. Extremes flagged by the audit
# Look at (do not delete) the days the validator flagged, and judge whether they
# are plausible weather events.

# %%
t = df["temperature_2m_mean"]
med = t.groupby([df["city"], df["doy"]]).transform("median")
mad = (t - med).abs().groupby([df["city"], df["doy"]]).transform("median")
df["robust_z"] = 0.6745 * (t - med) / mad.where(mad > 0)
flagged = df.loc[df["robust_z"].abs() > 5, ["city", "date", "temperature_2m_min",
                                            "temperature_2m_mean", "temperature_2m_max", "robust_z"]]
print(len(flagged), "flagged city-days")
flagged.round({"temperature_2m_min": 2, "temperature_2m_mean": 2, "temperature_2m_max": 2, "robust_z": 2}).sort_values("robust_z", key=abs, ascending=False).head(30)
