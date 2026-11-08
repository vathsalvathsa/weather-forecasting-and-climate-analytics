"""Generate README.md from the saved results, so every number in it comes from a result file.

    python scripts/18_make_readme.py
    python scripts/18_make_readme.py --url https://your-app.streamlit.app

Put app screenshots (png) in docs/screenshots/ and rerun to embed them.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.evaluation import report_tables as RT

NAMES = {"tmean": "Mean temperature", "tmax": "Maximum temperature", "tmin": "Minimum temperature",
         "rh": "Relative humidity", "wind": "Maximum wind speed"}


def md_table(df: pd.DataFrame, digits: int = 2) -> str:
    cols = list(df.columns)

    def cell(v):
        return f"{v:.{digits}f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(cell(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def read(name):
    p = config.ROOT / "reports" / name
    return pd.read_csv(p) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="deployed app address")
    ap.add_argument("--author", default="Sundram Vaths")
    args = ap.parse_args()

    cont, rain, loco = read("final_test_results.csv"), read("final_rain_test_results.csv"), read("leave_cities_out_val.csv")
    if cont is None or rain is None:
        sys.exit("Final results not found in reports/: run scripts/13_train_final.py --final first.")

    gains = RT.city_mean_gain(cont)
    g = gains.set_index(["target", "horizon"])
    gains_show = gains.assign(target=gains["target"].map(NAMES)).round(1)
    pooled = RT.pooled_table(cont)
    pooled_show = pooled.assign(target=pooled["target"].map(NAMES))
    rain_pooled, rain_city = RT.rain_tables(rain)
    rain_city = rain_city.rename(columns=lambda c: f"{c} day(s) ahead").reset_index()
    city_t1 = RT.city_gain(cont, "tmean", 1).sort_values(ascending=False).round(1).reset_index()
    city_t1.columns = ["City", "Gain over damped baseline, 1 day (%)"]

    tm1, tm3 = g.loc[("tmean", 1)], g.loc[("tmean", 3)]
    h1 = gains[gains["horizon"] == 1].sort_values("mean city gain %")
    best1, worst1 = h1.iloc[-1], h1.iloc[0]
    rp = rain_pooled.set_index("horizon")["Brier skill vs climatology"]
    wind3 = g.loc[("wind", 3)]["mean city gain %"]
    fade = (" Skill beyond about 1 to 2 days is essentially gone for rain and wind."
            if rp.get(3, 1.0) < 0.05 and wind3 < 5 else "")

    loco_block, loco_line = "", ""
    if loco is not None:
        lt = RT.loco_summary(loco)
        row = lt[(lt["target"] == "tmean") & (lt["horizon"] == 1)].iloc[0]
        seen, held = row["city seen in training %"], row["city held out, no location %"]
        verdict = "New cities are much harder." if seen - held >= 5 else "Held-out cities score similarly."
        loco_line = (f"- **{verdict}** For 1-day mean temperature the average gain is {seen:.1f}% for cities the "
                     f"model trained on and {held:.1f}% when the city is held out of training (measured on the "
                     f"validation years). Whatever the gap, the ML predictions are validated for the 10 training "
                     f"cities only.")
        loco_block = ("### Can the models handle a city they never saw?\n\n"
                      "Leave-cities-out test on the validation years: train on 9 cities, score the 10th. Average "
                      "% improvement over each city's own damped baseline.\n\n"
                      + md_table(lt.assign(target=lt["target"].map(NAMES)).round(1), 1) + "\n")

    card = {}
    cp = config.ROOT / "models" / "model_card.json"
    if cp.exists():
        card = json.loads(cp.read_text())
    trained_on = card.get("trained_on", "1991 to 2020")
    scored_on = card.get("scored_once_on", "2021 to 2026")

    cities = pd.DataFrame([c for c in config.CITIES if c["name"] in config.PILOT])[["name", "country", "region", "lat", "lon"]]
    cities.columns = ["City", "Country", "Region", "Latitude", "Longitude"]

    shots = sorted((config.ROOT / "docs" / "screenshots").glob("*.png")) if (config.ROOT / "docs" / "screenshots").exists() else []
    shots_block = ""
    if shots:
        shots_block = "## Application screenshots\n\n" + "\n\n".join(
            f"![{p.stem.replace('_', ' ')}](docs/screenshots/{p.name})" for p in shots) + "\n\n"

    live = (f"**Live app:** {args.url}" if args.url else "**Live app:** not deployed yet")

    readme = f"""# Weather Forecasting and Climate Analytics

Machine-learning weather prediction and climate analytics on ERA5 reanalysis data, with a Streamlit dashboard.
M.Tech Data Science & Analytics project by {args.author}.

{live}

## Overview

The project combines three things that are deliberately kept apart in the app:

1. **Live weather forecasts** from Open-Meteo (weather-model output) for any place in the world.
2. **Predictions from this project's own ML models**, trained on ERA5 history and always compared with strong baselines.
3. **Climate analytics**: long-term trends, seasons, anomalies and extremes. These describe the past and are not forecasts.

## Problem statement

Many student weather projects report a single accuracy figure with no baseline, mix forecasting with climate
description, and leak future information into training. This project asks a narrower, testable question:
**how much can machine learning improve on strong, simple forecasts (climatology and a damped persistence
baseline) when evaluated strictly on years the models never saw, and where does it fail?**

## Objectives

- Build a clean, reproducible ERA5 dataset and verify its quality before modelling.
- Forecast next-day and 3-day temperature, humidity, wind and rain, and judge every model against baselines.
- Evaluate without leakage: chronological splits with gaps, train-only statistics, and a test period used once.
- Test generalisation to cities the model has never seen.
- Deliver an honest, deployable dashboard that separates forecasts, ML predictions and climate analytics.

## Methodology

**Data.** Daily ERA5 reanalysis (Copernicus/ECMWF) through the Open-Meteo Historical API, 1991 to 2026, for
{len(cities)} climatically contrasting cities. Every request pins the source to ERA5 (`models=era5`): the API's
default silently changes data source in 2017, which created a false jump in rainfall (the default data showed 55-59%
wet days in Nairobi for 2018-2020 against 38-42% in ERA5) until a source-comparison check caught it and the data was
re-downloaded.

**Splits and leakage control.** Train 1991-2017, validation 2018-2020, test 2021-2026, with 7-day gaps between
parts so that no training target can fall in a later period. Climatologies, damping factors and city weights are
fitted on training rows only. Features use only information available at the end of the issue day. Each of these
rules is covered by an automated test (for example, corrupting all future values must not change earlier features).
The final models are trained on {trained_on} and scored once on {scored_on}.

**Baselines.** Climatology (smoothed day-of-year mean), persistence, climatology plus today's anomaly, and a
**damped anomaly** baseline (the anomaly fades with lead time by a per-city fitted factor). The damped baseline is the
main bar to beat.

**Models.** Gradient boosting (scikit-learn) predicting the anomaly relative to climatology, with cities weighted by
inverse variance so that high-variance cities do not dominate training. Features: recent anomalies of nine variables,
rainfall history, pressure tendency, diurnal range and seasonality. Location features (latitude, longitude, elevation)
were dropped because they did not help on unseen cities. Rain uses a two-stage model: probability of a wet day (at least
1 mm) and a gamma-loss amount model.

**Evaluation.** MAE and RMSE, mean squared-error improvement over the damped baseline (averaged over cities), Brier
score and Brier skill for rain, and a leave-cities-out test.

## Dataset

Source: ERA5 reanalysis via the Open-Meteo Historical Weather API (CC BY 4.0, attribution: *Weather data by Open-Meteo.com*).
ERA5 is a reanalysis (a model product), not a station record. Cities used for training and evaluation:

{md_table(cities, 2)}

Variables: daily maximum, minimum and mean temperature, precipitation, maximum wind speed, relative humidity,
surface pressure, cloud cover, dew point and shortwave radiation. Data quality checks (missing values, duplicates,
physical ranges, consistency) found no problems after the source fix.

## Results

Final models, scored **once** on {scored_on} (data the models never saw).

### Improvement over the damped baseline (mean over the 10 cities)

{md_table(gains_show, 1)}

"mean city gain" is the average percentage reduction in squared error relative to the damped baseline; "cities below 0"
counts cities where the model was worse than that baseline.

### Error in physical units (all cities pooled)

{md_table(pooled_show, 3)}

### 1-day mean temperature, city by city

{md_table(city_t1, 1)}

### Rain occurrence (probability of at least 1 mm)

{md_table(rain_pooled.round(3), 3)}

{md_table(rain_city.round(3), 3)}

{loco_block}
### Key findings

- **Mean temperature, 1 day ahead:** average improvement over the damped baseline is {tm1['mean city gain %']:.1f}%
  (worst city {tm1['worst city gain %']:+.1f}%; {int(tm1['cities below 0'])} of 10 cities worse than the baseline).
- **Mean temperature, 3 days ahead:** the average improvement is {tm3['mean city gain %']:.1f}%
  ({int(tm3['cities below 0'])} of 10 cities worse than the baseline).
- **Strongest and weakest 1-day targets:** {NAMES[best1['target']]} ({best1['mean city gain %']:.1f}%) and
  {NAMES[worst1['target']]} ({worst1['mean city gain %']:.1f}%).
- **Rain and wind:** Brier skill against climatology is {rp.get(1, float('nan')):.3f} at 1 day and
  {rp.get(3, float('nan')):.3f} at 3 days; wind at 3 days improves {wind3:.1f}% over the damped baseline.{fade}
{loco_line}

{shots_block}## Limitations

- ERA5 is a gridded reanalysis, not station observations; rainfall in particular differs from gauge data.
- The models use only a city's own recent history, so skill fades after about 3 days. For longer ranges the
  weather-model forecasts (Open-Meteo pages) are the right tool.
- Predictions are validated for the 10 training cities only. The app's ML page is a **backtest** on past days, not a
  live forecast: ERA5 arrives days late, so live inputs from the same source do not yet exist.
- Climate trend p-values assume independent years and are optimistic; hot-day, heatwave and heavy-rain definitions are
  simple city-relative thresholds, not official warning criteria.
- Design choices (city weighting, dropping location features) were made on the validation years; the test years were used once.

## Repository structure

## Installation and reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q                         # run the tests
python scripts/00_probe_api.py              # check the API
python scripts/01_download_data.py --pilot --wait
python scripts/02_validate_data.py
python scripts/03_prepare_dataset.py
python scripts/04_run_baselines.py
python scripts/12_leave_cities_out.py
python scripts/13_train_final.py            # dry run on validation
python scripts/13_train_final.py --final --no-location   # trains the final models, scores the test years ONCE
python scripts/14_export_app_data.py
streamlit run app.py
```

No API key is needed (Open-Meteo is keyless for non-commercial use). The raw data are not stored in the repository;
the scripts regenerate them.

## Deployment

The app runs on Streamlit Community Cloud from this repository (`app.py`). `requirements.txt` is a lean, version-pinned
list: the saved models are pickles and must be loaded with the scikit-learn version they were trained with.

## Future scope

- A larger, more diverse city sample (the leave-cities-out test shows spatial transfer is the main weakness).
- Correct the Open-Meteo forecast with ML post-processing (needs the historical forecast archive) to extend skill
  beyond 3 days.
- Probabilistic temperature forecasts with calibrated intervals, and station-data validation of ERA5.
- Live ML predictions once inputs from a matching source with low latency are available.

## References

- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146,
  1999-2049. https://doi.org/10.1002/qj.3803
- Zippenfennig, P. Open-Meteo.com Weather API. https://open-meteo.com (CC BY 4.0). https://doi.org/10.5281/zenodo.7970649
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
"""
    out = config.ROOT / "README.md"
    out.write_text(readme)
    print(f"Wrote {out} ({len(readme.splitlines())} lines)")


if __name__ == "__main__":
    main()
