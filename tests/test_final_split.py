import numpy as np
import pandas as pd

from src.data.final_split import final_training_split
from src.data.prepare import assign_split


def make_df():
    dates = pd.date_range("2015-01-01", "2023-12-31")
    df = pd.DataFrame({"city": "C", "date": dates})
    df["split"] = assign_split(df["date"])
    return df


def test_final_split_merges_train_and_val_and_keeps_test():
    out = final_training_split(make_df())
    lab = dict(zip(out["date"].dt.strftime("%Y-%m-%d"), out["split"]))
    assert lab["2017-12-28"] == "train"          # old gap inside the training block
    assert lab["2019-06-01"] == "train"          # old validation block
    assert lab["2020-12-24"] == "train"
    assert all(lab[f"2020-12-{d}"] == "gap" for d in range(25, 32))
    assert lab["2021-01-01"] == "test"
    assert "val" not in set(out["split"])


def test_final_split_does_not_touch_the_input_or_test_rows():
    df = make_df()
    before = df["split"].copy()
    out = final_training_split(df)
    assert df["split"].equals(before)            # original unchanged
    assert (out.loc[out["split"] == "test", "date"] >= pd.Timestamp("2021-01-01")).all()
    assert (out["split"] == "test").sum() == (df["split"] == "test").sum()


def test_bundle_roundtrip_keeps_predictions(tmp_path):
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    from src.models.baselines import Climatology
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 3)), columns=list("abc"))
    y = X["a"] * 2 + rng.normal(0, 0.1, 500)
    model = HistGradientBoostingRegressor(max_iter=20).fit(X, y)
    dates = pd.date_range("2000-01-01", "2010-12-31")
    clim = Climatology().fit(pd.DataFrame({"city": "C", "date": dates, "v": rng.normal(size=len(dates))}), "v")
    path = tmp_path / "bundle.joblib"
    joblib.dump({"m": model, "clim": clim}, path, compress=3)
    back = joblib.load(path)
    assert np.allclose(back["m"].predict(X), model.predict(X))
    assert np.allclose(back["clim"].table_.to_numpy(), clim.table_.to_numpy())
