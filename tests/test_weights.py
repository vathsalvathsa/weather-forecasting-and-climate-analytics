import numpy as np
import pandas as pd

from src.models.weights import city_weights


def make_train(seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "city": ["quiet"] * 5000 + ["wild"] * 5000,
        "y_anom": np.r_[rng.normal(0, 1.0, 5000), rng.normal(0, 4.0, 5000)],   # std 1 vs 4
    })


def test_weights_are_inverse_variance_and_average_one():
    t = make_train()
    w = city_weights(t)
    assert abs(w.mean() - 1.0) < 1e-9
    ratio = w[t["city"] == "quiet"].mean() / w[t["city"] == "wild"].mean()
    assert 14 < ratio < 18                      # (4 / 1) ** 2 = 16


def test_weights_use_only_the_rows_given():
    t = make_train()
    w1 = city_weights(t)
    # Appending extreme rows after the fact must change weights, proving they are
    # computed from the table passed in (callers pass TRAIN rows only).
    t2 = pd.concat([t, pd.DataFrame({"city": ["quiet"] * 100, "y_anom": [50.0] * 100})], ignore_index=True)
    assert not np.allclose(w1, city_weights(t2)[: len(t)])
