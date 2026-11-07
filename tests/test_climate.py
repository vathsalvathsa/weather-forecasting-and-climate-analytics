import numpy as np
import pytest

from src.climate.trends import annual_trend, count_runs


def test_trend_recovers_a_known_slope():
    years = np.arange(1991, 2026)
    rng = np.random.default_rng(0)
    values = 20 + 0.03 * (years - 1991) + rng.normal(0, 0.15, len(years))     # +0.3 per decade
    t = annual_trend(years, values)
    assert t["sen_slope_per_decade"] == pytest.approx(0.3, abs=0.08)
    assert t["sen_low_per_decade"] < 0.3 < t["sen_high_per_decade"] + 0.05
    assert t["p_value"] < 0.001 and t["kendall_tau"] > 0.5


def test_no_trend_gives_large_p_value():
    years = np.arange(1991, 2026)
    values = np.random.default_rng(1).normal(20, 0.5, len(years))
    assert annual_trend(years, values)["p_value"] > 0.05


def test_trend_needs_enough_years_and_ignores_missing():
    with pytest.raises(ValueError):
        annual_trend([2000, 2001, 2002], [1.0, 2.0, 3.0])
    years = np.arange(2000, 2020)
    values = years * 0.1
    values[3] = np.nan
    assert annual_trend(years, values)["n_years"] == 19


def test_count_runs():
    assert count_runs([1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], 3) == 2
    assert count_runs([1, 1, 0, 1, 1], 3) == 0
    assert count_runs([0, 0, 0], 3) == 0
    assert count_runs([1, 1, 1], 3) == 1                     # run touching the end counts
    assert count_runs([], 3) == 0
