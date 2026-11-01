import numpy as np
import pytest

from src.evaluation import metrics as M


def test_basic_errors():
    y, p = [1.0, 2.0, 3.0], [2.0, 2.0, 5.0]
    assert M.mae(y, p) == pytest.approx(1.0)
    assert M.rmse(y, p) == pytest.approx(np.sqrt(5 / 3))
    assert M.bias(y, p) == pytest.approx(1.0)      # forecasts too high


def test_skill_score():
    assert M.skill_score(0.0, 4.0) == 1.0
    assert M.skill_score(4.0, 4.0) == 0.0
    assert M.skill_score(8.0, 4.0) == -1.0
    assert np.isnan(M.skill_score(1.0, 0.0))


def test_brier():
    assert M.brier([0, 1, 1], [0.0, 1.0, 1.0]) == 0.0
    assert M.brier([0, 1], [0.5, 0.5]) == pytest.approx(0.25)


def test_auc_and_ap_known_values():
    y, s = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    assert M.roc_auc(y, s) == pytest.approx(0.75)
    assert M.average_precision(y, s) == pytest.approx(5 / 6)


def test_auc_perfect_random_and_single_class():
    assert M.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert M.roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5     # all ties
    assert np.isnan(M.roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_against_sklearn_when_available():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(int)
    s = np.round(rng.random(500) * 0.5 + 0.4 * y, 1)                # many ties on purpose
    assert M.roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s))
    assert M.average_precision(y, s) == pytest.approx(skm.average_precision_score(y, s))
