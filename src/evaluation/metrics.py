"""Evaluation metrics: small, dependency-light, and unit-tested."""
import numpy as np
from scipy.stats import rankdata


def _arr(x):
    return np.asarray(x, dtype=float)


def mae(y, p):
    return float(np.mean(np.abs(_arr(y) - _arr(p))))


def mse(y, p):
    return float(np.mean((_arr(y) - _arr(p)) ** 2))


def rmse(y, p):
    return float(np.sqrt(mse(y, p)))


def bias(y, p):
    """Mean error, prediction minus truth. Positive = forecasts run too high."""
    return float(np.mean(_arr(p) - _arr(y)))


def skill_score(model_error, reference_error):
    """1 - model/reference. 1 = perfect, 0 = no better than the reference, <0 = worse."""
    if reference_error == 0:
        return float("nan")
    return float(1.0 - model_error / reference_error)


def brier(y, p):
    """Brier score for a binary event: mean squared error of the probability."""
    return mse(y, p)


def roc_auc(y, score):
    """Rank-based (Mann-Whitney) ROC-AUC; ties get average ranks."""
    y, s = _arr(y), _arr(score)
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y, score):
    """PR-AUC as average precision (step-wise sum over distinct score thresholds)."""
    y, s = _arr(y), _arr(score)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s, yy = s[order], y[order]
    idx = np.r_[np.where(np.diff(s))[0], len(yy) - 1]
    tps = np.cumsum(yy)[idx]
    fps = (idx + 1) - tps
    precision = tps / (tps + fps)
    recall = tps / n_pos
    prev_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev_recall) * precision))
