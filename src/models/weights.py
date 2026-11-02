"""Sample weights that stop high-variance cities from dominating the loss."""
import numpy as np
import pandas as pd


def city_weights(train: pd.DataFrame, target_col: str = "y_anom") -> np.ndarray:
    """Weight each row by 1 / (that city's variance of the target), averaging 1.

    Pass TRAIN rows only: the variances are statistics fitted on training data.
    With squared-error loss this makes every city count about equally,
    whatever its natural day-to-day variability.
    """
    var = train.groupby("city")[target_col].transform("var").to_numpy(float)
    w = 1.0 / var
    return w / w.mean()
