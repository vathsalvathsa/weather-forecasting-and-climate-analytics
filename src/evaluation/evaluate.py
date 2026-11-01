"""Score prediction columns against y_true, per city and pooled ('ALL')."""
import numpy as np
import pandas as pd

from src.evaluation import metrics as M


def summarise(samples: pd.DataFrame, pred_cols: list, kind: str = "continuous",
              ref_col: str = "pred_climatology") -> pd.DataFrame:
    """kind='continuous': MAE, RMSE, bias, MSE-skill vs the reference column.
    kind='binary': Brier, Brier skill vs reference, ROC-AUC, PR-AUC.
    """
    groups = [("ALL", samples)] + [(c, g) for c, g in samples.groupby("city")]
    rows = []
    for city, g in groups:
        y = g["y_true"].to_numpy(float)
        ref = g[ref_col].to_numpy(float)
        for col in pred_cols:
            p = g[col].to_numpy(float)
            row = {"city": city, "model": col.replace("pred_", ""), "n": len(g)}
            if kind == "continuous":
                row.update(mae=M.mae(y, p), rmse=M.rmse(y, p), bias=M.bias(y, p),
                           mse_skill_vs_clim=M.skill_score(M.mse(y, p), M.mse(y, ref)))
            else:
                row.update(brier=M.brier(y, p),
                           bss_vs_clim=M.skill_score(M.brier(y, p), M.brier(y, ref)),
                           roc_auc=M.roc_auc(y, p), pr_auc=M.average_precision(y, p))
            rows.append(row)
    return pd.DataFrame(rows)
