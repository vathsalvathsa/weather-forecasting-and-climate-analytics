"""Split labels for the FINAL fit: train + val become one training block; test stays untouched."""
import pandas as pd

FINAL_TRAIN_END = pd.Timestamp("2020-12-24")   # last day of the old validation block


def final_training_split(df: pd.DataFrame) -> pd.DataFrame:
    """Everything up to 2020-12-24 (old train, the first gap and old val) becomes 'train'.

    The 7 days 2020-12-25 .. 2020-12-31 stay 'gap' so that no training target can fall
    inside the test period. Test rows are unchanged.
    """
    out = df.copy()
    out.loc[out["date"] <= FINAL_TRAIN_END, "split"] = "train"
    return out
