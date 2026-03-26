import numpy as np
import pandas as pd


def compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    metric_df = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna().copy()
    if metric_df.empty:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "n_obs": 0}

    errors = metric_df["actual"] - metric_df["predicted"]
    mae = errors.abs().mean()
    rmse = np.sqrt((errors**2).mean())

    non_zero = metric_df["actual"] != 0
    if non_zero.any():
        mape = (errors[non_zero].abs() / metric_df.loc[non_zero, "actual"]).mean() * 100.0
    else:
        mape = np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape) if not np.isnan(mape) else np.nan,
        "n_obs": int(len(metric_df)),
    }
