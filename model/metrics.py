"""Evaluation metrics used across forecasting models."""

import numpy as np
import pandas as pd


def compute_directional_accuracy(actual: pd.Series, predicted: pd.Series) -> float:
    """Compute directional accuracy from first differences.

    The metric compares whether predicted and actual month-to-month changes
    share the same sign.
    """
    # directional agreement is only meaningful when both series are observed.
    direction_df = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna().copy()
    if len(direction_df) < 2:
        return np.nan

    # compare month-over-month movement, not absolute level.
    direction_df["actual_diff"] = direction_df["actual"].diff()
    direction_df["predicted_diff"] = direction_df["predicted"].diff()
    direction_df = direction_df.dropna(subset=["actual_diff", "predicted_diff"]).copy()

    if direction_df.empty:
        return np.nan

    correct_direction = np.sign(direction_df["actual_diff"]) == np.sign(direction_df["predicted_diff"])
    return float(correct_direction.mean())


def compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """Compute MAE, RMSE, MAPE, directional accuracy, and valid observation count."""
    metric_df = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna().copy()
    if metric_df.empty:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "directional_accuracy": np.nan, "n_obs": 0}

    errors = metric_df["actual"] - metric_df["predicted"]
    mae = errors.abs().mean()
    rmse = np.sqrt((errors**2).mean())

    non_zero = metric_df["actual"] != 0
    if non_zero.any():
        # avoid division-by-zero blowups when actual values are zero.
        mape = (errors[non_zero].abs() / metric_df.loc[non_zero, "actual"]).mean() * 100.0
    else:
        mape = np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape) if not np.isnan(mape) else np.nan,
        "directional_accuracy": compute_directional_accuracy(metric_df["actual"], metric_df["predicted"]),
        "n_obs": int(len(metric_df)),
    }
