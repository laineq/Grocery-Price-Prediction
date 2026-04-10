"""Baseline forecasting models evaluated with expanding-window CV."""

import numpy as np
import pandas as pd

from metrics import compute_metrics


def run_baseline_cv(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    target_column: str = "price_adjusted",
    seasonal_period: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run naive and seasonal-naive baselines for all CV folds.

    Returns:
        A tuple containing:
        - Row-level predictions across all folds.
        - Aggregated metric summary by baseline model.
    """
    rows: list[dict] = []

    y = df[target_column]
    dates = df["date"]

    for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
        train = y.iloc[train_idx]
        # Why: keep baseline forecasts fixed per fold so they use only pre-test information.
        naive_value = train.iloc[-1] if len(train) >= 1 else np.nan
        seasonal_value = train.iloc[-seasonal_period] if len(train) >= seasonal_period else np.nan

        for step, idx in enumerate(test_idx, start=1):
            rows.append(
                {
                    "fold": fold_id,
                    "step": step,
                    "date": dates.iloc[idx],
                    "actual": y.iloc[idx],
                    "model": "baseline_naive",
                    "prediction": naive_value,
                }
            )
            rows.append(
                {
                    "fold": fold_id,
                    "step": step,
                    "date": dates.iloc[idx],
                    "actual": y.iloc[idx],
                    "model": "baseline_seasonal_naive",
                    "prediction": seasonal_value,
                }
            )

    predictions_df = pd.DataFrame(rows)
    summary_records: list[dict] = []
    for model_name, model_predictions in predictions_df.groupby("model"):
        metric_values = compute_metrics(model_predictions["actual"], model_predictions["prediction"])
        summary_records.append(
            {
                "model": model_name,
                "mae": metric_values["mae"],
                "rmse": metric_values["rmse"],
                "mape": metric_values["mape"],
                "directional_accuracy": metric_values["directional_accuracy"],
                "n_obs": metric_values["n_obs"],
                "n_folds": int(model_predictions["fold"].nunique()),
            }
        )
    summary_df = pd.DataFrame(summary_records)
    return predictions_df, summary_df
