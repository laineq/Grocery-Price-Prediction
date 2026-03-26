import numpy as np
import pandas as pd

from metrics import compute_metrics


def run_baseline_cv(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    target_column: str = "price_adjusted",
    seasonal_period: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    fold_metrics_rows: list[dict] = []

    y = df[target_column]
    dates = df["date"]

    for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
        train = y.iloc[train_idx]
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

        fold_actual = y.iloc[test_idx].reset_index(drop=True)
        naive_pred = pd.Series(np.full(len(test_idx), naive_value))
        seasonal_pred = pd.Series(np.full(len(test_idx), seasonal_value))

        naive_metrics = compute_metrics(fold_actual, naive_pred)
        fold_metrics_rows.append(
            {
                "fold": fold_id,
                "model": "baseline_naive",
                **naive_metrics,
            }
        )

        seasonal_metrics = compute_metrics(fold_actual, seasonal_pred)
        fold_metrics_rows.append(
            {
                "fold": fold_id,
                "model": "baseline_seasonal_naive",
                **seasonal_metrics,
            }
        )

    predictions_df = pd.DataFrame(rows)
    fold_metrics_df = pd.DataFrame(fold_metrics_rows)
    summary_df = (
        fold_metrics_df.groupby("model", as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mape=("mape", "mean"),
            n_obs=("n_obs", "mean"),
            n_folds=("fold", "nunique"),
        )
    )
    summary_df["n_obs"] = summary_df["n_obs"].round().astype("Int64")
    return predictions_df, summary_df
