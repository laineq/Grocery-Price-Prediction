import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from metrics import compute_metrics


SARIMA_ORDERS = {
    "tomato": {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)},
    "avocado": {"order": (1, 0, 1), "seasonal_order": (1, 1, 0, 12)},
}


def save_diagnostics(
    series: pd.Series,
    output_dir: Path,
    model_label: str,
    lags: int = 24,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_series = series.dropna()

    diagnostics = {
        "model": model_label,
        "adf_statistic": np.nan,
        "adf_pvalue": np.nan,
        "adf_critical_1pct": np.nan,
        "adf_critical_5pct": np.nan,
        "adf_critical_10pct": np.nan,
        "adf_nobs": int(len(clean_series)),
    }

    if len(clean_series) >= 3:
        adf_result = adfuller(clean_series, autolag="AIC")
        diagnostics["adf_statistic"] = float(adf_result[0])
        diagnostics["adf_pvalue"] = float(adf_result[1])
        diagnostics["adf_critical_1pct"] = float(adf_result[4]["1%"])
        diagnostics["adf_critical_5pct"] = float(adf_result[4]["5%"])
        diagnostics["adf_critical_10pct"] = float(adf_result[4]["10%"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    max_pacf_lags = min(lags, max(1, (len(clean_series) // 2) - 1))
    max_acf_lags = min(lags, max(1, len(clean_series) - 1))

    if len(clean_series) >= 3 and max_pacf_lags >= 1 and max_acf_lags >= 1:
        plot_acf(clean_series, lags=max_acf_lags, ax=axes[0])
        axes[0].set_title(f"{model_label} ACF")
        plot_pacf(clean_series, lags=max_pacf_lags, ax=axes[1], method="ywm")
        axes[1].set_title(f"{model_label} PACF")
    else:
        axes[0].text(0.5, 0.5, "Not enough data for ACF", ha="center", va="center")
        axes[0].set_axis_off()
        axes[1].text(0.5, 0.5, "Not enough data for PACF", ha="center", va="center")
        axes[1].set_axis_off()

    plt.tight_layout()
    plt.savefig(output_dir / f"{model_label}_acf_pacf.png", dpi=150)
    plt.close(fig)

    return diagnostics


def run_sarima_cv(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    product_name: str,
    use_log: bool = False,
    target_column: str = "price_adjusted",
    log_column: str = "log_price",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = SARIMA_ORDERS[product_name]
    fit_column = log_column if use_log else target_column
    model_name = "sarima_log" if use_log else "sarima_raw"

    y_fit = df[fit_column]
    y_actual = df[target_column]
    dates = df["date"]

    rows: list[dict] = []

    for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
        train_series = y_fit.iloc[train_idx]
        forecast_values = np.full(len(test_idx), np.nan, dtype=float)

        if not train_series.isna().any():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(
                        train_series,
                        order=config["order"],
                        seasonal_order=config["seasonal_order"],
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fitted = model.fit(disp=False)
                    forecast_result = fitted.get_forecast(steps=len(test_idx))
                    forecast_values = np.asarray(forecast_result.predicted_mean, dtype=float)
            except Exception:
                forecast_values = np.full(len(test_idx), np.nan, dtype=float)

        for step, (idx, forecast_value) in enumerate(zip(test_idx, forecast_values), start=1):
            if use_log:
                prediction = np.exp(forecast_value) if not np.isnan(forecast_value) else np.nan
            else:
                prediction = forecast_value

            rows.append(
                {
                    "fold": fold_id,
                    "step": step,
                    "date": dates.iloc[idx],
                    "actual": y_actual.iloc[idx],
                    "model": model_name,
                    "prediction": prediction,
                    "lower_bound": np.nan,
                    "upper_bound": np.nan,
                }
            )

    predictions_df = pd.DataFrame(rows)
    metrics = compute_metrics(predictions_df["actual"], predictions_df["prediction"])
    summary_df = pd.DataFrame(
        [
            {
                "model": model_name,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "directional_accuracy": metrics["directional_accuracy"],
                "n_obs": metrics["n_obs"],
                "n_folds": int(predictions_df["fold"].nunique()),
            }
        ]
    )
    return predictions_df, summary_df
