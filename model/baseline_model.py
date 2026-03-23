from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


data_folder = Path("AdjustedData")
output_folder = Path("baseline-model-output")
output_folder.mkdir(exist_ok=True)

SARIMA_ORDERS = {
    "avocado": {"order": (1, 0, 1), "seasonal_order": (1, 1, 0, 12)},
    "tomato": {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)},
}


TARGET_COLUMN = "price_adjusted"


def load_adjusted_price(file_name):
    df = pd.read_csv(data_folder / file_name)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    return df.sort_values("date")

# calculate mae, rmse, mape
def calculate_metrics(actual, forecast):
    model_df = pd.DataFrame({"actual": actual, "forecast": forecast}).dropna().copy()
    model_df["error"] = model_df["actual"] - model_df["forecast"]

    mae = model_df["error"].abs().mean()
    rmse = np.sqrt((model_df["error"] ** 2).mean())
    mape = (model_df["error"].abs() / model_df["actual"]).mean() * 100
    return model_df, mae, rmse, mape

# model 0 (baseline model): naive
def naive_model(df):
    df = df.copy()
    df["naive_forecast"] = df[TARGET_COLUMN].shift(1)
    model_df, mae, rmse, mape = calculate_metrics(df[TARGET_COLUMN], df["naive_forecast"])
    return df, model_df, mae, rmse, mape

# model 1: sarima (no external indicators)
def sarima_model(df, product_name):
    df = df.copy()
    config = SARIMA_ORDERS[product_name]
    model = SARIMAX(
        df[TARGET_COLUMN],
        order=config["order"],
        seasonal_order=config["seasonal_order"],
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted_model = model.fit(disp=False)
    df["sarima_forecast"] = fitted_model.predict(start=0, end=len(df) - 1)
    model_df, mae, rmse, mape = calculate_metrics(df[TARGET_COLUMN], df["sarima_forecast"])
    return df, model_df, mae, rmse, mape

# ACF / PACF, ADF test
def run_diagnostics(df, product_name):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(df[TARGET_COLUMN].dropna(), lags=24, ax=axes[0])
    axes[0].set_title(f"{product_name.title()} Adjusted Price ACF (24 lags)")
    plot_pacf(df[TARGET_COLUMN].dropna(), lags=24, ax=axes[1], method="ywm")
    axes[1].set_title(f"{product_name.title()} Adjusted Price PACF (24 lags)")
    plt.tight_layout()
    plt.savefig(output_folder / f"{product_name}_acf_pacf.png", dpi=150)
    plt.close()

    adf_result = adfuller(df[TARGET_COLUMN].dropna(), autolag="AIC")
    return adf_result


def run_baseline(product_name, file_name):
    df = load_adjusted_price(file_name)
    df, _, _, _, _ = naive_model(df)
    df, _, _, _, _ = sarima_model(df, product_name)
    adf_result = run_diagnostics(df, product_name)
    comparison_df = df.dropna(subset=[TARGET_COLUMN, "naive_forecast", "sarima_forecast"]).copy()
    naive_df, naive_mae, naive_rmse, naive_mape = calculate_metrics(
        comparison_df[TARGET_COLUMN], comparison_df["naive_forecast"]
    )
    sarima_df, sarima_mae, sarima_rmse, sarima_mape = calculate_metrics(
        comparison_df[TARGET_COLUMN], comparison_df["sarima_forecast"]
    )

    summary = {
        "product": product_name,
        "comparison_observations": len(comparison_df),
        "naive_mae": naive_mae,
        "naive_rmse": naive_rmse,
        "naive_mape_pct": naive_mape,
        "sarima_mae": sarima_mae,
        "sarima_rmse": sarima_rmse,
        "sarima_mape_pct": sarima_mape,
        "adf_statistic": adf_result[0],
        "adf_pvalue": adf_result[1],
        "adf_critical_5pct": adf_result[4]["5%"]
    }

    df["product"] = product_name
    return df, summary


tomato_df, tomato_summary = run_baseline("tomato", "tomato_price_adjusted.csv")
avocado_df, avocado_summary = run_baseline("avocado", "avocado_price_adjusted.csv")

all_results = pd.concat([tomato_df, avocado_df], ignore_index=True)
all_results.to_csv(output_folder / "baseline_results.csv", index=False)

summary_df = pd.DataFrame([tomato_summary, avocado_summary])
summary_df.to_csv(output_folder / "baseline_summary.csv", index=False)
