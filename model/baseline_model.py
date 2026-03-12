from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


data_folder = Path("clean-data")
output_folder = Path("baseline-model-output")
output_folder.mkdir(exist_ok=True)

SARIMA_ORDERS = {
    "avocado": {"order": (1, 0, 1), "seasonal_order": (1, 1, 0, 12)},
    "tomato": {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)},
}


# Get monthly food CPI and calculate the 2017 whole-year average
cpi = pd.read_csv(data_folder / "cpi_20160101_cleaned.csv")
food_cpi = cpi[cpi["Product"] == "Food"].iloc[0]

cpi_rows = []
for column in cpi.columns:
    if column != "Product":
        cpi_rows.append(
            {
                "date": pd.to_datetime(column, format="%Y%m"),
                "food_cpi": float(food_cpi[column]),
            }
        )

food_cpi_df = pd.DataFrame(cpi_rows).sort_values("date")
base_cpi_2017 = food_cpi_df[
    (food_cpi_df["date"] >= "2017-01-01") & (food_cpi_df["date"] <= "2017-12-01")
]["food_cpi"].mean()

# Adjust price use (price / cpi) * base cpi (2017 as base)
def adjust_price(file_name):
    prices = pd.read_csv(data_folder / file_name)
    prices = prices[prices["GEO"] == "Canada"][["REF_DATE", "VALUE"]].copy()
    prices["date"] = pd.to_datetime(prices["REF_DATE"], format="%Y-%m")
    prices["price"] = pd.to_numeric(prices["VALUE"])
    prices = prices[["date", "price"]].sort_values("date")

    df = prices.merge(food_cpi_df, on="date", how="left")
    df["real_price"] = (df["price"] / df["food_cpi"]) * base_cpi_2017
    return df

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
    df["naive_forecast"] = df["real_price"].shift(1)
    model_df, mae, rmse, mape = calculate_metrics(df["real_price"], df["naive_forecast"])
    return df, model_df, mae, rmse, mape

# model 1: sarima (no external indicators)
def sarima_model(df, product_name):
    df = df.copy()
    config = SARIMA_ORDERS[product_name]
    model = SARIMAX(
        df["real_price"],
        order=config["order"],
        seasonal_order=config["seasonal_order"],
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted_model = model.fit(disp=False)
    df["sarima_forecast"] = fitted_model.predict(start=0, end=len(df) - 1)
    model_df, mae, rmse, mape = calculate_metrics(df["real_price"], df["sarima_forecast"])
    return df, model_df, mae, rmse, mape

# ACF / PACF, ADF test
def run_diagnostics(df, product_name):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(df["real_price"].dropna(), lags=24, ax=axes[0])
    axes[0].set_title(f"{product_name.title()} Real Price ACF (24 lags)")
    plot_pacf(df["real_price"].dropna(), lags=24, ax=axes[1], method="ywm")
    axes[1].set_title(f"{product_name.title()} Real Price PACF (24 lags)")
    plt.tight_layout()
    plt.savefig(output_folder / f"{product_name}_acf_pacf.png", dpi=150)
    plt.close()

    adf_result = adfuller(df["real_price"].dropna(), autolag="AIC")
    return adf_result


def run_baseline(product_name, file_name):
    df = adjust_price(file_name)
    df, _, _, _, _ = naive_model(df)
    df, _, _, _, _ = sarima_model(df, product_name)
    adf_result = run_diagnostics(df, product_name)
    comparison_df = df.dropna(subset=["real_price", "naive_forecast", "sarima_forecast"]).copy()
    naive_df, naive_mae, naive_rmse, naive_mape = calculate_metrics(
        comparison_df["real_price"], comparison_df["naive_forecast"]
    )
    sarima_df, sarima_mae, sarima_rmse, sarima_mape = calculate_metrics(
        comparison_df["real_price"], comparison_df["sarima_forecast"]
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


tomato_df, tomato_summary = run_baseline("tomato", "tomato_price.csv")
avocado_df, avocado_summary = run_baseline("avocado", "avocado_price.csv")

all_results = pd.concat([tomato_df, avocado_df], ignore_index=True)
all_results.to_csv(output_folder / "baseline_results.csv", index=False)

summary_df = pd.DataFrame([tomato_summary, avocado_summary])
summary_df.to_csv(output_folder / "baseline_summary.csv", index=False)
