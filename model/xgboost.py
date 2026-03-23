from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
CURRENT_DIR = Path(__file__).resolve().parent
sys.path = [
    path
    for path in sys.path
    if Path(path or ".").resolve() != CURRENT_DIR
]

try:
    from xgboost import XGBRegressor
except ImportError:
    sys.modules.pop("xgboost", None)
    fallback_paths = [
        Path.home() / "miniconda3" / "Lib" / "site-packages",
    ]
    for fallback_path in fallback_paths:
        if (fallback_path / "xgboost").exists():
            sys.path.insert(0, str(fallback_path))
            break
    from xgboost import XGBRegressor


INPUT_FILE = Path("xgboost-output/xgboost_merged_data.csv")
OUTPUT_FOLDER = Path("xgboost-output")
OUTPUT_FOLDER.mkdir(exist_ok=True)

DATE_COLUMN = "date"
TARGET_COLUMN = "log_avocado_price_adjusted"
PRICE_COLUMN = "avocado_price_adjusted"
LAG_FEATURES = [1, 2, 3, 6, 12]
BASE_FEATURES = ["month"] + [f"lag{lag}" for lag in LAG_FEATURES]
EXTERNAL_FEATURES = [
    "avocado_import_qty",
    "gas_price_integrated_gas_price",
    "xrate_adjusted_mxn_cad",
    "xrate_adjusted_usd_cad",
    "mexico_weather_adjusted_state_michoacán_mean_c",
    "mexico_weather_adjusted_state_michoacán_precipitation_mm",
]


def mean_absolute_percentage_error(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def load_data():
    df = pd.read_csv(INPUT_FILE)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format="%Y-%m")
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    return df


def add_lag_features(df):
    df = df.copy()
    for lag in LAG_FEATURES:
        df[f"lag{lag}"] = df[TARGET_COLUMN].shift(lag)
    return df


def prepare_model_data(df, feature_columns):
    model_df = df[[DATE_COLUMN, PRICE_COLUMN, TARGET_COLUMN] + feature_columns].copy()
    model_df = model_df.dropna().reset_index(drop=True)
    return model_df


def time_split(df, train_ratio=0.8):
    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def train_model(train_df, feature_columns):
    model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(train_df[feature_columns], train_df[TARGET_COLUMN])
    return model


def evaluate_model(model_name, model, test_df, feature_columns):
    result_df = test_df[[DATE_COLUMN, PRICE_COLUMN, TARGET_COLUMN]].copy()
    result_df[f"{model_name}_pred_log"] = model.predict(test_df[feature_columns])
    result_df[f"{model_name}_pred_price"] = np.exp(result_df[f"{model_name}_pred_log"])
    result_df["actual_price"] = result_df[PRICE_COLUMN]

    mae = mean_absolute_error(result_df["actual_price"], result_df[f"{model_name}_pred_price"])
    rmse = np.sqrt(
        mean_squared_error(result_df["actual_price"], result_df[f"{model_name}_pred_price"])
    )
    mape = mean_absolute_percentage_error(
        result_df["actual_price"], result_df[f"{model_name}_pred_price"]
    )

    metrics = {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }
    return result_df, metrics


def plot_predictions(prediction_frames):
    plt.figure(figsize=(12, 6))
    actual_dates = prediction_frames[0][DATE_COLUMN]
    actual_values = prediction_frames[0]["actual_price"]

    plt.plot(actual_dates, actual_values, label="Actual", linewidth=2, color="black")
    plt.plot(
        prediction_frames[0][DATE_COLUMN],
        prediction_frames[0]["model_1_target_lags_only_pred_price"],
        label="Model 1 Predicted",
        linewidth=2,
    )
    plt.plot(
        prediction_frames[1][DATE_COLUMN],
        prediction_frames[1]["model_2_target_lags_plus_external_pred_price"],
        label="Model 2 Predicted",
        linewidth=2,
    )
    plt.title("Avocado Price Forecast: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Avocado Price Adjusted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "avocado_xgboost_actual_vs_predicted.png", dpi=150)
    plt.close()


def main():
    df = load_data()
    df = add_lag_features(df)

    model_specs = [
        ("model_1_target_lags_only", BASE_FEATURES),
        ("model_2_target_lags_plus_external", BASE_FEATURES + EXTERNAL_FEATURES),
    ]

    metrics_rows = []
    prediction_frames = []

    for model_name, feature_columns in model_specs:
        model_df = prepare_model_data(df, feature_columns)
        train_df, test_df = time_split(model_df)

        if train_df.empty or test_df.empty:
            raise ValueError(f"{model_name} does not have enough data after dropna().")

        model = train_model(train_df, feature_columns)
        predictions_df, metrics = evaluate_model(model_name, model, test_df, feature_columns)
        metrics_rows.append(metrics)
        prediction_frames.append(predictions_df)

    comparison_df = prediction_frames[0][
        [DATE_COLUMN, "actual_price", "log_avocado_price_adjusted", "model_1_target_lags_only_pred_log", "model_1_target_lags_only_pred_price"]
    ].merge(
        prediction_frames[1][
            [DATE_COLUMN, "model_2_target_lags_plus_external_pred_log", "model_2_target_lags_plus_external_pred_price"]
        ],
        on=DATE_COLUMN,
        how="inner",
    )

    results_df = pd.DataFrame(metrics_rows)

    comparison_df[DATE_COLUMN] = comparison_df[DATE_COLUMN].dt.strftime("%Y-%m")
    comparison_df.to_csv(OUTPUT_FOLDER / "avocado_xgboost_predictions.csv", index=False)
    results_df.to_csv(OUTPUT_FOLDER / "avocado_xgboost_metrics.csv", index=False)

    plot_predictions(prediction_frames)


if __name__ == "__main__":
    main()
