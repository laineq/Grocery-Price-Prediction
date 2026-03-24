from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
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


INPUT_FILE = PROJECT_ROOT / "xgboost-output" / "xgboost_merged_data.csv"
FEATURE_ENGINEERING_FOLDER = PROJECT_ROOT / "Feature-Engineering"
OUTPUT_ROOT = PROJECT_ROOT / "xgboost-output"
OUTPUT_ROOT.mkdir(exist_ok=True)

DATE_COLUMN = "date"
LAG_FEATURES = [1, 2, 3, 6, 12]
BASE_FEATURES = ["month"] + [f"lag{lag}" for lag in LAG_FEATURES]
SELECTIVE_TARGET_COLUMN = "log_price_adjusted"
SELECTIVE_PRICE_COLUMN = "price_adjusted"

PRODUCT_CONFIGS = {
    "avocado": {
        "merged_target_column": "log_avocado_price_adjusted",
        "merged_price_column": "avocado_price_adjusted",
        "merged_external_features": [
            "avocado_import_qty",
            "gas_price_integrated_gas_price",
            "xrate_adjusted_mxn_cad",
            "xrate_adjusted_usd_cad",
            "mexico_weather_adjusted_state_michoacán_mean_c",
            "mexico_weather_adjusted_state_michoacán_precipitation_mm",
        ],
        "selective_input_file": FEATURE_ENGINEERING_FOLDER / "avocado_final_selective_log.csv",
        "selective_feature_columns": [
            "import_qty_lag_2",
            "MEAN_C_lag_6",
            "PRECIPITATION_MM_lag_2",
            "MXN_CAD_lag_3",
            "USD_CAD_lag_2",
            "integrated_gas_price_lag_3",
        ],
        "model_4_feature_columns": [
            "lag1",
            "lag2",
            "lag12",
            "MEAN_C_lag_6",
            "integrated_gas_price_lag_3",
        ],
    },
    "tomato": {
        "merged_target_column": "log_tomato_price_adjusted",
        "merged_price_column": "tomato_price_adjusted",
        "merged_external_features": [
            "tomato_import_partner_mexico_qty",
            "tomato_import_partner_usa_qty",
            "gas_price_integrated_gas_price",
            "xrate_adjusted_mxn_cad",
            "xrate_adjusted_usd_cad",
            "mexico_weather_adjusted_state_michoacán_mean_c",
            "mexico_weather_adjusted_state_michoacán_precipitation_mm",
        ],
        "selective_input_file": FEATURE_ENGINEERING_FOLDER / "tomato_final_selective_log.csv",
        "selective_feature_columns": [
            "import_qty_lag_0",
            "MEAN_C_lag_3",
            "PRECIPITATION_MM_lag_5",
            "MXN_CAD_lag_0",
            "USD_CAD_lag_0",
            "integrated_gas_price_lag_0",
        ],
        "model_4_feature_columns": [
            "lag1",
            "lag2",
            "lag12",
            "MEAN_C_lag_3",
            "integrated_gas_price_lag_0",
        ],
    },
}


def mean_absolute_percentage_error(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def load_merged_data():
    df = pd.read_csv(INPUT_FILE)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format="%Y-%m")
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    return df


def load_selective_data(input_file):
    df = pd.read_csv(input_file)
    df = df.rename(columns={"Date": DATE_COLUMN})
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format="%Y-%m")
    df[SELECTIVE_TARGET_COLUMN] = np.log(df[SELECTIVE_PRICE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    return df


def add_target_lag_features(df, target_column):
    df = df.copy()
    for lag in LAG_FEATURES:
        df[f"lag{lag}"] = df[target_column].shift(lag)
    return df


def prepare_model_data(df, feature_columns, target_column, price_column):
    model_df = df[[DATE_COLUMN, price_column, target_column] + feature_columns].copy()
    model_df = model_df.dropna().reset_index(drop=True)
    return model_df


def time_split(df, train_ratio=0.8):
    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def train_model(train_df, feature_columns, target_column):
    model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(train_df[feature_columns], train_df[target_column])
    return model


def evaluate_model(model_name, model, test_df, feature_columns, target_column, price_column):
    result_df = test_df[[DATE_COLUMN, price_column, target_column]].copy()
    result_df[f"{model_name}_pred_log"] = model.predict(test_df[feature_columns])
    result_df[f"{model_name}_pred_price"] = np.exp(result_df[f"{model_name}_pred_log"])
    result_df["actual_price"] = result_df[price_column]
    result_df["actual_log_price"] = result_df[target_column]

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


def get_feature_importance_df(model, feature_columns):
    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    return importance_df


def print_feature_importance(product_name, model_name, importance_df):
    print(f"\nFeature importance for {product_name} - {model_name}:")
    for row in importance_df.itertuples(index=False):
        print(f"{row.feature}: {row.importance:.6f}")


def plot_feature_importance(product_name, model_name, importance_df, output_folder):
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"], importance_df["importance"], color="steelblue")
    plt.gca().invert_yaxis()
    plt.title(f"{product_name.title()} Feature Importance: {model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_folder / f"{model_name}_feature_importance.png", dpi=150)
    plt.close()


def plot_correlation_matrix(product_name, df, columns, output_folder):
    correlation_df = df[columns].dropna().corr()
    plt.figure(figsize=(12, 10))
    image = plt.imshow(correlation_df, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(correlation_df.columns)), correlation_df.columns, rotation=90)
    plt.yticks(range(len(correlation_df.index)), correlation_df.index)
    plt.title(f"{product_name.title()} Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_folder / "xgboost_correlation_matrix.png", dpi=150)
    plt.close()


def plot_predictions(product_name, prediction_frames, output_folder):
    if not prediction_frames:
        return

    plt.figure(figsize=(12, 6))
    first_predictions_df, _ = prediction_frames[0]
    actual_dates = first_predictions_df[DATE_COLUMN]
    actual_values = first_predictions_df["actual_price"]

    plt.plot(actual_dates, actual_values, label="Actual", linewidth=2, color="black")
    for predictions_df, model_name in prediction_frames:
        plt.plot(
            predictions_df[DATE_COLUMN],
            predictions_df[f"{model_name}_pred_price"],
            label=f"{model_name} Predicted",
            linewidth=2,
        )
    plt.title(f"{product_name.title()} Price Forecast: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price Adjusted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_folder / "xgboost_actual_vs_predicted.png", dpi=150)
    plt.close()


def run_product_models(product_name, config, merged_df):
    output_folder = OUTPUT_ROOT / product_name
    output_folder.mkdir(exist_ok=True)

    merged_with_lags_df = add_target_lag_features(merged_df, config["merged_target_column"])
    selective_df = add_target_lag_features(load_selective_data(config["selective_input_file"]), SELECTIVE_TARGET_COLUMN)

    model_specs = [
        {
            "model_name": "model_1_target_lags_only",
            "df": merged_with_lags_df,
            "feature_columns": BASE_FEATURES,
            "target_column": config["merged_target_column"],
            "price_column": config["merged_price_column"],
        },
        {
            "model_name": "model_2_target_lags_plus_external",
            "df": merged_with_lags_df,
            "feature_columns": BASE_FEATURES + config["merged_external_features"],
            "target_column": config["merged_target_column"],
            "price_column": config["merged_price_column"],
        },
        {
            "model_name": "model_3_selective_features_plus_target_lags",
            "df": selective_df,
            "feature_columns": [f"lag{lag}" for lag in LAG_FEATURES] + config["selective_feature_columns"],
            "target_column": SELECTIVE_TARGET_COLUMN,
            "price_column": SELECTIVE_PRICE_COLUMN,
        },
        {
            "model_name": "model_4_core_lags_mean_temp_gas",
            "df": selective_df,
            "feature_columns": config["model_4_feature_columns"],
            "target_column": SELECTIVE_TARGET_COLUMN,
            "price_column": SELECTIVE_PRICE_COLUMN,
        },
    ]

    metrics_rows = []
    prediction_frames = []
    comparison_df = None
    feature_importance_frames = []

    for spec in model_specs:
        model_name = spec["model_name"]
        feature_columns = spec["feature_columns"]
        target_column = spec["target_column"]
        price_column = spec["price_column"]

        model_df = prepare_model_data(
            spec["df"],
            feature_columns,
            target_column=target_column,
            price_column=price_column,
        )
        train_df, test_df = time_split(model_df)

        if train_df.empty or test_df.empty:
            raise ValueError(f"{product_name} - {model_name} does not have enough data after dropna().")

        model = train_model(train_df, feature_columns, target_column=target_column)
        predictions_df, metrics = evaluate_model(
            model_name,
            model,
            test_df,
            feature_columns,
            target_column=target_column,
            price_column=price_column,
        )
        metrics_rows.append(metrics)
        prediction_frames.append((predictions_df, model_name))

        importance_df = get_feature_importance_df(model, feature_columns)
        importance_df.insert(0, "model_name", model_name)
        feature_importance_frames.append(importance_df)
        print_feature_importance(product_name, model_name, importance_df[["feature", "importance"]])
        plot_feature_importance(
            product_name,
            model_name,
            importance_df[["feature", "importance"]],
            output_folder,
        )

        model_comparison_df = predictions_df[
            [
                DATE_COLUMN,
                "actual_price",
                "actual_log_price",
                f"{model_name}_pred_log",
                f"{model_name}_pred_price",
            ]
        ].rename(columns={"actual_log_price": target_column})

        if comparison_df is None:
            comparison_df = model_comparison_df
        else:
            comparison_df = comparison_df.merge(
                model_comparison_df[
                    [
                        DATE_COLUMN,
                        f"{model_name}_pred_log",
                        f"{model_name}_pred_price",
                    ]
                ],
                on=DATE_COLUMN,
                how="inner",
            )

    results_df = pd.DataFrame(metrics_rows)
    feature_importance_df = pd.concat(feature_importance_frames, ignore_index=True)

    comparison_df[DATE_COLUMN] = comparison_df[DATE_COLUMN].dt.strftime("%Y-%m")
    comparison_df.to_csv(output_folder / "xgboost_predictions.csv", index=False)
    results_df.to_csv(output_folder / "xgboost_metrics.csv", index=False)
    feature_importance_df.to_csv(output_folder / "xgboost_feature_importance.csv", index=False)

    plot_predictions(product_name, prediction_frames, output_folder)
    plot_correlation_matrix(
        product_name,
        selective_df,
        [SELECTIVE_TARGET_COLUMN] + [f"lag{lag}" for lag in LAG_FEATURES] + config["selective_feature_columns"],
        output_folder,
    )


def main():
    merged_df = load_merged_data()

    for product_name, config in PRODUCT_CONFIGS.items():
        run_product_models(product_name, config, merged_df)


if __name__ == "__main__":
    main()
