from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# py model/xgboost.py
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from xgboost import XGBRegressor
except ImportError:
    sys.modules.pop("xgboost", None)
    fallback_paths = [Path.home() / "miniconda3" / "Lib" / "site-packages"]
    for fallback_path in fallback_paths:
        if (fallback_path / "xgboost").exists():
            sys.path.insert(0, str(fallback_path))
            break
    from xgboost import XGBRegressor

from cv import create_expanding_window_folds
from metrics import compute_metrics
from comparison_plots import generate_comparison_plots

DATE_COLUMN = "date"
LAG_FEATURES = [1, 2, 3, 6, 12]
TARGET_LAG_FEATURE_COLUMNS = [f"lag{lag}" for lag in LAG_FEATURES]
SELECTIVE_TARGET_COLUMN = "log_price_adjusted"
SELECTIVE_PRICE_COLUMN = "price_adjusted"
INITIAL_WINDOW = 60
HORIZON = 1

OUTPUT_ROOT = CURRENT_DIR / "output"
PREDICTION_RESULT_DIR = PROJECT_ROOT / "prediction-result"
ADJUSTED_DATA_DIR = PROJECT_ROOT / "AdjustedData"
FEATURE_ENGINEERING_DIR = PROJECT_ROOT / "Feature-Engineering"

PRODUCT_CONFIGS = {
    "avocado": {
        "source_a_price_file": ADJUSTED_DATA_DIR / "avocado_price_adjusted.csv",
        "source_a_import_file": ADJUSTED_DATA_DIR / "avocado_import.csv",
        "source_a_price_column": "avocado_price_adjusted",
        "source_a_target_column": "log_avocado_price_adjusted",
        "source_a_external_features": [
            "import_qty",
            "integrated_gas_price",
            "MXN_CAD",
            "USD_CAD",
            "MEAN_C",
            "PRECIPITATION_MM",
        ],
        "selective_input_file": FEATURE_ENGINEERING_DIR / "avocado_final_selective_log.csv",
        "model_4_feature_columns": [
            "MEAN_C_lag_0",
            "PRECIPITATION_MM_lag_5",
        ],
    },
    "tomato": {
        "source_a_price_file": ADJUSTED_DATA_DIR / "tomato_price_adjusted.csv",
        "source_a_import_file": ADJUSTED_DATA_DIR / "tomato_import.csv",
        "source_a_price_column": "tomato_price_adjusted",
        "source_a_target_column": "log_tomato_price_adjusted",
        "source_a_external_features": [
            "tomato_import_partner_mexico_qty",
            "tomato_import_partner_usa_qty",
            "integrated_gas_price",
            "MXN_CAD",
            "USD_CAD",
            "MEAN_C",
            "PRECIPITATION_MM",
        ],
        "selective_input_file": FEATURE_ENGINEERING_DIR / "tomato_final_selective_log.csv",
        "model_4_feature_columns": [
            "MEAN_C_lag_2",
            "PRECIPITATION_MM_lag_3",
            "import_qty_lag_3",
        ],
    },
}


def _train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def _load_source_a_merged_dataset(product_name: str, config: dict) -> pd.DataFrame:
    price_df = pd.read_csv(config["source_a_price_file"])
    price_df = price_df.rename(columns={"date": DATE_COLUMN, "price_adjusted": config["source_a_price_column"]})
    price_df[DATE_COLUMN] = pd.to_datetime(price_df[DATE_COLUMN], errors="coerce")
    price_df[config["source_a_target_column"]] = np.log(price_df[config["source_a_price_column"]])
    price_df = price_df[[DATE_COLUMN, config["source_a_price_column"], config["source_a_target_column"]]]

    import_df = pd.read_csv(config["source_a_import_file"])
    import_df = import_df.rename(columns={"date": DATE_COLUMN})
    import_df[DATE_COLUMN] = pd.to_datetime(import_df[DATE_COLUMN], errors="coerce")

    if product_name == "tomato":
        import_df["partner"] = import_df["partner"].astype(str).str.strip().str.lower()
        import_df = (
            import_df.pivot_table(index=DATE_COLUMN, columns="partner", values="qty", aggfunc="sum")
            .reset_index()
            .rename(
                columns={
                    "mexico": "tomato_import_partner_mexico_qty",
                    "usa": "tomato_import_partner_usa_qty",
                }
            )
        )
    else:
        import_df = import_df[[DATE_COLUMN, "qty"]].rename(columns={"qty": "import_qty"})

    gas_df = pd.read_csv(ADJUSTED_DATA_DIR / "gas_price.csv")
    gas_df = gas_df.rename(columns={"date": DATE_COLUMN})
    gas_df[DATE_COLUMN] = pd.to_datetime(gas_df[DATE_COLUMN], errors="coerce")
    gas_df = gas_df[[DATE_COLUMN, "integrated_gas_price"]]

    xrate_df = pd.read_csv(ADJUSTED_DATA_DIR / "xrate_adjusted.csv")
    xrate_df = xrate_df.rename(columns={"date": DATE_COLUMN})
    xrate_df[DATE_COLUMN] = pd.to_datetime(xrate_df[DATE_COLUMN], errors="coerce")
    xrate_df = xrate_df[[DATE_COLUMN, "MXN_CAD", "USD_CAD"]]

    weather_df = pd.read_csv(ADJUSTED_DATA_DIR / "mexico_weather_adjusted.csv")
    weather_df = weather_df.rename(columns={"date": DATE_COLUMN})
    weather_df[DATE_COLUMN] = pd.to_datetime(weather_df[DATE_COLUMN], errors="coerce")
    weather_df["STATE_normalized"] = (
        weather_df["STATE"]
        .astype(str)
        .str.normalize("NFKD")
        .str.encode("ascii", "ignore")
        .str.decode("ascii")
        .str.lower()
    )
    weather_df = weather_df[weather_df["STATE_normalized"] == "michoacan"][[DATE_COLUMN, "MEAN_C", "PRECIPITATION_MM"]]

    out_df = (
        price_df
        .merge(import_df, on=DATE_COLUMN, how="left")
        .merge(gas_df, on=DATE_COLUMN, how="left")
        .merge(xrate_df, on=DATE_COLUMN, how="left")
        .merge(weather_df, on=DATE_COLUMN, how="left")
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )
    out_df["month"] = out_df[DATE_COLUMN].dt.month
    return out_df


def _load_source_b_selective_dataset(config: dict) -> pd.DataFrame:
    df = pd.read_csv(config["selective_input_file"])
    df = df.rename(columns={"Date": DATE_COLUMN})
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df[SELECTIVE_TARGET_COLUMN] = np.log(df[SELECTIVE_PRICE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    return df


def _add_target_lags(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    out = df.copy()
    for lag in LAG_FEATURES:
        out[f"lag{lag}"] = out[target_column].shift(lag)
    return out


def _resolve_feature_columns(
    df: pd.DataFrame,
    preferred_columns: list[str],
    context: str,
    *,
    raise_if_empty: bool = True,
) -> list[str]:
    available = [col for col in preferred_columns if col in df.columns]
    missing = [col for col in preferred_columns if col not in df.columns]
    if missing:
        print(f"[xgboost] {context}: missing required columns: {missing}")
    if not available and raise_if_empty:
        raise ValueError(f"{context}: no feature columns available from preferred list.")
    return available


def _derive_selective_feature_pool(df: pd.DataFrame) -> list[str]:
    excluded_columns = {
        DATE_COLUMN,
        "Date",
        SELECTIVE_PRICE_COLUMN,
        SELECTIVE_TARGET_COLUMN,
    }
    return [col for col in df.columns if col not in excluded_columns]


def _build_model_specs(product_name: str, config: dict, source_a_df: pd.DataFrame, source_b_df: pd.DataFrame) -> list[dict]:
    source_a_external_features = _resolve_feature_columns(
        source_a_df,
        config["source_a_external_features"],
        context=f"{product_name} model_2 source_a external features",
    )

    source_b_feature_pool = _derive_selective_feature_pool(source_b_df)
    if not source_b_feature_pool:
        raise ValueError(f"{product_name} model_3 source_b features: none detected from selective input.")

    model_3_features = TARGET_LAG_FEATURE_COLUMNS + [
        col for col in source_b_feature_pool if col not in TARGET_LAG_FEATURE_COLUMNS
    ]

    model_4_features = _resolve_feature_columns(
        source_b_df,
        config["model_4_feature_columns"],
        context=f"{product_name} model_4 source_b selected features",
        raise_if_empty=True,
    )
    missing_model_4 = [col for col in config["model_4_feature_columns"] if col not in model_4_features]
    if missing_model_4:
        raise ValueError(
            f"{product_name} model_4: missing required selected features: {missing_model_4}"
        )

    return [
        {
            "model_variant": "model_1",
            "model_name": "model_1_target_lags_only",
            "model_label": "baseline",
            "feature_lag_label": "only target lag",
            "df": source_a_df,
            "target_column": config["source_a_target_column"],
            "price_column": config["source_a_price_column"],
            "feature_columns": ["month"] + TARGET_LAG_FEATURE_COLUMNS,
        },
        {
            "model_variant": "model_2",
            "model_name": "model_2_target_lags_plus_external",
            "model_label": "+ external",
            "feature_lag_label": "target lag + external (non-lagged)",
            "df": source_a_df,
            "target_column": config["source_a_target_column"],
            "price_column": config["source_a_price_column"],
            "feature_columns": ["month"] + TARGET_LAG_FEATURE_COLUMNS + source_a_external_features,
        },
        {
            "model_variant": "model_3",
            "model_name": "model_3_full_selective_with_target_lags",
            "model_label": "full feature engineering",
            "feature_lag_label": "all selective csv features + lagged target",
            "df": source_b_df,
            "target_column": SELECTIVE_TARGET_COLUMN,
            "price_column": SELECTIVE_PRICE_COLUMN,
            "feature_columns": model_3_features,
        },
        {
            "model_variant": "model_4",
            "model_name": "model_4_reduced_selective_features",
            "model_label": "core feature selection",
            "feature_lag_label": "selected external features only",
            "df": source_b_df,
            "target_column": SELECTIVE_TARGET_COLUMN,
            "price_column": SELECTIVE_PRICE_COLUMN,
            "feature_columns": model_4_features,
        },
    ]


def _prepare_model_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    price_column: str,
) -> pd.DataFrame:
    keep_cols = [DATE_COLUMN, price_column, target_column] + feature_columns
    model_df = df[keep_cols].copy().dropna().reset_index(drop=True)
    return model_df


def _run_expanding_window_cv(
    model_df: pd.DataFrame,
    model_name: str,
    feature_columns: list[str],
    target_column: str,
    price_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = create_expanding_window_folds(
        n_samples=len(model_df),
        initial_window=INITIAL_WINDOW,
        horizon=HORIZON,
    )

    cv_rows: list[dict] = []
    feature_importance_rows: list[dict] = []

    for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
        train_df = model_df.iloc[train_idx]
        test_df = model_df.iloc[test_idx]

        model = _train_model(train_df[feature_columns], train_df[target_column])
        pred_log = model.predict(test_df[feature_columns])
        pred_price = np.exp(pred_log)

        for idx, pred in zip(test_idx, pred_price):
            cv_rows.append(
                {
                    "fold": fold_id,
                    "date": model_df.iloc[idx][DATE_COLUMN],
                    "actual": model_df.iloc[idx][price_column],
                    "predicted": float(pred),
                }
            )

        fold_importance = pd.DataFrame(
            {
                "fold": fold_id,
                "feature": feature_columns,
                "importance": model.feature_importances_,
            }
        )
        feature_importance_rows.append(fold_importance)

    cv_predictions_df = pd.DataFrame(cv_rows).sort_values(["date", "fold"]).reset_index(drop=True)
    predictions_df = (
        cv_predictions_df.groupby("date", as_index=False)
        .agg(actual=("actual", "mean"), predicted=("predicted", "mean"))
        .sort_values("date")
        .reset_index(drop=True)
    )

    metric_values = compute_metrics(predictions_df["actual"], predictions_df["predicted"])
    metrics_df = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "mae": metric_values["mae"],
                "rmse": metric_values["rmse"],
                "mape": metric_values["mape"],
                "directional_accuracy": metric_values["directional_accuracy"],
            }
        ]
    )

    feature_importance_df = (
        pd.concat(feature_importance_rows, ignore_index=True)
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return cv_predictions_df, predictions_df, metrics_df, feature_importance_df


def _save_importance_plot(feature_importance_df: pd.DataFrame, output_dir: Path, product_name: str, model_name: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df["feature"], feature_importance_df["importance"], color="steelblue")
    plt.gca().invert_yaxis()
    plt.title(f"{product_name.title()} {model_name} Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "importance_plot.png", dpi=150)
    plt.close()


def _save_actual_vs_predicted_plot(predictions_df: pd.DataFrame, output_dir: Path, product_name: str, model_name: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(predictions_df[DATE_COLUMN], predictions_df["actual"], label="Actual", linewidth=2, color="black")
    plt.plot(predictions_df[DATE_COLUMN], predictions_df["predicted"], label="Predicted", linewidth=2)
    plt.title(f"{product_name.title()} {model_name}: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()


def _save_outputs(
    product_name: str,
    model_variant: str,
    model_name: str,
    model_label: str,
    feature_lag_label: str,
    cv_predictions_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
) -> None:
    output_dir = OUTPUT_ROOT / product_name / "xgboost" / model_variant
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_out = metrics_df.copy()
    metrics_out.insert(0, "product", product_name)
    metrics_out.insert(1, "xgboost_model", f"xg-boost {model_variant}")
    metrics_out.insert(2, "label_model", model_label)
    metrics_out.insert(3, "feature_lag", feature_lag_label)
    metrics_out.to_csv(output_dir / "metrics.csv", index=False)

    predictions_out = predictions_df.copy()
    predictions_out[DATE_COLUMN] = pd.to_datetime(predictions_out[DATE_COLUMN]).dt.strftime("%Y-%m")
    predictions_out.to_csv(output_dir / "predictions.csv", index=False)

    cv_out = cv_predictions_df.copy()
    cv_out[DATE_COLUMN] = pd.to_datetime(cv_out[DATE_COLUMN]).dt.strftime("%Y-%m")
    cv_out.to_csv(output_dir / "cv_predictions.csv", index=False)

    feature_importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    _save_importance_plot(feature_importance_df, output_dir, product_name, model_name)
    _save_actual_vs_predicted_plot(predictions_df, output_dir, product_name, model_name)


def _export_dashboard_model_1_predictions(
    product_name: str,
    predictions_df: pd.DataFrame,
) -> None:
    product_output_dir = PREDICTION_RESULT_DIR / product_name
    product_output_dir.mkdir(parents=True, exist_ok=True)

    export_df = predictions_df.copy()
    export_df[DATE_COLUMN] = pd.to_datetime(export_df[DATE_COLUMN], errors="coerce")
    export_df["actual"] = pd.to_numeric(export_df["actual"], errors="coerce")
    export_df["predicted"] = pd.to_numeric(export_df["predicted"], errors="coerce")
    export_df = export_df.dropna(subset=[DATE_COLUMN, "predicted"]).sort_values(DATE_COLUMN).reset_index(drop=True)

    residuals = (export_df["actual"] - export_df["predicted"]).dropna()
    sigma = residuals.std(ddof=1) if len(residuals) > 1 else float("nan")
    margin = 1.96 * sigma

    dashboard_out = pd.DataFrame(
        {
            "date": export_df[DATE_COLUMN].dt.strftime("%Y-%m"),
            "prediction": export_df["predicted"],
            "lower_bound": export_df["predicted"] - margin,
            "upper_bound": export_df["predicted"] + margin,
        }
    )
    dashboard_out.to_csv(product_output_dir / f"xgboost_model_1_{product_name}.csv", index=False)


def run_product_models(product_name: str, config: dict) -> None:
    source_a_df = _load_source_a_merged_dataset(product_name, config)
    source_a_df = _add_target_lags(source_a_df, config["source_a_target_column"])

    source_b_df = _load_source_b_selective_dataset(config)
    source_b_df = _add_target_lags(source_b_df, SELECTIVE_TARGET_COLUMN)

    model_specs = _build_model_specs(product_name, config, source_a_df, source_b_df)
    metrics_rows: list[pd.DataFrame] = []

    for spec in model_specs:
        model_df = _prepare_model_frame(
            df=spec["df"],
            feature_columns=spec["feature_columns"],
            target_column=spec["target_column"],
            price_column=spec["price_column"],
        )

        if len(model_df) < (INITIAL_WINDOW + HORIZON):
            raise ValueError(
                f"{product_name} {spec['model_variant']} does not have enough rows after preprocessing "
                f"({len(model_df)} rows, need at least {INITIAL_WINDOW + HORIZON})."
            )

        cv_predictions_df, predictions_df, metrics_df, feature_importance_df = _run_expanding_window_cv(
            model_df=model_df,
            model_name=spec["model_name"],
            feature_columns=spec["feature_columns"],
            target_column=spec["target_column"],
            price_column=spec["price_column"],
        )

        _save_outputs(
            product_name=product_name,
            model_variant=spec["model_variant"],
            model_name=spec["model_name"],
            model_label=spec["model_label"],
            feature_lag_label=spec["feature_lag_label"],
            cv_predictions_df=cv_predictions_df,
            predictions_df=predictions_df,
            metrics_df=metrics_df,
            feature_importance_df=feature_importance_df,
        )
        if spec["model_variant"] == "model_1":
            _export_dashboard_model_1_predictions(
                product_name=product_name,
                predictions_df=predictions_df,
            )
        metrics_row = metrics_df.copy()
        metrics_row.insert(0, "model_variant", spec["model_variant"])
        metrics_row.insert(1, "xgboost_model", f"xg-boost {spec['model_variant']}")
        metrics_row.insert(2, "label_model", spec["model_label"])
        metrics_row.insert(3, "feature_lag", spec["feature_lag_label"])
        metrics_rows.append(metrics_row)

    if metrics_rows:
        product_metrics_df = pd.concat(metrics_rows, ignore_index=True)
        product_metrics_df.insert(0, "product", product_name)
        product_metrics_df = product_metrics_df[
            ["product", "xgboost_model", "feature_lag", "mae", "rmse", "mape", "directional_accuracy"]
        ]
        product_metrics_path = OUTPUT_ROOT / product_name / "xgboost" / "metrics.csv"
        product_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        product_metrics_df.to_csv(product_metrics_path, index=False)

    generate_comparison_plots(OUTPUT_ROOT, product_name)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for product_name, config in PRODUCT_CONFIGS.items():
        run_product_models(product_name, config)


if __name__ == "__main__":
    main()
