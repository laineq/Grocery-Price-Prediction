"""Run baseline and SARIMA CV pipelines and export outputs."""
# to run py model/run_baseline_sarima.py

from pathlib import Path

import pandas as pd

from baseline import run_baseline_cv
from comparison_plots import generate_comparison_plots
from cv import create_expanding_window_folds
from data_preprocessing import load_price_data
from metrics import compute_metrics
from sarima import run_sarima_cv, save_diagnostics

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_DIR = PROJECT_ROOT / "AdjustedData"
OUTPUT_ROOT = CURRENT_DIR / "output"
PREDICTION_RESULT_DIR = PROJECT_ROOT / "prediction-result"

PRODUCT_FILES = {
    "tomato": "tomato_price_adjusted.csv",
    "avocado": "avocado_price_adjusted.csv",
}

INITIAL_WINDOW = 60
HORIZON = 1


def _build_dashboard_prediction_frame(model_df: pd.DataFrame) -> pd.DataFrame:
    """Build dashboard prediction output with simple residual-based bounds."""
    residuals = (model_df["actual"] - model_df["prediction"]).dropna()
    sigma = residuals.std(ddof=1) if len(residuals) > 1 else float("nan")
    # use historical residual spread as a lightweight uncertainty estimate for UI display.
    margin = 1.96 * sigma
    return pd.DataFrame(
        {
            "date": model_df["date"].dt.strftime("%Y-%m"),
            "prediction": model_df["prediction"],
            "lower_bound": model_df["prediction"] - margin,
            "upper_bound": model_df["prediction"] + margin,
        }
    )


def _export_dashboard_predictions(all_predictions_df: pd.DataFrame) -> None:
    """Export dashboard CSVs for baseline-naive and SARIMA-log models."""
    PREDICTION_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    export_df = all_predictions_df.copy()
    export_df["date"] = pd.to_datetime(export_df["date"], errors="coerce")
    export_df["actual"] = pd.to_numeric(export_df.get("actual"), errors="coerce")
    export_df["prediction"] = pd.to_numeric(export_df["prediction"], errors="coerce")
    export_df["lower_bound"] = pd.to_numeric(export_df.get("lower_bound"), errors="coerce")
    export_df["upper_bound"] = pd.to_numeric(export_df.get("upper_bound"), errors="coerce")
    products = sorted(export_df["product"].dropna().unique())

    for product_name in products:
        product_output_dir = PREDICTION_RESULT_DIR / product_name
        product_output_dir.mkdir(parents=True, exist_ok=True)

        product_df = export_df[export_df["product"] == product_name]

        naive_df = (
            product_df[product_df["model"] == "baseline_naive"][["date", "actual", "prediction"]]
            .dropna(subset=["date", "prediction"])
            .sort_values("date")
            .copy()
        )
        naive_out = _build_dashboard_prediction_frame(naive_df)
        naive_out.to_csv(product_output_dir / f"naive_{product_name}.csv", index=False)

        sarima_df = (
            product_df[product_df["model"] == "sarima_log"][["date", "actual", "prediction"]]
            .dropna(subset=["date", "prediction"])
            .sort_values("date")
            .copy()
        )
        sarima_out = _build_dashboard_prediction_frame(sarima_df)
        sarima_out.to_csv(product_output_dir / f"sarima_{product_name}.csv", index=False)


def _build_metric_row(model_name: str, actual: pd.Series, predicted: pd.Series) -> dict:
    """Build one standardized metric row from actual and predicted series."""
    metric_values = compute_metrics(actual, predicted)
    return {
        "model": model_name,
        "mae": metric_values["mae"],
        "rmse": metric_values["rmse"],
        "mape": metric_values["mape"],
        "directional_accuracy": metric_values["directional_accuracy"],
    }


def _export_final_metrics(products: list[str]) -> None:
    """Export per-product final metric table from existing prediction outputs."""
    for product_name in products:
        naive_predictions_path = OUTPUT_ROOT / product_name / "naive_sarima" / "predictions.csv"
        xgboost_model_1_path = OUTPUT_ROOT / product_name / "xgboost" / "model_1" / "predictions.csv"

        if not naive_predictions_path.exists():
            raise FileNotFoundError(f"Missing required file: {naive_predictions_path}")
        if not xgboost_model_1_path.exists():
            raise FileNotFoundError(f"Missing required file: {xgboost_model_1_path}")

        naive_df = pd.read_csv(naive_predictions_path)
        xgb_df = pd.read_csv(xgboost_model_1_path)

        baseline_df = naive_df[naive_df["model"] == "baseline_naive"].copy()
        sarima_log_df = naive_df[naive_df["model"] == "sarima_log"].copy()

        final_metrics_df = pd.DataFrame(
            [
                _build_metric_row(
                    "xgboost_model_1",
                    actual=pd.to_numeric(xgb_df["actual"], errors="coerce"),
                    predicted=pd.to_numeric(xgb_df["predicted"], errors="coerce"),
                ),
                _build_metric_row(
                    "sarima_log",
                    actual=pd.to_numeric(sarima_log_df["actual"], errors="coerce"),
                    predicted=pd.to_numeric(sarima_log_df["prediction"], errors="coerce"),
                ),
                _build_metric_row(
                    "baseline_naive",
                    actual=pd.to_numeric(baseline_df["actual"], errors="coerce"),
                    predicted=pd.to_numeric(baseline_df["prediction"], errors="coerce"),
                ),
            ],
            columns=["model", "mae", "rmse", "mape", "directional_accuracy"],
        )

        product_result_dir = PREDICTION_RESULT_DIR / product_name
        product_result_dir.mkdir(parents=True, exist_ok=True)
        final_metrics_df.to_csv(product_result_dir / f"{product_name}_final_metric.csv", index=False)


def run_product(product_name: str, file_name: str) -> pd.DataFrame:
    """Execute the full baseline + SARIMA workflow for one product."""
    product_output_dir = OUTPUT_ROOT / product_name
    product_output_dir.mkdir(parents=True, exist_ok=True)
    naive_sarima_output_dir = product_output_dir / "naive_sarima"
    naive_sarima_output_dir.mkdir(parents=True, exist_ok=True)

    df, quality_report = load_price_data(DATA_DIR, file_name, include_log=True)

    # diagnostics describe the full series and do not need recomputation per fold.
    diagnostics_rows = [
        save_diagnostics(df["price_adjusted"], naive_sarima_output_dir, "sarima_raw"),
        save_diagnostics(df["log_price"], naive_sarima_output_dir, "sarima_log"),
    ]

    folds = create_expanding_window_folds(
        n_samples=len(df),
        initial_window=INITIAL_WINDOW,
        horizon=HORIZON,
    )

    baseline_predictions, baseline_summary = run_baseline_cv(
        df=df,
        folds=folds,
        target_column="price_adjusted",
        seasonal_period=12,
    )

    sarima_raw_predictions, sarima_raw_summary = run_sarima_cv(
        df=df,
        folds=folds,
        product_name=product_name,
        use_log=False,
        target_column="price_adjusted",
        log_column="log_price",
    )

    sarima_log_predictions, sarima_log_summary = run_sarima_cv(
        df=df,
        folds=folds,
        product_name=product_name,
        use_log=True,
        target_column="price_adjusted",
        log_column="log_price",
    )

    predictions_df = pd.concat(
        [baseline_predictions, sarima_raw_predictions, sarima_log_predictions],
        ignore_index=True,
    )
    predictions_df.insert(0, "product", product_name)
    #  persist month-level keys to keep downstream joins stable across tools.
    predictions_df["date"] = pd.to_datetime(predictions_df["date"]).dt.strftime("%Y-%m")
    predictions_df.to_csv(naive_sarima_output_dir / "predictions.csv", index=False)

    summary_df = pd.concat([baseline_summary, sarima_raw_summary, sarima_log_summary], ignore_index=True)
    summary_df.insert(0, "product", product_name)
    summary_df["cv_initial_window"] = INITIAL_WINDOW
    summary_df["cv_horizon"] = HORIZON
    summary_df.to_csv(naive_sarima_output_dir / "summary_metrics.csv", index=False)

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    diagnostics_df.insert(0, "product", product_name)
    diagnostics_df.to_csv(naive_sarima_output_dir / "diagnostics_adf.csv", index=False)

    quality_df = pd.DataFrame([quality_report])
    quality_df.insert(0, "product", product_name)
    quality_df.to_csv(naive_sarima_output_dir / "data_quality_report.csv", index=False)

    generate_comparison_plots(OUTPUT_ROOT, product_name)
    return predictions_df


def main() -> None:
    """Run the pipeline for all configured products."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_predictions: list[pd.DataFrame] = []
    for product_name, file_name in PRODUCT_FILES.items():
        all_predictions.append(run_product(product_name, file_name))

    _export_dashboard_predictions(pd.concat(all_predictions, ignore_index=True))
    _export_final_metrics(sorted(PRODUCT_FILES.keys()))


if __name__ == "__main__":
    main()
