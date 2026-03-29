from pathlib import Path

import pandas as pd

from baseline import run_baseline_cv
from comparison_plots import generate_comparison_plots
from cv import create_expanding_window_folds
from data_preprocessing import load_price_data
from sarima import run_sarima_cv, save_diagnostics
#py model/run_baseline_sarima.py

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


def _export_dashboard_predictions(all_predictions_df: pd.DataFrame) -> None:
    PREDICTION_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    model_exports = {
        "naive_baseline_predictions.csv": "baseline_naive",
        "sarima_no_external_indicator_predictions.csv": "sarima_raw",
    }

    export_df = all_predictions_df.copy()
    export_df["date"] = pd.to_datetime(export_df["date"], errors="coerce")
    export_df["prediction"] = pd.to_numeric(export_df["prediction"], errors="coerce")
    products = sorted(export_df["product"].dropna().unique())

    for product_name in products:
        product_output_dir = PREDICTION_RESULT_DIR / product_name
        product_output_dir.mkdir(parents=True, exist_ok=True)

        product_df = export_df[export_df["product"] == product_name]
        for output_file, model_name in model_exports.items():
            model_df = (
                product_df[product_df["model"] == model_name][["date", "prediction"]]
                .dropna(subset=["date", "prediction"])
                .sort_values("date")
            )
            model_df["date"] = model_df["date"].dt.strftime("%Y-%m")
            model_df.to_csv(product_output_dir / output_file, index=False)


def run_product(product_name: str, file_name: str) -> pd.DataFrame:
    product_output_dir = OUTPUT_ROOT / product_name
    product_output_dir.mkdir(parents=True, exist_ok=True)

    df, quality_report = load_price_data(DATA_DIR, file_name, include_log=True)

    # Run diagnostics once at the beginning (not per CV fold).
    diagnostics_rows = [
        save_diagnostics(df["price_adjusted"], product_output_dir, "sarima_raw"),
        save_diagnostics(df["log_price"], product_output_dir, "sarima_log"),
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
    predictions_df["date"] = pd.to_datetime(predictions_df["date"]).dt.strftime("%Y-%m")
    predictions_df.to_csv(product_output_dir / "predictions.csv", index=False)

    summary_df = pd.concat([baseline_summary, sarima_raw_summary, sarima_log_summary], ignore_index=True)
    summary_df.insert(0, "product", product_name)
    summary_df["cv_initial_window"] = INITIAL_WINDOW
    summary_df["cv_horizon"] = HORIZON
    summary_df.to_csv(product_output_dir / "summary_metrics.csv", index=False)

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    diagnostics_df.insert(0, "product", product_name)
    diagnostics_df.to_csv(product_output_dir / "diagnostics_adf.csv", index=False)

    quality_df = pd.DataFrame([quality_report])
    quality_df.insert(0, "product", product_name)
    quality_df.to_csv(product_output_dir / "data_quality_report.csv", index=False)

    generate_comparison_plots(OUTPUT_ROOT, product_name)
    return predictions_df


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_predictions: list[pd.DataFrame] = []
    for product_name, file_name in PRODUCT_FILES.items():
        all_predictions.append(run_product(product_name, file_name))

    _export_dashboard_predictions(pd.concat(all_predictions, ignore_index=True))


if __name__ == "__main__":
    main()
