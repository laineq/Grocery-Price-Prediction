"""Create baseline and XGBoost comparison charts against actual values."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

DATE_COLUMN = "date"
ACTUAL_COLUMN = "actual"

BASELINE_MODEL_LABELS = {
    "baseline_naive": "Baseline Naive",
    "baseline_seasonal_naive": "Seasonal Naive",
    "sarima_raw": "SARIMA (no external indicators)",
    "sarima_log": "SARIMA (log-transformed)",
}


def _coerce_date_index(df: pd.DataFrame, date_column: str = DATE_COLUMN) -> pd.DatetimeIndex:
    """Parse the date column to a DatetimeIndex-compatible series."""
    return pd.to_datetime(df[date_column], errors="coerce")


def _build_aligned_frame_from_series(series_by_name: dict[str, pd.Series], actual_series: pd.Series) -> pd.DataFrame:
    """Align actual and model prediction series on a shared datetime index."""
    all_indices = set(actual_series.index)
    for series in series_by_name.values():
        all_indices.update(series.index)

    # use union of dates so models with partial coverage stay comparable on one timeline.
    aligned_index = pd.DatetimeIndex(sorted(all_indices))
    aligned = pd.DataFrame(index=aligned_index)
    aligned[ACTUAL_COLUMN] = actual_series.reindex(aligned_index)

    for series_name, series in series_by_name.items():
        aligned[series_name] = series.reindex(aligned_index)

    return aligned.sort_index()


def _load_baseline_vs_actual_frame(product_output_dir: Path) -> pd.DataFrame | None:
    """Load and align baseline predictions for plotting."""
    baseline_path = product_output_dir / "naive_sarima" / "predictions.csv"
    if not baseline_path.exists():
        return None

    baseline_df = pd.read_csv(baseline_path)
    if baseline_df.empty:
        return None

    baseline_df[DATE_COLUMN] = _coerce_date_index(baseline_df, DATE_COLUMN)
    baseline_df = baseline_df.dropna(subset=[DATE_COLUMN]).copy()
    baseline_df = baseline_df[baseline_df["model"].isin(BASELINE_MODEL_LABELS)].copy()
    if baseline_df.empty:
        return None

    actual_by_date = baseline_df.groupby(DATE_COLUMN)[ACTUAL_COLUMN].mean()

    model_series: dict[str, pd.Series] = {}
    for raw_model_name, display_name in BASELINE_MODEL_LABELS.items():
        model_slice = baseline_df[baseline_df["model"] == raw_model_name]
        if model_slice.empty:
            continue
        model_series[display_name] = model_slice.groupby(DATE_COLUMN)["prediction"].mean()

    if not model_series:
        return None

    return _build_aligned_frame_from_series(model_series, actual_by_date)


def _load_xgboost_vs_actual_frame(product_output_dir: Path) -> pd.DataFrame | None:
    """Load and align XGBoost model predictions for plotting."""
    xgboost_root = product_output_dir / "xgboost"
    if not xgboost_root.exists():
        return None

    prediction_files = sorted(xgboost_root.glob("model_*/predictions.csv"))
    if not prediction_files:
        return None

    actual_by_date = pd.Series(dtype=float)
    model_series: dict[str, pd.Series] = {}

    for prediction_file in prediction_files:
        model_variant = prediction_file.parent.name
        model_label = f"XGBoost {model_variant}"

        model_df = pd.read_csv(prediction_file)
        if model_df.empty:
            continue

        model_df[DATE_COLUMN] = _coerce_date_index(model_df, DATE_COLUMN)
        model_df = model_df.dropna(subset=[DATE_COLUMN]).copy()
        if model_df.empty or "predicted" not in model_df.columns:
            continue

        grouped = model_df.groupby(DATE_COLUMN).agg(actual=("actual", "mean"), predicted=("predicted", "mean"))

        if actual_by_date.empty:
            actual_by_date = grouped["actual"]
        else:
            actual_by_date = actual_by_date.combine_first(grouped["actual"])

        model_series[model_label] = grouped["predicted"]

    if actual_by_date.empty or not model_series:
        return None

    return _build_aligned_frame_from_series(model_series, actual_by_date)


def _plot_comparison(frame: pd.DataFrame, title: str, output_path: Path, product_name: str) -> None:
    """Render and save a comparison line plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 7))
    plt.plot(frame.index, frame[ACTUAL_COLUMN], label="Actual values", color="black", linewidth=2.5)

    for column in frame.columns:
        if column == ACTUAL_COLUMN:
            continue
        plt.plot(frame.index, frame[column], label=column, linewidth=1.8)

    plt.title(f"{product_name.title()} - {title}")
    plt.xlabel("Date")
    plt.ylabel("Target value")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_comparison_plots(output_root: Path, product_name: str) -> None:
    """Generate all available comparison plots for one product."""
    product_output_dir = output_root / product_name
    product_output_dir.mkdir(parents=True, exist_ok=True)

    baseline_frame = _load_baseline_vs_actual_frame(product_output_dir)
    if baseline_frame is not None and not baseline_frame.empty:
        _plot_comparison(
            frame=baseline_frame,
            title="Baseline Models vs Actual",
            output_path=product_output_dir / "naive_sarima" / "baseline_vs_actual.png",
            product_name=product_name,
        )

    xgboost_frame = _load_xgboost_vs_actual_frame(product_output_dir)
    if xgboost_frame is not None and not xgboost_frame.empty:
        _plot_comparison(
            frame=xgboost_frame,
            title="XGBoost Models vs Actual",
            output_path=product_output_dir / "xgboost" / "xgboost_vs_actual.png",
            product_name=product_name,
        )
