from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
from datetime import datetime
from io import BytesIO, StringIO
import boto3
import json
import os
import pandas as pd


BUCKET_NAME = os.environ["BUCKET_NAME"]

PRODUCTS = {
    "avocado": {
        "unit_label": "unit",
        "prediction_key": "prediction/avocado_predictions.csv",
    },
    "tomato": {
        "unit_label": "kg",
        "prediction_key": "prediction/tomato_predictions.csv",
    },
}

BASE_DATE = "2017-01"
START_DATE = "2017-01"


def read_partitioned_dataset(s3, prefix):
    """
    Read all Parquet partitions under a given S3 prefix into one DataFrame.
    """

    resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    keys = sorted(
        obj["Key"] for obj in resp.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    )

    if not keys:
        raise FileNotFoundError(f"No parquet files found under {prefix}")

    frames = []
    for key in keys:
        print(f"Reading: s3://{BUCKET_NAME}/{key}")
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        frames.append(pd.read_parquet(BytesIO(obj["Body"].read())))

    return pd.concat(frames, ignore_index=True)


def read_csv_from_s3(s3, key):
    """
    Read a CSV file from S3 into a DataFrame.
    """

    print(f"Reading: s3://{BUCKET_NAME}/{key}")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))


def s3_key_exists(s3, key):
    """
    Check whether an S3 object exists.
    """

    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=key)
        return True
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in {"404", "NoSuchKey"}:
            return False
        raise


def format_prediction_month(date_value):
    """
    Convert YYYY-MM string to 'Month YYYY'.
    """

    return pd.to_datetime(date_value, format="%Y-%m").strftime("%B %Y")


def load_prediction_df(s3, key):
    """
    Load prediction CSV and normalize its schema.
    """

    df = read_csv_from_s3(s3, key).rename(columns={
        "Date": "date",
        "Predicted_Price": "price_adjusted",
        "Lower_CI": "lower_adjusted",
        "Upper_CI": "upper_adjusted",
    })

    required_columns = {"date", "price_adjusted", "lower_adjusted", "upper_adjusted"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing prediction columns in {key}: {sorted(missing_columns)}")

    df["date"] = df["date"].astype(str).str[:7]
    for column in ["price_adjusted", "lower_adjusted", "upper_adjusted"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return (
        df[["date", "price_adjusted", "lower_adjusted", "upper_adjusted"]]
        .dropna()
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def apply_reverse_cpi(prediction_df, cpi_df):
    """
    Convert CPI-adjusted predictions back to nominal prices using the
    reverse of the grocery_price_adjusted transformation.
    """

    cpi_df = cpi_df.copy()
    cpi_df["date"] = cpi_df["date"].astype(str).str[:7]
    cpi_df["cpi"] = pd.to_numeric(cpi_df["value"], errors="coerce")
    cpi_df = (
        cpi_df[["date", "cpi"]]
        .dropna()
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )

    base_rows = cpi_df.loc[cpi_df["date"] == BASE_DATE, "cpi"]
    if base_rows.empty:
        raise FileNotFoundError(f"Base CPI date {BASE_DATE} not found.")

    base_cpi = float(base_rows.iloc[0])

    prediction_df = prediction_df.copy()
    prediction_df["date_dt"] = pd.to_datetime(prediction_df["date"], format="%Y-%m")
    cpi_df["date_dt"] = pd.to_datetime(cpi_df["date"], format="%Y-%m")

    merged = pd.merge_asof(
        prediction_df.sort_values("date_dt"),
        cpi_df[["date_dt", "cpi"]].sort_values("date_dt"),
        on="date_dt",
        direction="backward",
    )

    if merged["cpi"].isna().any():
        raise FileNotFoundError("Unable to map CPI values to all prediction months.")

    for source_column, target_column in [
        ("price_adjusted", "price"),
        ("lower_adjusted", "lower_bound"),
        ("upper_adjusted", "upper_bound"),
    ]:
        merged[target_column] = (merged[source_column] * merged["cpi"] / base_cpi).round(2)

    merged["forecast"] = True
    return merged[["date", "price", "lower_bound", "upper_bound", "forecast"]]


def build_app_output(product, unit_label, actual_df, prediction_df):
    """
    Merge actual and prediction data, prioritizing actual data for overlapping months.
    """

    actual_df = actual_df.copy()
    actual_df["date"] = actual_df["date"].astype(str).str[:7]
    actual_df["price"] = pd.to_numeric(actual_df["price"], errors="coerce")
    actual_df = (
        actual_df[["date", "price"]]
        .dropna()
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    actual_df = actual_df[actual_df["date"] >= START_DATE].copy()
    actual_df["forecast"] = False

    filtered_prediction_df = prediction_df[
        ~prediction_df["date"].isin(actual_df["date"])
    ].copy()

    combined_df = pd.concat(
        [
            actual_df[["date", "price", "forecast"]],
            filtered_prediction_df[["date", "price", "lower_bound", "upper_bound", "forecast"]],
        ],
        ignore_index=True,
        sort=False,
    ).sort_values("date").reset_index(drop=True)

    future_prediction_df = filtered_prediction_df[
        filtered_prediction_df["date"] > actual_df["date"].max()
    ].sort_values("date")

    if future_prediction_df.empty:
        prediction_month = None
        forecast_price = None
        change_pct = None
    else:
        selected_prediction = future_prediction_df.iloc[0]
        latest_actual_price = float(actual_df.iloc[-1]["price"])
        forecast_price = round(float(selected_prediction["price"]), 2)
        prediction_month = format_prediction_month(selected_prediction["date"])
        change_pct = round(((forecast_price - latest_actual_price) / latest_actual_price) * 100, 1)

    series = []
    for _, row in combined_df.iterrows():
        item = {
            "date": row["date"],
            "price": round(float(row["price"]), 2),
            "forecast": bool(row["forecast"]),
        }
        if item["forecast"]:
            item["lower_bound"] = round(float(row["lower_bound"]), 2)
            item["upper_bound"] = round(float(row["upper_bound"]), 2)
        series.append(item)

    return {
        "product": product,
        "unit_label": unit_label,
        "prediction_month": prediction_month,
        "forecast_price": forecast_price,
        "change_pct": change_pct,
        "series": series,
    }


def transform_to_app_output():
    """
    Build app-ready JSON files from actual Silver data plus prediction CSVs.
    """

    s3 = boto3.client("s3")

    missing_prediction_keys = [
        config["prediction_key"]
        for config in PRODUCTS.values()
        if not s3_key_exists(s3, config["prediction_key"])
    ]
    if missing_prediction_keys:
        raise AirflowSkipException(
            f"Prediction files are not ready yet: {missing_prediction_keys}"
        )

    for product, config in PRODUCTS.items():
        actual_df = read_partitioned_dataset(
            s3,
            f"silver/canadian_grocery_prices/{product}/",
        )
        cpi_df = read_partitioned_dataset(
            s3,
            f"silver/consumer_price_index/{product}/",
        )
        prediction_adjusted_df = load_prediction_df(s3, config["prediction_key"])
        prediction_nominal_df = apply_reverse_cpi(prediction_adjusted_df, cpi_df)

        payload = build_app_output(
            product=product,
            unit_label=config["unit_label"],
            actual_df=actual_df,
            prediction_df=prediction_nominal_df,
        )

        output_key = f"app-output/{product}.json"
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=output_key,
            Body=json.dumps(payload, ensure_ascii=False, indent=2),
            ContentType="application/json",
        )

        print(f"Uploaded app output: s3://{BUCKET_NAME}/{output_key}")


default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}


with DAG(
    dag_id="app_output_monthly",
    default_args=default_args,
    schedule=None,
    catchup=False,
) as dag:

    transform_task = PythonOperator(
        task_id="transform_to_app_output",
        python_callable=transform_to_app_output,
    )
