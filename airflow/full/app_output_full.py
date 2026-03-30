from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
from io import BytesIO
import boto3
import json
import os
import pandas as pd


BUCKET_NAME = os.environ["BUCKET_NAME"]

PRODUCTS = {
    "avocado": {"unit_label": "unit"},
    "tomato": {"unit_label": "kg"},
}

START_DATE = "2017-01"
END_DATE = "2025-12"


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


def build_historical_output(product, unit_label, actual_df):
    """
    Build app-ready output using actual historical prices only.
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
    actual_df = actual_df[
        (actual_df["date"] >= START_DATE) &
        (actual_df["date"] <= END_DATE)
    ].copy()
    actual_df["forecast"] = False

    series = [
        {
            "date": row["date"],
            "price": round(float(row["price"]), 2),
            "forecast": False,
        }
        for _, row in actual_df.iterrows()
    ]

    return {
        "product": product,
        "unit_label": unit_label,
        "prediction_month": None,
        "forecast_price": None,
        "change_pct": None,
        "series": series,
    }


def transform_to_app_output():
    """
    Build app-ready historical JSON files from Silver grocery price data.
    """

    s3 = boto3.client("s3")

    for product, config in PRODUCTS.items():
        actual_df = read_partitioned_dataset(
            s3,
            f"silver/canadian_grocery_prices/{product}/",
        )

        payload = build_historical_output(
            product=product,
            unit_label=config["unit_label"],
            actual_df=actual_df,
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
    dag_id="app_output_full_history",
    default_args=default_args,
    schedule=None,
    catchup=False,
) as dag:

    transform_task = PythonOperator(
        task_id="transform_to_app_output",
        python_callable=transform_to_app_output,
    )
