from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
from io import BytesIO
import requests
import json
import boto3
import os
import pandas as pd


# S3 bucket name (configured via docker-compose environment variable)
BUCKET_NAME = os.environ["BUCKET_NAME"]

# Bank of Canada FX endpoints
USD_CAD_2016_URL = (
    "https://www.bankofcanada.ca/valet/observations/IEXM0102_AVG/json"
    "?start_date=2016-01-01&end_date=2016-12-31"
)
MXN_CAD_2016_URL = (
    "https://www.bankofcanada.ca/valet/observations/IEXM2001/json"
    "?start_date=2016-01-01&end_date=2016-12-31"
)
FX_GROUP_2017_2025_URL = (
    "https://www.bankofcanada.ca/valet/observations/group/FX_RATES_MONTHLY/json"
    "?start_date=2017-01-01&end_date=2025-12-31"
)


def fetch_json(url):
    """
    Download a JSON payload from a public API endpoint.
    """

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.json()


def parse_single_series(payload, series_code, output_col):
    """
    Parse a Bank of Canada single-series response into a normalized DataFrame.
    """

    rows = []

    for obs in payload.get("observations", []):
        value_obj = obs.get(series_code)
        if not value_obj or "v" not in value_obj:
            continue

        rows.append({
            "date": obs["d"][:7],
            output_col: float(value_obj["v"]),
        })

    return pd.DataFrame(rows)


def parse_group_series(payload):
    """
    Parse the Bank of Canada FX group response into a normalized DataFrame.
    """

    rows = []

    for obs in payload.get("observations", []):
        usd = obs.get("FXMUSDCAD", {}).get("v")
        mxn = obs.get("FXMMXNCAD", {}).get("v")

        if usd is None and mxn is None:
            continue

        rows.append({
            "date": obs["d"][:7],
            "USD_CAD": float(usd) if usd is not None else None,
            "MXN_CAD": float(mxn) if mxn is not None else None,
        })

    return pd.DataFrame(rows)


def save_bronze_full_task():
    """
    Fetch full historical FX rate data from the Bank of Canada,
    normalize the 2016 and 2017-2025 responses into one table,
    and store raw JSON in the S3 Bronze layer.
    """

    print("Requesting Bank of Canada FX data...")

    usd_cad_2016_payload = fetch_json(USD_CAD_2016_URL)
    mxn_cad_2016_payload = fetch_json(MXN_CAD_2016_URL)
    fx_group_payload = fetch_json(FX_GROUP_2017_2025_URL)

    usd_cad_2016 = parse_single_series(
        usd_cad_2016_payload, "IEXM0102_AVG", "USD_CAD"
    )
    mxn_cad_2016 = parse_single_series(
        mxn_cad_2016_payload, "IEXM2001", "MXN_CAD"
    )
    fx_2016 = usd_cad_2016.merge(mxn_cad_2016, on="date", how="outer")
    fx_2017_2025 = parse_group_series(fx_group_payload)

    fx_df = pd.concat([fx_2016, fx_2017_2025], ignore_index=True)
    fx_df = fx_df.sort_values("date").reset_index(drop=True)

    print(f"Total FX rows collected: {len(fx_df):,}")
    print(fx_df.head())

    records = fx_df.to_dict(orient="records")

    s3 = boto3.client("s3")
    s3_key = "bronze/exchange_rate/full_history/raw_2016_2025.json"

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json.dumps(records),
        ContentType="application/json",
    )

    print(f"Uploaded FULL Bronze: s3://{BUCKET_NAME}/{s3_key}")


def transform_to_silver():
    """
    Transform full Bronze JSON into yearly Silver Parquet datasets.
    """

    s3 = boto3.client("s3")

    # Bronze full-history file location
    bronze_key = "bronze/exchange_rate/full_history/raw_2016_2025.json"
    print(f"Reading Bronze: s3://{BUCKET_NAME}/{bronze_key}")

    # Load Bronze JSON
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=bronze_key)
    data = json.loads(obj["Body"].read())

    # Build normalized DataFrame
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["USD_CAD"] = pd.to_numeric(df["USD_CAD"], errors="coerce")
    df["MXN_CAD"] = pd.to_numeric(df["MXN_CAD"], errors="coerce")
    df["year"] = df["date"].dt.strftime("%Y")
    df["date"] = df["date"].dt.strftime("%Y-%m")

    print(df.head())
    print(f"Total rows: {len(df):,}")

    # Write one Parquet file per year
    for year, group in df.groupby("year"):
        buffer = BytesIO()
        group.drop(columns=["year"]).to_parquet(buffer, index=False)

        silver_key = f"silver/exchange_rate/year={year}/data.parquet"

        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=silver_key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )

        print(f"Uploaded: s3://{BUCKET_NAME}/{silver_key}")


# Default DAG configuration
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}


# Airflow DAG definition (manual run only)
with DAG(
    dag_id="exchange_rate_full_history",
    default_args=default_args,
    schedule=None,
    catchup=False,
) as dag:

    # Task 1: Fetch full exchange rate history → Bronze
    fetch_task = PythonOperator(
        task_id="save_bronze_full_task",
        python_callable=save_bronze_full_task,
    )

    # Task 2: Transform Bronze → Silver
    transform_task = PythonOperator(
        task_id="transform_to_silver",
        python_callable=transform_to_silver,
    )

    # Task dependency
    fetch_task >> transform_task
