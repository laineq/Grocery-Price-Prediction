from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime
from io import BytesIO
import requests
import json
import boto3
import os
import pandas as pd


# S3 bucket name (configured via docker-compose environment variable)
BUCKET_NAME = os.environ["BUCKET_NAME"]

# Bank of Canada FX endpoint
CURRENT_YEAR = 2026
FX_GROUP_URL = (
    "https://www.bankofcanada.ca/valet/observations/group/FX_RATES_MONTHLY/json"
)


def fetch_json(url, params=None):
    """
    Download a JSON payload from a public API endpoint.
    """

    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def parse_group_series(payload):
    """
    Parse the Bank of Canada FX group response into normalized rows.
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

    return rows


def save_bronze_task():
    """
    Fetch the current-year FX rate snapshot from the Bank of Canada API
    and store raw JSON into the S3 Bronze layer.
    """

    params = {
        "start_date": f"{CURRENT_YEAR}-01-01",
        "end_date": datetime.now().strftime("%Y-%m-%d"),
    }

    print("Requesting latest Bank of Canada FX data...")

    payload = fetch_json(FX_GROUP_URL, params=params)
    data = parse_group_series(payload)

    current_year_data = [
        row for row in data
        if row["date"].startswith(f"{CURRENT_YEAR}-")
    ]

    if not current_year_data:
        raise Exception(f"No FX data found for {CURRENT_YEAR}")

    latest_period = current_year_data[-1]["date"]
    print(
        f"Latest FX month detected for {CURRENT_YEAR}: {latest_period} "
        f"({len(current_year_data)} rows in current-year snapshot)"
    )

    s3 = boto3.client("s3")
    s3_key = f"bronze/exchange_rate/year={CURRENT_YEAR}/raw.json"

    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        existing_payload = json.loads(obj["Body"].read())
        existing_latest_period = (
            existing_payload[-1]["date"] if existing_payload else None
        )

        if existing_latest_period == latest_period:
            print(f"Bronze already up to date: {s3_key}")
            raise AirflowSkipException("Current-year exchange rate snapshot already up to date")

        print(
            f"New month detected. Existing latest month: "
            f"{existing_latest_period}, new latest month: {latest_period}"
        )
    except s3.exceptions.NoSuchKey:
        print("No current-year Bronze snapshot found. Saving Bronze...")
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in {"404", "NoSuchKey"}:
            print("No current-year Bronze snapshot found. Saving Bronze...")
        else:
            raise

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json.dumps(current_year_data),
        ContentType="application/json",
    )

    print(f"Uploaded Bronze: s3://{BUCKET_NAME}/{s3_key}")


def transform_to_silver():
    """
    Transform the current-year Bronze JSON into a yearly Silver Parquet dataset.
    """

    s3 = boto3.client("s3")

    bronze_key = f"bronze/exchange_rate/year={CURRENT_YEAR}/raw.json"
    print(f"Reading Bronze: s3://{BUCKET_NAME}/{bronze_key}")

    obj = s3.get_object(Bucket=BUCKET_NAME, Key=bronze_key)
    data = json.loads(obj["Body"].read())

    df = pd.DataFrame(data)
    df["USD_CAD"] = pd.to_numeric(df["USD_CAD"], errors="coerce")
    df["MXN_CAD"] = pd.to_numeric(df["MXN_CAD"], errors="coerce")

    print(df.head())
    print(f"Total rows: {len(df):,}")

    buffer = BytesIO()
    df.to_parquet(buffer, index=False)

    silver_key = f"silver/exchange_rate/year={CURRENT_YEAR}/data.parquet"

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=silver_key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream",
    )

    print(f"Uploaded Silver: s3://{BUCKET_NAME}/{silver_key}")


# Default DAG configuration
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}


# Airflow DAG definition
with DAG(
    dag_id="exchange_rate_monthly",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
) as dag:

    # Task 1: Fetch latest exchange rate data → Bronze
    fetch_task = PythonOperator(
        task_id="save_bronze_task",
        python_callable=save_bronze_task,
    )

    # Task 2: Transform Bronze → Silver
    transform_task = PythonOperator(
        task_id="transform_to_silver",
        python_callable=transform_to_silver,
    )

    trigger_gold_task = TriggerDagRunOperator(
        task_id="trigger_gold_features_monthly",
        trigger_dag_id="gold_features_monthly",
        wait_for_completion=False,
    )

    # Task dependency
    fetch_task >> transform_task >> trigger_gold_task
