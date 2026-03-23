from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import requests
import json
import boto3
import os


# S3 bucket name (configured via docker-compose environment variable)
BUCKET_NAME = os.environ["BUCKET_NAME"]

# EIA API endpoint and request parameters
API_URL = "https://api.eia.gov/v2/petroleum/pri/gnd/data/"
API_PARAMS = {
    "frequency": "monthly",
    "data[0]": "value",
    "facets[series][]": "EMD_EPD2D_PTE_NUS_DPG",
    "start": "2016-01",
    "end": "2026-03",
    "sort[0][column]": "period",
    "sort[0][direction]": "asc",
    "length": 5000,
    "api_key": "1LagLJXXow2sdsFkMzBHgxrdYeiZYwqJAiakHmVo",
}


def save_us_bronze_full_task():
    """
    Fetch full historical U.S. diesel price data from the EIA API,
    keep the 2016-2025 range, and store raw JSON in the S3 Bronze layer.
    """

    print("Requesting EIA oil price data...")

    # Call EIA API
    resp = requests.get(API_URL, params=API_PARAMS, timeout=120)
    resp.raise_for_status()
    payload = resp.json()

    data = payload.get("response", {}).get("data", [])
    print(f"Total rows received: {len(data):,}")

    # Keep only the target history range for Bronze backfill
    filtered_data = [
        row for row in data
        if "2016-01" <= row.get("period", "") <= "2025-12"
    ]

    print(f"Filtered rows (2016-01 → 2025-12): {len(filtered_data):,}")

    bronze_payload = {
        "response": {
            "total": str(len(filtered_data)),
            "dateFormat": payload.get("response", {}).get("dateFormat"),
            "frequency": payload.get("response", {}).get("frequency"),
            "data": filtered_data,
        }
    }

    # Create S3 client
    s3 = boto3.client("s3")

    # Store as a single raw Bronze file (full snapshot)
    s3_key = "bronze/oil_prices/us/full_history/raw_2016_2025.json"

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json.dumps(bronze_payload),
        ContentType="application/json",
    )

    print(f"Uploaded FULL Bronze: s3://{BUCKET_NAME}/{s3_key}")


# Default DAG configuration
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}


# Airflow DAG definition (manual run only)
with DAG(
    dag_id="oil_prices_full_history",
    default_args=default_args,
    schedule=None,
    catchup=False,
) as dag:

    # Task 1: Fetch full U.S. history → Bronze
    fetch_task = PythonOperator(
        task_id="save_us_bronze_full_task",
        python_callable=save_us_bronze_full_task,
    )
