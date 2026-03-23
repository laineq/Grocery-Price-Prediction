from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
from datetime import datetime
import requests
import json
import boto3
import os


# S3 bucket name (configured via docker-compose environment variable)
BUCKET_NAME = os.environ["BUCKET_NAME"]

# EIA API endpoint
API_URL = "https://api.eia.gov/v2/petroleum/pri/gnd/data/"
CURRENT_YEAR = 2026

# Static request parameters
API_PARAMS = {
    "frequency": "monthly",
    "data[0]": "value",
    "facets[series][]": "EMD_EPD2D_PTE_NUS_DPG",
    "start": "2016-01",
    "sort[0][column]": "period",
    "sort[0][direction]": "asc",
    "length": 5000,
    "api_key": "1LagLJXXow2sdsFkMzBHgxrdYeiZYwqJAiakHmVo",
}


def save_us_bronze_task():
    """
    Fetch the current-year U.S. diesel price snapshot from the EIA API
    and store raw JSON into the S3 Bronze layer.
    """

    params = API_PARAMS.copy()
    params["end"] = datetime.now().strftime("%Y-%m")

    print("Requesting latest EIA oil price data...")

    # Call EIA API
    resp = requests.get(API_URL, params=params, timeout=120)
    resp.raise_for_status()
    payload = resp.json()

    data = payload.get("response", {}).get("data", [])
    if not data:
        raise Exception("No oil price data returned from EIA API")

    # Keep only the current-year snapshot
    current_year_data = [
        row for row in data
        if row.get("period", "").startswith(f"{CURRENT_YEAR}-")
    ]

    if not current_year_data:
        raise Exception(f"No oil price data found for {CURRENT_YEAR}")

    latest_period = current_year_data[-1]["period"]
    print(
        f"Latest month detected for {CURRENT_YEAR}: {latest_period} "
        f"({len(current_year_data)} rows in current-year snapshot)"
    )

    bronze_payload = {
        "response": {
            "total": str(len(current_year_data)),
            "dateFormat": payload.get("response", {}).get("dateFormat"),
            "frequency": payload.get("response", {}).get("frequency"),
            "data": current_year_data,
        }
    }

    # Create S3 client
    s3 = boto3.client("s3")

    # Bronze storage path (partitioned by year)
    s3_key = f"bronze/oil_prices/us/year={CURRENT_YEAR}/raw.json"

    # Skip if the stored snapshot already contains the same latest month
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        existing_payload = json.loads(obj["Body"].read())
        existing_data = existing_payload.get("response", {}).get("data", [])
        existing_latest_period = (
            existing_data[-1]["period"] if existing_data else None
        )

        if existing_latest_period == latest_period:
            print(f"Bronze already up to date: {s3_key}")
            raise AirflowSkipException("Current-year snapshot already up to date")

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

    # Upload raw JSON to S3 Bronze layer
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json.dumps(bronze_payload),
        ContentType="application/json",
    )

    print(f"Uploaded Bronze: s3://{BUCKET_NAME}/{s3_key}")


# Default DAG configuration
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}


# Airflow DAG definition
with DAG(
    dag_id="oil_prices_monthly",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
) as dag:

    # Task 1: Fetch latest U.S. monthly data → Bronze
    fetch_task = PythonOperator(
        task_id="save_us_bronze_task",
        python_callable=save_us_bronze_task,
    )
