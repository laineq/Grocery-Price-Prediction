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

# Statistics Canada CPI vector metadata
VECTOR_META = {
    "41691015": "avocado",
    "41691023": "tomato",
}

# StatCan Web Data Service endpoint
API_URL = (
    "https://www150.statcan.gc.ca/t1/wds/rest/"
    "getDataFromVectorByReferencePeriodRange"
)


def fetch_vector_data(vector_id):
    """
    Fetch full historical CPI data for a single StatCan vector.
    """

    params = {
        "vectorIds": f'"{vector_id}"',
        "startRefPeriod": "2016-01-01",
        "endReferencePeriod": "2025-12-31",
    }

    resp = requests.get(API_URL, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def save_avocado_bronze_full_task():
    """
    Fetch full historical avocado CPI proxy data and store raw JSON in Bronze.
    """

    vector_id = "41691015"
    print(f"Requesting full CPI data for vector {vector_id}...")

    payload = fetch_vector_data(vector_id)

    s3 = boto3.client("s3")
    s3_key = "bronze/consumer_price_index/avocado/full_history/raw_2016_2025.json"

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json.dumps(payload),
        ContentType="application/json",
    )

    print(f"Uploaded FULL Bronze: s3://{BUCKET_NAME}/{s3_key}")


def save_tomato_bronze_full_task():
    """
    Fetch full historical tomato CPI data and store raw JSON in Bronze.
    """

    vector_id = "41691023"
    print(f"Requesting full CPI data for vector {vector_id}...")

    payload = fetch_vector_data(vector_id)

    s3 = boto3.client("s3")
    s3_key = "bronze/consumer_price_index/tomato/full_history/raw_2016_2025.json"

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json.dumps(payload),
        ContentType="application/json",
    )

    print(f"Uploaded FULL Bronze: s3://{BUCKET_NAME}/{s3_key}")


def transform_to_silver():
    """
    Transform full Bronze CPI JSON into yearly Silver Parquet datasets.
    """

    s3 = boto3.client("s3")

    for product in VECTOR_META.values():
        bronze_key = (
            f"bronze/consumer_price_index/{product}/"
            f"full_history/raw_2016_2025.json"
        )
        print(f"Reading Bronze: s3://{BUCKET_NAME}/{bronze_key}")

        obj = s3.get_object(Bucket=BUCKET_NAME, Key=bronze_key)
        payload = json.loads(obj["Body"].read())

        rows = []
        for item in payload:
            if item.get("status") != "SUCCESS":
                continue

            datapoints = item.get("object", {}).get("vectorDataPoint", [])
            for dp in datapoints:
                date = dp.get("refPer", "")[:7]
                value = dp.get("value")

                if not date or value is None:
                    continue

                year = date[:4]
                rows.append({
                    "date": date,
                    "value": value,
                    "year": year,
                })

        df = pd.DataFrame(rows)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

        print(df.head())
        print(f"Total rows for {product}: {len(df):,}")

        for year, group in df.groupby("year"):
            buffer = BytesIO()
            group.drop(columns=["year"]).to_parquet(buffer, index=False)

            silver_key = (
                f"silver/consumer_price_index/{product}/"
                f"year={year}/data.parquet"
            )

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
    dag_id="consumer_price_index_full_history",
    default_args=default_args,
    schedule=None,
    catchup=False,
) as dag:

    # Task 1: Fetch full avocado CPI history → Bronze
    avocado_fetch_task = PythonOperator(
        task_id="save_avocado_bronze_full_task",
        python_callable=save_avocado_bronze_full_task,
    )

    # Task 2: Fetch full tomato CPI history → Bronze
    tomato_fetch_task = PythonOperator(
        task_id="save_tomato_bronze_full_task",
        python_callable=save_tomato_bronze_full_task,
    )

    # Task 3: Transform Bronze → Silver
    transform_task = PythonOperator(
        task_id="transform_to_silver",
        python_callable=transform_to_silver,
    )

    # Task dependency
    [avocado_fetch_task, tomato_fetch_task] >> transform_task
