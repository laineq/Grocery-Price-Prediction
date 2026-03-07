from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
import json
import boto3
from io import BytesIO
import os
import pandas as pd


# S3 bucket name (configured via docker-compose environment variable)
BUCKET_NAME = os.environ["BUCKET_NAME"]


# Mapping: vectorId → (Geography, Product name)
# Single source of truth for all vectors
VECTOR_META = {
    # Avocado
    "1353834299": ("Canada", "Avocado, unit"),
    "1159447003": ("Newfoundland and Labrador", "Avocado, unit"),
    "1159447043": ("Prince Edward Island", "Avocado, unit"),
    "1159447083": ("Nova Scotia", "Avocado, unit"),
    "1159447123": ("New Brunswick", "Avocado, unit"),
    "1159447163": ("Quebec", "Avocado, unit"),
    "1159447203": ("Ontario", "Avocado, unit"),
    "1159447243": ("Manitoba", "Avocado, unit"),
    "1159447283": ("Saskatchewan", "Avocado, unit"),
    "1159447323": ("Alberta", "Avocado, unit"),
    "1159447363": ("British Columbia", "Avocado, unit"),

    # Tomato
    "1353834301": ("Canada", "Tomatoes, per kilogram"),
    "1159447005": ("Newfoundland and Labrador", "Tomatoes, per kilogram"),
    "1159447045": ("Prince Edward Island", "Tomatoes, per kilogram"),
    "1159447085": ("Nova Scotia", "Tomatoes, per kilogram"),
    "1159447125": ("New Brunswick", "Tomatoes, per kilogram"),
    "1159447165": ("Quebec", "Tomatoes, per kilogram"),
    "1159447205": ("Ontario", "Tomatoes, per kilogram"),
    "1159447245": ("Manitoba", "Tomatoes, per kilogram"),
    "1159447285": ("Saskatchewan", "Tomatoes, per kilogram"),
    "1159447325": ("Alberta", "Tomatoes, per kilogram"),
    "1159447365": ("British Columbia", "Tomatoes, per kilogram"),
}


def save_bronze_full_task():
    """
    Fetch full historical grocery price data from Statistics Canada API
    (2017–2026 range) and store raw JSON into the S3 Bronze layer.
    """

    # StatCan Web Data Service endpoint for bulk historical retrieval
    url = "https://www150.statcan.gc.ca/t1/wds/rest/getBulkVectorDataByRange"

    # Use all vector IDs defined in metadata
    vectors = list(VECTOR_META.keys())

    # Request payload: full historical date range
    payload = {
        "vectorIds": vectors,
        "startDataPointReleaseDate": "2017-01-01T00:00",
        "endDataPointReleaseDate": "2026-02-28T23:59"
    }

    # Call API
    resp = requests.post(url, json=payload)
    data = resp.json()

    print("Fetched full historical data")

    # Create S3 client
    s3 = boto3.client("s3")

    # Store as a single raw Bronze file (full snapshot)
    s3_key = "bronze/canadian_grocery_prices/full_history/raw_2017_2026.json"

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json.dumps(data),
        ContentType="application/json"
    )

    print(f"Uploaded FULL Bronze: s3://{BUCKET_NAME}/{s3_key}")


def transform_to_silver():
    """
    Transform full Bronze JSON into structured Parquet files
    partitioned by year/month in the Silver layer.
    """

    s3 = boto3.client("s3")

    # Bronze full-history file location
    bronze_key = "bronze/canadian_grocery_prices/full_history/raw_2017_2026.json"
    print(f"Reading Bronze: s3://{BUCKET_NAME}/{bronze_key}")

    # Load Bronze JSON
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=bronze_key)
    data = json.loads(obj["Body"].read())

    rows = []

    # Parse API response
    for item in data:
        # Skip failed vectors
        if item.get("status") != "SUCCESS":
            continue

        vector_id = str(item["object"]["vectorId"])
        datapoints = item["object"]["vectorDataPoint"]

        geography, product = VECTOR_META.get(vector_id, ("Unknown", "Unknown"))

        # Flatten monthly datapoints
        for dp in datapoints:
            date_full = dp["refPer"][:7]   # YYYY-MM
            price = dp.get("value")

            # Skip missing values
            if price is None:
                continue

            year, month = date_full.split("-")

            rows.append({
                "year": year,
                "month": month,
                "date": date_full,
                "geography": geography,
                "product": product,
                "price": price
            })

    # Build DataFrame
    df = pd.DataFrame(rows)
    print(df.head())
    print(f"Total rows: {len(df):,}")

    # Partitioned Parquet write (year/month)
    for (year, month), group in df.groupby(["year", "month"]):
        buffer = BytesIO()
        group.to_parquet(buffer, index=False)

        silver_key = (
            f"silver/canadian_grocery_prices/"
            f"year={year}/month={month}/data.parquet"
        )

        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=silver_key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream"
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
    dag_id="canadian_grocery_prices_full_history",
    default_args=default_args,
    schedule=None,      # Run manually (one-time backfill)
    catchup=False,
) as dag:

    # Task 1: Fetch full history → Bronze
    fetch_task = PythonOperator(
        task_id="save_bronze_full_task",
        python_callable=save_bronze_full_task,
    )

    # Task 2: Transform Bronze → Silver partitions
    transform_task = PythonOperator(
        task_id="transform_to_silver",
        python_callable=transform_to_silver,
    )

    # Task dependency
    fetch_task >> transform_task