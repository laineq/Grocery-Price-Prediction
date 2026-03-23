from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
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

AVOCADO_VECTORS = [
    vector_id
    for vector_id, (_, product) in VECTOR_META.items()
    if product == "Avocado, unit"
]

TOMATO_VECTORS = [
    vector_id
    for vector_id, (_, product) in VECTOR_META.items()
    if product == "Tomatoes, per kilogram"
]


def fetch_full_history(vectors):
    """
    Fetch full historical grocery price data for a given vector list.
    """

    url = "https://www150.statcan.gc.ca/t1/wds/rest/getBulkVectorDataByRange"

    payload = {
        "vectorIds": vectors,
        "startDataPointReleaseDate": "2017-01-01T00:00",
        "endDataPointReleaseDate": "2026-02-28T23:59"
    }

    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


def save_avocado_bronze_full_task():
    """
    Fetch full historical avocado price data from Statistics Canada API
    and store raw JSON into the S3 Bronze layer.
    """

    data = fetch_full_history(AVOCADO_VECTORS)
    print("Fetched full historical avocado data")

    s3 = boto3.client("s3")
    s3_key = (
        "bronze/canadian_grocery_prices/avocado/"
        "full_history/raw_2017_2025.json"
    )

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json.dumps(data),
        ContentType="application/json"
    )

    print(f"Uploaded FULL Bronze: s3://{BUCKET_NAME}/{s3_key}")


def save_tomato_bronze_full_task():
    """
    Fetch full historical tomato price data from Statistics Canada API
    and store raw JSON into the S3 Bronze layer.
    """

    data = fetch_full_history(TOMATO_VECTORS)
    print("Fetched full historical tomato data")

    s3 = boto3.client("s3")
    s3_key = (
        "bronze/canadian_grocery_prices/tomato/"
        "full_history/raw_2017_2025.json"
    )

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
    partitioned by year in the Silver layer.
    """

    s3 = boto3.client("s3")

    bronze_keys = {
        "avocado": (
            "bronze/canadian_grocery_prices/avocado/"
            "full_history/raw_2017_2025.json"
        ),
        "tomato": (
            "bronze/canadian_grocery_prices/tomato/"
            "full_history/raw_2017_2025.json"
        ),
    }

    for product_name, bronze_key in bronze_keys.items():
        print(f"Reading Bronze: s3://{BUCKET_NAME}/{bronze_key}")

        obj = s3.get_object(Bucket=BUCKET_NAME, Key=bronze_key)
        data = json.loads(obj["Body"].read())
        rows = []

        for item in data:
            if item.get("status") != "SUCCESS":
                continue

            vector_id = str(item["object"]["vectorId"])
            datapoints = item["object"]["vectorDataPoint"]

            geography, _ = VECTOR_META.get(vector_id, ("Unknown", "Unknown"))
            if geography != "Canada":
                continue

            for dp in datapoints:
                date_full = dp["refPer"][:7]
                price = dp.get("value")

                if price is None:
                    continue

                rows.append({
                    "year": date_full.split("-")[0],
                    "date": date_full,
                    "price": price
                })

        df = pd.DataFrame(rows)
        print(f"{product_name} sample:")
        print(df.head())
        print(f"Total rows for {product_name}: {len(df):,}")

        # Partitioned Parquet write (year only)
        for year, group in df.groupby("year"):
            buffer = BytesIO()
            group.drop(columns=["year"]).to_parquet(buffer, index=False)

            silver_key = (
                f"silver/canadian_grocery_prices/{product_name}/"
                f"year={year}/data.parquet"
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

    # Task 1: Fetch full avocado history → Bronze
    avocado_fetch_task = PythonOperator(
        task_id="save_avocado_bronze_full_task",
        python_callable=save_avocado_bronze_full_task,
    )

    # Task 2: Fetch full tomato history → Bronze
    tomato_fetch_task = PythonOperator(
        task_id="save_tomato_bronze_full_task",
        python_callable=save_tomato_bronze_full_task,
    )

    # Task 3: Transform Bronze → Silver partitions
    transform_task = PythonOperator(
        task_id="transform_to_silver",
        python_callable=transform_to_silver,
    )

    # Task dependency
    [avocado_fetch_task, tomato_fetch_task] >> transform_task
