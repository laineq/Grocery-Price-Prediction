from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
import json
import boto3
from io import BytesIO
import os
import pandas as pd
from airflow.exceptions import AirflowSkipException

# S3 bucket name (set via docker-compose environment variable)
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


def save_bronze_task():
    """
    Fetch latest monthly grocery price data from Statistics Canada API
    and store raw JSON into the S3 Bronze layer.
    """

    # StatCan Web Data Service endpoint (latest N periods)
    url = "https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorsAndLatestNPeriods"

    # Use all vector IDs from metadata (no duplication)
    vectors = list(VECTOR_META.keys())

    # Request: get latest 1 data point per vector
    payload = [{"vectorId": v, "latestN": 1} for v in vectors]

    # Call API
    resp = requests.post(url, json=payload)
    data = resp.json()

    # Log raw response for debugging
    print(json.dumps(data, indent=2))

    # Create S3 client
    s3 = boto3.client("s3")

    # Extract reference period (e.g., "2026-01-01")
    ref_per = data[0]["object"]["vectorDataPoint"][0]["refPer"]
    year = ref_per[:4]
    month = ref_per[5:7]

    # Bronze storage path (partitioned by year/month)
    s3_key = (
        f"bronze/canadian_grocery_prices/"
        f"year={year}/month={month}/raw.json"
    )

    # Skip if this month already exists
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=s3_key)
        print(f"Bronze already exists: {s3_key}")
        raise AirflowSkipException("Latest month already collected")
    except s3.exceptions.ClientError:
        print("New month detected. Saving Bronze...")

    # Upload raw JSON to S3 Bronze layer
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json.dumps(data),
        ContentType="application/json"
    )

    print(f"Uploaded Bronze: s3://{BUCKET_NAME}/{s3_key}")


def transform_to_silver():
    """
    Transform latest Bronze JSON into structured Parquet
    and store in S3 Silver layer (partitioned by year/month).
    """

    s3 = boto3.client("s3")

    # Bronze prefix
    bronze_prefix = "bronze/canadian_grocery_prices/"

    # Find latest Bronze file
    resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=bronze_prefix)
    latest_key = sorted([obj["Key"] for obj in resp["Contents"]])[-1]

    print(f"Reading Bronze: s3://{BUCKET_NAME}/{latest_key}")

    # Load Bronze JSON
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=latest_key)
    data = json.loads(obj["Body"].read())

    rows = []

    # Transform nested API structure → flat table
    for item in data:
        vector_id = str(item["object"]["vectorId"])
        dp = item["object"]["vectorDataPoint"][0]

        date = dp["refPer"][:7]  # YYYY-MM
        year, month = date.split("-")
        price = dp["value"]

        geography, product = VECTOR_META.get(vector_id, ("Unknown", "Unknown"))

        rows.append({
            "year": year,
            "month": month,
            "date": date,
            "geography": geography,
            "product": product,
            "price": price
        })

    # Create DataFrame
    df = pd.DataFrame(rows)
    print(df.head())

    # Convert to Parquet (columnar format for analytics)
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)

    # Silver storage path (partitioned)
    year = df["year"].iloc[0]
    month = df["month"].iloc[0]

    silver_key = (
        f"silver/canadian_grocery_prices/"
        f"year={year}/month={month}/data.parquet"
    )

    # Upload Parquet to S3 Silver layer
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=silver_key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream"
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
    dag_id="canadian_grocery_prices_monthly",
    default_args=default_args,
    schedule="@daily",   # Run daily to check for new monthly data
    catchup=False,
) as dag:

    # Task 1: Fetch latest data → Bronze
    fetch_task = PythonOperator(
        task_id="save_bronze_task",
        python_callable=save_bronze_task,
    )

    # Task 2: Transform Bronze → Silver
    transform_task = PythonOperator(
        task_id="transform_to_silver",
        python_callable=transform_to_silver,
    )

    # Task dependency
    fetch_task >> transform_task