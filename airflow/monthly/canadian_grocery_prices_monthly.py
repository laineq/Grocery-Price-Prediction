from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime
import requests
import json
import boto3
from io import BytesIO
import os
import pandas as pd
from airflow.exceptions import AirflowSkipException
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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


def build_retry_session():
    """
    Create an HTTP session with retry logic for intermittent upstream failures.
    """

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )

    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {"User-Agent": "GroceryPrediction-Airflow/1.0"}
    )
    return session


def fetch_latest_snapshot(vectors):
    """
    Fetch the latest monthly StatCan grocery price data for a vector list.
    """

    url = "https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorsAndLatestNPeriods"
    payload = [{"vectorId": v, "latestN": 1} for v in vectors]
    session = build_retry_session()

    resp = session.post(url, json=payload, timeout=(60, 300))
    resp.raise_for_status()
    return resp.json()


def save_product_bronze_task(vectors, product_name):
    """
    Fetch latest monthly grocery price data from Statistics Canada API
    and store the current-year raw JSON snapshot into the S3 Bronze layer.
    """

    print(
        f"Requesting latest StatCan grocery price snapshot for "
        f"{product_name} ({len(vectors)} vectors)"
    )
    data = fetch_latest_snapshot(vectors)
    print(
        f"Received {len(data)} vector responses for {product_name}"
    )

    # Create S3 client
    s3 = boto3.client("s3")

    # Extract latest reference period (e.g., "2026-01-01")
    ref_per = data[0]["object"]["vectorDataPoint"][0]["refPer"]
    year = ref_per[:4]
    print(f"Latest reference period for {product_name}: {ref_per}")

    # Keep only the current-year snapshot
    current_year_data = []
    for item in data:
        datapoints = item.get("object", {}).get("vectorDataPoint", [])
        current_year_points = [
            dp for dp in datapoints
            if dp["refPer"].startswith(f"{year}-")
        ]

        if not current_year_points:
            continue

        item_copy = json.loads(json.dumps(item))
        item_copy["object"]["vectorDataPoint"] = current_year_points
        current_year_data.append(item_copy)

    print(
        f"Prepared {len(current_year_data)} current-year vector snapshots "
        f"for {product_name}"
    )

    # Bronze storage path (partitioned by year)
    s3_key = (
        f"bronze/canadian_grocery_prices/{product_name}/"
        f"year={year}/raw.json"
    )

    # Skip if the stored snapshot already contains the same latest month
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        existing_payload = json.loads(obj["Body"].read())
        existing_latest = (
            existing_payload[0]["object"]["vectorDataPoint"][0]["refPer"]
            if existing_payload else None
        )

        if existing_latest == ref_per:
            print(f"Bronze already up to date: {s3_key}")
            raise AirflowSkipException("Current-year snapshot already up to date")

        print(
            f"New month detected. Existing latest month: "
            f"{existing_latest}, new latest month: {ref_per}"
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
        Body=json.dumps(current_year_data),
        ContentType="application/json"
    )

    print(f"Uploaded Bronze: s3://{BUCKET_NAME}/{s3_key}")


def save_avocado_bronze_task():
    """
    Fetch latest avocado grocery price data and store Bronze snapshot.
    """

    save_product_bronze_task(AVOCADO_VECTORS, "avocado")


def save_tomato_bronze_task():
    """
    Fetch latest tomato grocery price data and store Bronze snapshot.
    """

    save_product_bronze_task(TOMATO_VECTORS, "tomato")


def transform_to_silver():
    """
    Transform latest Bronze JSON into structured Parquet
    and store in S3 Silver layer (partitioned by year).
    """

    s3 = boto3.client("s3")

    bronze_keys = {
        "avocado": "bronze/canadian_grocery_prices/avocado/",
        "tomato": "bronze/canadian_grocery_prices/tomato/",
    }

    for product_name, bronze_prefix in bronze_keys.items():
        resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=bronze_prefix)
        latest_key = sorted(
            obj["Key"] for obj in resp["Contents"]
            if obj["Key"].endswith("raw.json")
        )[-1]

        print(f"Reading Bronze: s3://{BUCKET_NAME}/{latest_key}")

        obj = s3.get_object(Bucket=BUCKET_NAME, Key=latest_key)
        data = json.loads(obj["Body"].read())
        rows = []

        for item in data:
            vector_id = str(item["object"]["vectorId"])
            datapoints = item["object"]["vectorDataPoint"]

            geography, _ = VECTOR_META.get(vector_id, ("Unknown", "Unknown"))
            if geography != "Canada":
                continue

            for dp in datapoints:
                date = dp["refPer"][:7]  # YYYY-MM
                year = date.split("-")[0]
                price = dp["value"]

                rows.append({
                    "year": year,
                    "date": date,
                    "price": price
                })

        df = pd.DataFrame(rows)
        print(f"{product_name} sample:")
        print(df.head())

        # Convert to Parquet (columnar format for analytics)
        buffer = BytesIO()
        df.drop(columns=["year"]).to_parquet(buffer, index=False)

        # Silver storage path (partitioned by year)
        year = df["year"].iloc[0]
        silver_key = (
            f"silver/canadian_grocery_prices/{product_name}/"
            f"year={year}/data.parquet"
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

    # Task 1: Fetch latest avocado data → Bronze
    avocado_fetch_task = PythonOperator(
        task_id="save_avocado_bronze_task",
        python_callable=save_avocado_bronze_task,
    )

    # Task 2: Fetch latest tomato data → Bronze
    tomato_fetch_task = PythonOperator(
        task_id="save_tomato_bronze_task",
        python_callable=save_tomato_bronze_task,
    )

    # Task 3: Transform Bronze → Silver
    transform_task = PythonOperator(
        task_id="transform_to_silver",
        python_callable=transform_to_silver,
    )

    trigger_adjusted_task = TriggerDagRunOperator(
        task_id="trigger_grocery_price_adjusted_monthly",
        trigger_dag_id="grocery_price_adjusted_monthly",
        wait_for_completion=False,
    )

    # Task dependency
    [avocado_fetch_task, tomato_fetch_task] >> transform_task >> trigger_adjusted_task
