from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
import json
import boto3
from io import BytesIO, StringIO
import os
import pandas as pd
from airflow.exceptions import AirflowSkipException


# S3 bucket name (configured via docker-compose environment variable)
BUCKET_NAME = os.environ["BUCKET_NAME"]


def save_bronze_task():
    """
    Ingest the latest monthly Mexico weather dataset.
    The Bronze layer stores raw, untransformed data exactly as ingested.
    """

    # Public dataset CSV URL (Mexico monthly weather dataset)
    csv_url = (
        "https://datamx.io/dataset/af0e01e5-7867-44bc-84ac-0b31573aa0ae/"
        "resource/917506fc-a09d-4e81-9d75-5e1f11bf1a14/download/data.csv"
    )

    print("Downloading CSV dataset...")

    # Download dataset
    resp = requests.get(csv_url, timeout=120)
    resp.raise_for_status()

    # Load CSV into pandas DataFrame
    df = pd.read_csv(StringIO(resp.content.decode("utf-8")))

    print(f"Total rows downloaded: {len(df):,}")

    # Convert date column to datetime for filtering
    df["PERIODO"] = pd.to_datetime(df["PERIODO"])

    # Identify the most recent month available in the dataset
    latest_date = df["PERIODO"].max()

    year = latest_date.strftime("%Y")
    month = latest_date.strftime("%m")

    print(f"Latest month detected: {year}-{month}")

    # Filter records for the latest month only
    df_latest = df[df["PERIODO"] == latest_date]

    # Convert records to JSON format for Bronze storage
    records = df_latest.to_dict(orient="records")

    # Initialize S3 client
    s3 = boto3.client("s3")

    # Bronze storage path (partitioned by year/month)
    s3_key = (
        f"bronze/mexico_weather/"
        f"year={year}/month={month}/raw.json"
    )

    # Check if this month already exists in Bronze
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=s3_key)

        # If object exists → skip ingestion
        print(f"Bronze already exists: {s3_key}")
        raise AirflowSkipException("Latest month already ingested")

    except s3.exceptions.ClientError as e:

        # Only treat 404 as "new data"
        if e.response["Error"]["Code"] == "404":
            print("New month detected. Saving Bronze...")

        # Any other error should fail the task
        else:
            raise

    # Upload raw JSON data to Bronze layer
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json.dumps(records, default=str, ensure_ascii=False),
        ContentType="application/json"
    )

    print(f"Uploaded Bronze: s3://{BUCKET_NAME}/{s3_key}")


def transform_to_silver():
    """
    Transform the latest Bronze dataset into a structured dataset.
    The Silver layer contains cleaned, analytics-ready data.
    """

    s3 = boto3.client("s3")

    # Bronze prefix where raw ingestion files are stored
    bronze_prefix = "bronze/mexico_weather/"

    print("Searching latest Bronze partition...")

    # List all objects under the Bronze prefix
    resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=bronze_prefix)

    if "Contents" not in resp:
        raise Exception("No Bronze data found")

    # Identify the latest raw.json file
    bronze_files = [
        obj["Key"] for obj in resp["Contents"]
        if obj["Key"].endswith("raw.json")
    ]

    latest_key = sorted(bronze_files)[-1]

    print(f"Reading Bronze file: s3://{BUCKET_NAME}/{latest_key}")

    # Load Bronze JSON file
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=latest_key)
    data = json.loads(obj["Body"].read())

    # Convert JSON records into pandas DataFrame
    df = pd.DataFrame(data)

    print("Raw sample:")
    print(df.head())

    # Standardize column names
    df = df.rename(columns={
        "PERIODO": "PERIOD",
        "CVE_ENT": "STATE_ID",
        "ENTIDAD": "STATE",
        "MINIMA": "MIN_C",
        "MEDIA": "MEAN_C",
        "MAXIMA": "MAX_C",
        "PRECIPITACION": "PRECIPITATION_MM"
    })

    # Convert columns to proper numeric types
    df["PERIOD"] = pd.to_datetime(df["PERIOD"])
    df["STATE_ID"] = pd.to_numeric(df["STATE_ID"], errors="coerce")
    df["MIN_C"] = pd.to_numeric(df["MIN_C"], errors="coerce")
    df["MEAN_C"] = pd.to_numeric(df["MEAN_C"], errors="coerce")
    df["MAX_C"] = pd.to_numeric(df["MAX_C"], errors="coerce")
    df["PRECIPITATION_MM"] = pd.to_numeric(df["PRECIPITATION_MM"], errors="coerce")

    # Replace aggregated "Nacional" label with "Mexico"
    df["STATE"] = df["STATE"].replace({"Nacional": "Mexico"})

    # Extract partition values from date column
    year = df["PERIOD"].dt.strftime("%Y").iloc[0]
    month = df["PERIOD"].dt.strftime("%m").iloc[0]

    # Convert date to string for storage consistency
    df["PERIOD"] = df["PERIOD"].dt.strftime("%Y-%m-%d")

    print("Cleaned sample:")
    print(df.head())

    print(f"Total rows: {len(df):,}")

    # Convert DataFrame into Parquet format (columnar storage)
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)

    # Silver storage path
    silver_key = (
        f"silver/mexico_weather/"
        f"year={year}/month={month}/data.parquet"
    )

    # Upload transformed dataset to Silver layer
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
    dag_id="mexico_weather_monthly",
    default_args=default_args,

    # Run daily to check if a new monthly dataset has been published
    schedule="@daily",

    # Do not backfill historical runs
    catchup=False,
) as dag:

    # Task 1: Ingest latest dataset into Bronze layer
    fetch_task = PythonOperator(
        task_id="save_bronze_task",
        python_callable=save_bronze_task,
    )

    # Task 2: Transform Bronze → Silver
    transform_task = PythonOperator(
        task_id="transform_to_silver",
        python_callable=transform_to_silver,
    )

    # Define task dependency
    fetch_task >> transform_task