from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import requests
import json
import boto3
from io import StringIO
from io import BytesIO
import os
import pandas as pd

# S3 bucket name provided through environment variable
BUCKET_NAME = os.environ["BUCKET_NAME"]

# Only keep the Mexico states used in downstream avocado/tomato features
TARGET_STATES = [
    "Sinaloa",
    "Michoacán",
    "Jalisco",
    "Estado de México",
]


def save_bronze_full_task():
    """
    Download the full Mexico weather dataset (CSV),
    filter the desired time range, and store the raw data
    as JSON in the Bronze layer of the data lake.
    """

    # Source CSV dataset URL
    csv_url = (
        "https://datamx.io/dataset/af0e01e5-7867-44bc-84ac-0b31573aa0ae/"
        "resource/917506fc-a09d-4e81-9d75-5e1f11bf1a14/download/data.csv"
    )

    print("Downloading CSV dataset...")

    # Download CSV file from the source
    resp = requests.get(csv_url, timeout=120)
    resp.raise_for_status()

    # Load CSV into a pandas DataFrame
    df = pd.read_csv(StringIO(resp.content.decode("utf-8")))

    print(f"Total rows downloaded: {len(df):,}")

    # Convert the PERIODO column to datetime format
    df["PERIODO"] = pd.to_datetime(df["PERIODO"])

    # Define the time window we want to keep
    start_date = "2016-01-01"
    end_date = "2026-01-01"

    # Filter dataset to only include the target date range
    df_filtered = df[
        (df["PERIODO"] >= start_date) &
        (df["PERIODO"] <= end_date)
    ]

    print(f"Filtered rows (2016-01 → 2026-01): {len(df_filtered):,}")

    # Convert filtered DataFrame to JSON records
    records = df_filtered.to_dict(orient="records")

    # Create S3 client
    s3 = boto3.client("s3")

    # Bronze storage location (raw snapshot of dataset)
    s3_key = "bronze/mexico_weather/full_history/raw_2016_2026.json"

    # Upload raw JSON to S3 Bronze layer
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json.dumps(records, default=str, ensure_ascii=False),
        ContentType="application/json"
    )

    print(f"Uploaded FULL Bronze: s3://{BUCKET_NAME}/{s3_key}")


def transform_to_silver():
    """
    Transform raw Bronze JSON data into a cleaned
    and structured dataset stored in the Silver layer
    as partitioned Parquet files.
    """

    # Create S3 client
    s3 = boto3.client("s3")

    # Location of Bronze dataset
    bronze_key = "bronze/mexico_weather/full_history/raw_2016_2026.json"
    print(f"Reading Bronze: s3://{BUCKET_NAME}/{bronze_key}")

    # Load Bronze JSON from S3
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=bronze_key)
    data = json.loads(obj["Body"].read())

    # Convert JSON records into a pandas DataFrame
    df = pd.DataFrame(data)

    # Rename columns to standardized names
    df = df.rename(columns={
        "PERIODO": "PERIOD",
        "CVE_ENT": "STATE_ID",
        "ENTIDAD": "STATE",
        "MINIMA": "MIN_C",
        "MEDIA": "MEAN_C",
        "MAXIMA": "MAX_C",
        "PRECIPITACION": "PRECIPITATION_MM"
    })

    # Convert columns to appropriate data types
    df["PERIOD"] = pd.to_datetime(df["PERIOD"])
    df["STATE_ID"] = pd.to_numeric(df["STATE_ID"], errors="coerce")
    df["MIN_C"] = pd.to_numeric(df["MIN_C"], errors="coerce")
    df["MEAN_C"] = pd.to_numeric(df["MEAN_C"], errors="coerce")
    df["MAX_C"] = pd.to_numeric(df["MAX_C"], errors="coerce")
    df["PRECIPITATION_MM"] = pd.to_numeric(df["PRECIPITATION_MM"], errors="coerce")

    # Replace aggregated "Nacional" label with "Mexico"
    df["STATE"] = df["STATE"].replace({"Nacional": "Mexico"})
    df = df[df["STATE"].isin(TARGET_STATES)].copy()
    df["year"] = df["PERIOD"].dt.strftime("%Y")

    # Convert date column to string format for consistency
    df["PERIOD"] = df["PERIOD"].dt.strftime("%Y-%m-%d")

    print("Cleaned sample:")
    print(df.head())

    print(f"Total rows: {len(df):,}")

    # Write yearly Parquet files to the Silver layer
    for year, group in df.groupby("year"):

        # Convert group DataFrame to Parquet in memory
        buffer = BytesIO()
        group.drop(columns=["year"]).to_parquet(buffer, index=False)

        # Silver partition path
        silver_key = f"silver/mexico_weather/year={year}/data.parquet"

        # Upload Parquet file to S3
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


# DAG definition for one-time historical backfill
with DAG(
    dag_id="mexico_weather_full_history",
    default_args=default_args,
    schedule=None,      # Manual execution only
    catchup=False,
) as dag:

    # Task 1: Fetch full dataset → Bronze layer
    fetch_task = PythonOperator(
        task_id="save_bronze_full_task",
        python_callable=save_bronze_full_task,
    )

    # Task 2: Transform Bronze → Silver partitioned dataset
    transform_task = PythonOperator(
        task_id="transform_to_silver",
        python_callable=transform_to_silver,
    )

    # Task dependency order
    fetch_task >> transform_task
