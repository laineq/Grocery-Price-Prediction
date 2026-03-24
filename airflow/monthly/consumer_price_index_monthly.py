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

# Current-year configuration
CURRENT_YEAR = 2026

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
    Fetch current-year CPI data for a single StatCan vector.
    """

    params = {
        "vectorIds": f'"{vector_id}"',
        "startRefPeriod": f"{CURRENT_YEAR}-01-01",
        "endReferencePeriod": datetime.now().strftime("%Y-%m-01"),
    }

    resp = requests.get(API_URL, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def save_product_bronze_task(vector_id, product):
    """
    Fetch the current-year CPI snapshot for one product and store it in Bronze.
    """

    print(f"Requesting latest CPI data for {product} (vector {vector_id})...")

    payload = fetch_vector_data(vector_id)

    rows = []
    latest_period = None
    for item in payload:
        if item.get("status") != "SUCCESS":
            continue

        datapoints = item.get("object", {}).get("vectorDataPoint", [])
        for dp in datapoints:
            ref_per = dp.get("refPer", "")
            if not ref_per.startswith(f"{CURRENT_YEAR}-"):
                continue

            rows.append(dp)
            latest_period = ref_per

    if not rows:
        raise Exception(f"No CPI data found for {product} in {CURRENT_YEAR}")

    print(
        f"Latest month detected for {product}: {latest_period[:7]} "
        f"({len(rows)} rows in current-year snapshot)"
    )

    bronze_payload = [{
        "status": "SUCCESS",
        "object": {
            "vectorId": int(vector_id),
            "vectorDataPoint": rows,
        }
    }]

    s3 = boto3.client("s3")
    s3_key = f"bronze/consumer_price_index/{product}/year={CURRENT_YEAR}/raw.json"

    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        existing_payload = json.loads(obj["Body"].read())
        existing_data = (
            existing_payload[0].get("object", {}).get("vectorDataPoint", [])
            if existing_payload else []
        )
        existing_latest_period = (
            existing_data[-1]["refPer"] if existing_data else None
        )

        if existing_latest_period == latest_period:
            print(f"Bronze already up to date: {s3_key}")
            raise AirflowSkipException(
                f"Current-year {product} CPI snapshot already up to date"
            )

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
        Body=json.dumps(bronze_payload),
        ContentType="application/json",
    )

    print(f"Uploaded Bronze: s3://{BUCKET_NAME}/{s3_key}")


def save_avocado_bronze_task():
    """
    Fetch the current-year avocado CPI proxy snapshot and store it in Bronze.
    """

    save_product_bronze_task("41691015", "avocado")


def save_tomato_bronze_task():
    """
    Fetch the current-year tomato CPI snapshot and store it in Bronze.
    """

    save_product_bronze_task("41691023", "tomato")


def transform_to_silver():
    """
    Transform current-year Bronze CPI JSON into yearly Silver Parquet datasets.
    """

    s3 = boto3.client("s3")

    for product in VECTOR_META.values():
        bronze_key = (
            f"bronze/consumer_price_index/{product}/"
            f"year={CURRENT_YEAR}/raw.json"
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

                rows.append({
                    "date": date,
                    "value": value,
                })

        df = pd.DataFrame(rows)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

        print(df.head())
        print(f"Total rows for {product}: {len(df):,}")

        buffer = BytesIO()
        df.to_parquet(buffer, index=False)

        silver_key = (
            f"silver/consumer_price_index/{product}/"
            f"year={CURRENT_YEAR}/data.parquet"
        )

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
    dag_id="consumer_price_index_monthly",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
) as dag:

    # Task 1: Fetch latest avocado CPI data → Bronze
    avocado_fetch_task = PythonOperator(
        task_id="save_avocado_bronze_task",
        python_callable=save_avocado_bronze_task,
    )

    # Task 2: Fetch latest tomato CPI data → Bronze
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
