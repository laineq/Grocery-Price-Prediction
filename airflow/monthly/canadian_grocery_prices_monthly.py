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
LATEST_N = int(os.environ.get("STATCAN_GROCERY_LATEST_N", "1"))


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
    payload = [{"vectorId": v, "latestN": LATEST_N} for v in vectors]
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

    all_ref_pers = sorted(
        {
            dp["refPer"]
            for item in data
            for dp in item.get("object", {}).get("vectorDataPoint", [])
        }
    )
    if not all_ref_pers:
        raise AirflowSkipException("No monthly datapoints returned from StatCan")

    latest_ref_per = all_ref_pers[-1]
    print(
        f"Latest reference period for {product_name}: {latest_ref_per} "
        f"(latestN={LATEST_N}, months returned={all_ref_pers})"
    )

    month_payloads = {}
    for ref_per in all_ref_pers:
        year = ref_per[:4]
        month = ref_per[5:7]
        month_items = []

        for item in data:
            datapoints = item.get("object", {}).get("vectorDataPoint", [])
            month_points = [dp for dp in datapoints if dp["refPer"] == ref_per]
            if not month_points:
                continue

            item_copy = json.loads(json.dumps(item))
            item_copy["object"]["vectorDataPoint"] = month_points
            month_items.append(item_copy)

        if month_items:
            month_payloads[(year, month)] = month_items

    if not month_payloads:
        raise AirflowSkipException("No monthly payloads prepared for Bronze")

    new_partition_count = 0
    for (year, month), month_items in month_payloads.items():
        s3_key = (
            f"bronze/canadian_grocery_prices/{product_name}/"
            f"year={year}/month={month}/raw.json"
        )

        try:
            obj = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
            existing_payload = json.loads(obj["Body"].read())
            existing_ref_per = (
                existing_payload[0]["object"]["vectorDataPoint"][0]["refPer"]
                if existing_payload else None
            )

            if existing_ref_per == f"{year}-{month}-01":
                print(f"Bronze already up to date: {s3_key}")
                continue
        except s3.exceptions.NoSuchKey:
            pass
        except s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] not in {"404", "NoSuchKey"}:
                raise

        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=json.dumps(month_items),
            ContentType="application/json"
        )
        new_partition_count += 1
        print(f"Uploaded Bronze: s3://{BUCKET_NAME}/{s3_key}")

    if new_partition_count == 0:
        raise AirflowSkipException("Bronze monthly partitions already up to date")


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
    Transform Bronze monthly JSON partitions into cumulative yearly Silver
    Parquet files while preserving previously stored months.
    """

    s3 = boto3.client("s3")

    bronze_keys = {
        "avocado": "bronze/canadian_grocery_prices/avocado/",
        "tomato": "bronze/canadian_grocery_prices/tomato/",
    }

    for product_name, bronze_prefix in bronze_keys.items():
        resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=bronze_prefix)
        partition_keys = sorted(
            obj["Key"] for obj in resp["Contents"]
            if obj["Key"].endswith("raw.json")
        )

        if not partition_keys:
            raise AirflowSkipException(
                f"No Bronze monthly partitions found for {product_name}"
            )

        yearly_frames = {}

        for bronze_key in partition_keys:
            print(f"Reading Bronze: s3://{BUCKET_NAME}/{bronze_key}")

            obj = s3.get_object(Bucket=BUCKET_NAME, Key=bronze_key)
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
                    month = date.split("-")[1]
                    price = dp["value"]

                    rows.append({
                        "year": year,
                        "month": month,
                        "date": date,
                        "price": price
                    })

            if not rows:
                print(f"No Canada rows found in Bronze partition: {bronze_key}")
                continue

            df = pd.DataFrame(rows).drop_duplicates(subset=["date"], keep="last")
            print(f"{product_name} sample from {bronze_key}:")
            print(df.head())

            year = df["year"].iloc[0]
            yearly_frames.setdefault(year, []).append(df[["date", "price"]].copy())

        if not yearly_frames:
            raise AirflowSkipException(
                f"No Canada rows were produced for {product_name}"
            )

        for year, frames in yearly_frames.items():
            combined_df = pd.concat(frames, ignore_index=True)

            silver_key = (
                f"silver/canadian_grocery_prices/{product_name}/"
                f"year={year}/data.parquet"
            )

            try:
                existing_obj = s3.get_object(Bucket=BUCKET_NAME, Key=silver_key)
                existing_df = pd.read_parquet(BytesIO(existing_obj["Body"].read()))
                combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
                print(
                    f"Merging with existing Silver year file: "
                    f"s3://{BUCKET_NAME}/{silver_key}"
                )
            except s3.exceptions.NoSuchKey:
                print(f"No Silver year file found yet for {product_name} {year}")
            except s3.exceptions.ClientError as e:
                if e.response["Error"]["Code"] not in {"404", "NoSuchKey"}:
                    raise
                print(f"No Silver year file found yet for {product_name} {year}")

            combined_df = (
                combined_df
                .drop_duplicates(subset=["date"], keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )

            buffer = BytesIO()
            combined_df.to_parquet(buffer, index=False)

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
