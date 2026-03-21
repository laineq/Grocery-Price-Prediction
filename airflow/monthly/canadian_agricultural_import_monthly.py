from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
from datetime import datetime
import os
from io import BytesIO
from zipfile import ZipFile

import boto3
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# S3 bucket name (configured via docker-compose environment variable)
BUCKET_NAME = os.environ["BUCKET_NAME"]

# Current-year dataset metadata
YEAR = 2026
ZIP_URL = (
    "https://www150.statcan.gc.ca/n1/pub/71-607-x/2021004/zip/"
    f"CIMT-CICM_Imp_{YEAR}.zip"
)

# Bronze naming convention
ZIP_FILENAME = f"CIMT-CICM_Imp_{YEAR}.zip"
BRONZE_PREFIX = f"bronze/canadian_agricultural_import/year={YEAR}/"

# Product codes used for Silver transformations
AVOCADO_HS6 = "080440"
TOMATO_HS6 = "070200"


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
        allowed_methods=["GET"],
    )

    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "GroceryPrediction-Airflow/1.0"})
    return session


def find_latest_csv_member(members):
    """
    Pick the latest monthly ODPFN015 CSV from the annual ZIP archive.
    """

    # The current-year ZIP contains a new ODPFN015 monthly snapshot
    # whenever Statistics Canada publishes another month
    matching_members = sorted(
        name for name in members
        if os.path.basename(name).startswith(f"ODPFN015_{YEAR}")
        and name.lower().endswith(".csv")
    )

    if not matching_members:
        raise FileNotFoundError(
            f"No CSV starting with ODPFN015_{YEAR} found in ZIP archive"
        )

    return matching_members[-1]


def save_bronze_task():
    """
    Download the latest annual ZIP snapshot for the current year and save it
    to Bronze only when a new monthly ODPFN015 CSV appears.
    """

    print(f"Downloading ZIP file for {YEAR}: {ZIP_URL}")

    session = build_retry_session()
    buffer = BytesIO()

    with session.get(ZIP_URL, stream=True, timeout=(60, 900)) as resp:
        resp.raise_for_status()

        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                buffer.write(chunk)

    zip_bytes = buffer.getvalue()
    print(f"Downloaded ZIP size for {YEAR}: {len(zip_bytes):,} bytes")

    with ZipFile(BytesIO(zip_bytes)) as zip_file:
        members = zip_file.namelist()
        print(f"ZIP contains {len(members)} files for {YEAR}")

        target_member = find_latest_csv_member(members)
        target_csv_filename = os.path.basename(target_member)
        csv_bytes = zip_file.read(target_member)

        print(
            f"Selected latest CSV {target_csv_filename} from {target_member} "
            f"({len(csv_bytes):,} bytes)"
        )

    s3 = boto3.client("s3")

    csv_s3_key = f"{BRONZE_PREFIX}{target_csv_filename}"
    zip_s3_key = f"{BRONZE_PREFIX}{ZIP_FILENAME}"

    # Skip ingestion if the latest monthly CSV is already stored in Bronze
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=csv_s3_key)
        print(f"Bronze already exists: {csv_s3_key}")
        raise AirflowSkipException("Latest monthly agricultural import already ingested")
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("New month detected. Saving Bronze...")
        else:
            raise

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=zip_s3_key,
        Body=zip_bytes,
        ContentType="application/zip",
    )
    print(f"Uploaded ZIP: s3://{BUCKET_NAME}/{zip_s3_key}")

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=csv_s3_key,
        Body=csv_bytes,
        ContentType="text/csv",
    )
    print(f"Uploaded CSV: s3://{BUCKET_NAME}/{csv_s3_key}")


def transform_to_silver():
    """
    Transform the latest current-year Bronze CSV into yearly Silver datasets.
    """

    s3 = boto3.client("s3")

    # Keep only partners needed for downstream modeling
    country_map = {
        "US": "USA",
        "MX": "Mexico",
    }

    # Read only the columns required for monthly quantity aggregation
    usecols = [
        "YearMonth/AnnéeMois",
        "HS6",
        "Country/Pays",
        "Quantity/Quantité",
    ]

    print(f"Searching latest Bronze CSV for {YEAR}: s3://{BUCKET_NAME}/{BRONZE_PREFIX}")

    resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=BRONZE_PREFIX)
    csv_keys = sorted(
        obj["Key"] for obj in resp.get("Contents", [])
        if obj["Key"].lower().endswith(".csv")
        and os.path.basename(obj["Key"]).startswith(f"ODPFN015_{YEAR}")
    )

    if not csv_keys:
        raise FileNotFoundError(f"No Bronze CSV found under {BRONZE_PREFIX}")

    # Use the latest current-year monthly snapshot stored in Bronze
    bronze_key = csv_keys[-1]
    print(f"Reading Bronze CSV: s3://{BUCKET_NAME}/{bronze_key}")

    obj = s3.get_object(Bucket=BUCKET_NAME, Key=bronze_key)

    # Accumulate monthly totals in memory instead of loading the full
    # current-year CSV at once to reduce peak memory usage
    avocado_totals = {}
    tomato_totals = {}

    for chunk in pd.read_csv(
        obj["Body"],
        usecols=usecols,
        dtype=str,
        chunksize=200_000,
    ):
        chunk["HS6"] = chunk["HS6"].str.zfill(6)
        chunk["date"] = pd.to_datetime(
            chunk["YearMonth/AnnéeMois"],
            format="%Y%m",
            errors="coerce",
        ).dt.strftime("%Y-%m")
        chunk["qty"] = pd.to_numeric(
            chunk["Quantity/Quantité"], errors="coerce"
        ).fillna(0)

        # Avocado dataset keeps only imports from Mexico
        avocado_chunk = (
            chunk[
                (chunk["HS6"] == AVOCADO_HS6) &
                (chunk["Country/Pays"] == "MX") &
                (chunk["date"].notna())
            ]
            .groupby("date")["qty"]
            .sum()
        )
        for date, qty in avocado_chunk.items():
            avocado_totals[date] = avocado_totals.get(date, 0) + qty

        # Tomato dataset keeps only USA and Mexico partners
        tomato_chunk = chunk[
            (chunk["HS6"] == TOMATO_HS6) &
            (chunk["Country/Pays"].isin(country_map.keys())) &
            (chunk["date"].notna())
        ].copy()
        tomato_chunk["partner"] = tomato_chunk["Country/Pays"].map(country_map)
        tomato_chunk = tomato_chunk.groupby(["date", "partner"])["qty"].sum()
        for (date, partner), qty in tomato_chunk.items():
            key = (date, partner)
            tomato_totals[key] = tomato_totals.get(key, 0) + qty

    avocado_df = pd.DataFrame(
        [
            {"date": date, "qty": qty}
            for date, qty in sorted(avocado_totals.items())
        ]
    ).sort_values("date")

    # Overwrite the current-year Silver file with the latest yearly snapshot
    avocado_buffer = BytesIO()
    avocado_df.to_parquet(avocado_buffer, index=False)
    avocado_key = (
        f"silver/canadian_agricultural_import/"
        f"avocado/year={YEAR}/data.parquet"
    )
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=avocado_key,
        Body=avocado_buffer.getvalue(),
        ContentType="application/octet-stream",
    )
    print(f"Uploaded Avocado Silver: s3://{BUCKET_NAME}/{avocado_key}")

    # Overwrite the current-year Silver file with the latest yearly snapshot
    tomato_df = pd.DataFrame(
        [
            {"date": date, "partner": partner, "qty": qty}
            for (date, partner), qty in sorted(tomato_totals.items())
        ]
    ).sort_values(["date", "partner"])

    tomato_buffer = BytesIO()
    tomato_df.to_parquet(tomato_buffer, index=False)
    tomato_key = (
        f"silver/canadian_agricultural_import/"
        f"tomato/year={YEAR}/data.parquet"
    )
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=tomato_key,
        Body=tomato_buffer.getvalue(),
        ContentType="application/octet-stream",
    )
    print(f"Uploaded Tomato Silver: s3://{BUCKET_NAME}/{tomato_key}")


# Default DAG configuration
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 3,
}


# Airflow DAG definition
with DAG(
    dag_id="canadian_agricultural_import_monthly",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
) as dag:

    # Task 1: Check latest current-year ZIP/CSV → Bronze
    fetch_task = PythonOperator(
        task_id="save_bronze_task",
        python_callable=save_bronze_task,
    )

    # Task 2: Transform latest Bronze snapshot → yearly Silver datasets
    transform_task = PythonOperator(
        task_id="transform_to_silver",
        python_callable=transform_to_silver,
    )

    # Task dependency
    fetch_task >> transform_task
