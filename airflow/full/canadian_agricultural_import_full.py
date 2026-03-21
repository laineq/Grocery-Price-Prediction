from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
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

# Statistics Canada annual ZIP file pattern
ZIP_URL_TEMPLATE = (
    "https://www150.statcan.gc.ca/n1/pub/71-607-x/2021004/zip/"
    "CIMT-CICM_Imp_{year}.zip"
)

# Bronze naming convention
ZIP_FILENAME_TEMPLATE = "CIMT-CICM_Imp_{year}.zip"
START_YEAR = 2016
END_YEAR = 2025
S3_PREFIX_TEMPLATE = "bronze/canadian_agricultural_import/year={year}/"

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
    session.headers.update(
        {"User-Agent": "GroceryPrediction-Airflow/1.0"}
    )
    return session


def save_bronze_full_task():
    """
    Download annual Statistics Canada agricultural import ZIP files,
    extract the requested CSV from each archive, and upload both files
    into the S3 Bronze layer.
    """

    session = build_retry_session()
    s3 = boto3.client("s3")

    for year in range(START_YEAR, END_YEAR + 1):
        zip_url = ZIP_URL_TEMPLATE.format(year=year)
        zip_filename = ZIP_FILENAME_TEMPLATE.format(year=year)
        s3_prefix = S3_PREFIX_TEMPLATE.format(year=year)
        csv_stem = f"ODPFN015_{year}12"

        print(f"Downloading ZIP file for {year}: {zip_url}")

        buffer = BytesIO()

        with session.get(zip_url, stream=True, timeout=(60, 900)) as resp:
            resp.raise_for_status()

            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    buffer.write(chunk)

        zip_bytes = buffer.getvalue()
        print(f"Downloaded ZIP size for {year}: {len(zip_bytes):,} bytes")

        with ZipFile(BytesIO(zip_bytes)) as zip_file:
            members = zip_file.namelist()
            print(f"ZIP contains {len(members)} files for {year}")

            csv_members = [
                name for name in members
                if name.lower().endswith(".csv")
            ]

            matching_members = sorted(
                name for name in csv_members
                if os.path.basename(name).startswith(csv_stem)
            )

            print(
                f"Found {len(matching_members)} matching CSV files for "
                f"{year}: {[os.path.basename(name) for name in matching_members]}"
            )

            if len(matching_members) != 1:
                raise FileNotFoundError(
                    f"Expected exactly 1 CSV starting with {csv_stem} for "
                    f"{year}, but found: "
                    f"{[os.path.basename(name) for name in matching_members]}"
                )

            target_member = matching_members[0]
            target_csv_filename = os.path.basename(target_member)
            csv_bytes = zip_file.read(target_member)
            print(
                f"Extracted {target_csv_filename} from {target_member} "
                f"({len(csv_bytes):,} bytes)"
            )

        zip_s3_key = f"{s3_prefix}{zip_filename}"
        csv_s3_key = f"{s3_prefix}{target_csv_filename}"

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
    Transform annual Bronze CSV files into yearly Silver Parquet datasets.
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

    for year in range(START_YEAR, END_YEAR + 1):
        bronze_prefix = S3_PREFIX_TEMPLATE.format(year=year)
        print(f"Searching Bronze CSV for {year}: s3://{BUCKET_NAME}/{bronze_prefix}")

        # Each year folder should contain one selected ODPFN015 CSV file
        resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=bronze_prefix)
        csv_keys = sorted(
            obj["Key"] for obj in resp.get("Contents", [])
            if obj["Key"].lower().endswith(".csv")
            and f"ODPFN015_{year}12" in os.path.basename(obj["Key"])
        )

        if len(csv_keys) != 1:
            raise FileNotFoundError(
                f"Expected exactly 1 Bronze CSV for {year}, but found: {csv_keys}"
            )

        bronze_key = csv_keys[0]
        print(f"Reading Bronze CSV: s3://{BUCKET_NAME}/{bronze_key}")

        obj = s3.get_object(Bucket=BUCKET_NAME, Key=bronze_key)

        # Accumulate monthly totals in memory instead of loading the full
        # yearly CSV at once to reduce peak memory usage
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

        avocado_df = (
            pd.DataFrame(
                [
                    {"date": date, "qty": qty}
                    for date, qty in sorted(avocado_totals.items())
                ]
            )
            .sort_values("date")
        )

        # Write one Parquet file per year for avocado imports
        avocado_buffer = BytesIO()
        avocado_df.to_parquet(avocado_buffer, index=False)
        avocado_key = (
            f"silver/canadian_agricultural_import/"
            f"avocado/year={year}/data.parquet"
        )
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=avocado_key,
            Body=avocado_buffer.getvalue(),
            ContentType="application/octet-stream",
        )
        print(f"Uploaded Avocado Silver: s3://{BUCKET_NAME}/{avocado_key}")

        tomato_df = (
            pd.DataFrame(
                [
                    {"date": date, "partner": partner, "qty": qty}
                    for (date, partner), qty in sorted(tomato_totals.items())
                ]
            )
            .sort_values(["date", "partner"])
        )

        # Write one Parquet file per year for tomato imports
        tomato_buffer = BytesIO()
        tomato_df.to_parquet(tomato_buffer, index=False)
        tomato_key = (
            f"silver/canadian_agricultural_import/"
            f"tomato/year={year}/data.parquet"
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


# Airflow DAG definition (manual run only)
with DAG(
    dag_id="canadian_agricultural_import_full_history",
    default_args=default_args,
    schedule=None,
    catchup=False,
) as dag:

    # Task 1: Download annual ZIP/CSV files → Bronze
    fetch_task = PythonOperator(
        task_id="save_bronze_full_task",
        python_callable=save_bronze_full_task,
    )

    # Task 2: Transform Bronze CSV files → yearly Silver datasets
    transform_task = PythonOperator(
        task_id="transform_to_silver",
        python_callable=transform_to_silver,
    )

    # Task dependency
    fetch_task >> transform_task
