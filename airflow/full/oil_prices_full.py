from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
from io import BytesIO
import requests
import json
import boto3
import os
import pandas as pd


# S3 bucket name (configured via docker-compose environment variable)
BUCKET_NAME = os.environ["BUCKET_NAME"]

# EIA API endpoint and request parameters
API_URL = "https://api.eia.gov/v2/petroleum/pri/gnd/data/"
API_PARAMS = {
    "frequency": "monthly",
    "data[0]": "value",
    "facets[series][]": "EMD_EPD2D_PTE_NUS_DPG",
    "start": "2016-01",
    "end": "2026-03",
    "sort[0][column]": "period",
    "sort[0][direction]": "asc",
    "length": 5000,
    "api_key": "1LagLJXXow2sdsFkMzBHgxrdYeiZYwqJAiakHmVo",
}

# Statistics Canada vector metadata for city-level Canada diesel prices
CANADA_VECTOR_META = {
    "65584802": "St. John's",
    "735144": "Charlottetown and Summerside",
    "65584803": "Halifax",
    "65584804": "Saint John",
    "735145": "Québec",
    "735146": "Montréal",
    "735147": "Ottawa-Gatineau",
    "735148": "Toronto",
    "735135": "Winnipeg",
    "735136": "Regina",
    "735137": "Saskatoon",
    "735138": "Edmonton",
    "735139": "Calgary",
    "735140": "Vancouver",
    "735141": "Victoria",
    "735142": "Whitehorse",
    "735143": "Yellowknife",
}

CANADA_API_URL = (
    "https://www150.statcan.gc.ca/t1/wds/rest/"
    "getDataFromVectorByReferencePeriodRange"
)
START_YEAR = 2016
END_YEAR = 2025


def save_us_bronze_full_task():
    """
    Fetch full historical U.S. diesel price data from the EIA API,
    keep the 2016-2025 range, and store raw JSON in the S3 Bronze layer.
    """

    print("Requesting EIA oil price data...")

    # Call EIA API
    resp = requests.get(API_URL, params=API_PARAMS, timeout=120)
    resp.raise_for_status()
    payload = resp.json()

    data = payload.get("response", {}).get("data", [])
    print(f"Total rows received: {len(data):,}")

    # Keep only the target history range for Bronze backfill
    filtered_data = [
        row for row in data
        if "2016-01" <= row.get("period", "") <= "2025-12"
    ]

    print(f"Filtered rows (2016-01 → 2025-12): {len(filtered_data):,}")

    bronze_payload = {
        "response": {
            "total": str(len(filtered_data)),
            "dateFormat": payload.get("response", {}).get("dateFormat"),
            "frequency": payload.get("response", {}).get("frequency"),
            "data": filtered_data,
        }
    }

    # Create S3 client
    s3 = boto3.client("s3")

    # Store as a single raw Bronze file (full snapshot)
    s3_key = "bronze/oil_prices/us/full_history/raw_2016_2025.json"

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json.dumps(bronze_payload),
        ContentType="application/json",
    )

    print(f"Uploaded FULL Bronze: s3://{BUCKET_NAME}/{s3_key}")


def save_canada_bronze_full_task():
    """
    Fetch full historical Canada city-level diesel price data from
    Statistics Canada and store raw JSON in the S3 Bronze layer.
    """

    vectors = ",".join(f'"{vector_id}"' for vector_id in CANADA_VECTOR_META.keys())
    params = {
        "vectorIds": vectors,
        "startRefPeriod": "2016-01-01",
        "endReferencePeriod": "2025-12-31",
    }

    print("Requesting Statistics Canada oil price data for Canada cities...")

    resp = requests.get(CANADA_API_URL, params=params, timeout=120)
    resp.raise_for_status()
    payload = resp.json()

    rows = []
    for item in payload:
        if item.get("status") != "SUCCESS":
            continue

        obj = item.get("object", {})
        vector_id = str(obj.get("vectorId"))
        city = CANADA_VECTOR_META.get(vector_id, "Unknown")

        for dp in obj.get("vectorDataPoint", []):
            ref_per = dp.get("refPer", "")
            value = dp.get("value")

            if not ref_per or value is None:
                continue

            rows.append({
                "date": ref_per[:7],
                "city": city,
                "value": value,
            })

    print(f"Total city-month rows received: {len(rows):,}")
    rows = (
        pd.DataFrame(rows)
        .sort_values(["date", "city"])
        .to_dict(orient="records")
    )

    s3 = boto3.client("s3")
    s3_key = "bronze/oil_prices/canada/full_history/raw_2016_2025.json"

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json.dumps(rows, ensure_ascii=False),
        ContentType="application/json",
    )

    print(f"Uploaded FULL Bronze: s3://{BUCKET_NAME}/{s3_key}")


def read_partitioned_dataset(s3, prefix):
    """
    Read all Parquet partitions under a given S3 prefix into one DataFrame.
    """

    resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    keys = sorted(
        obj["Key"] for obj in resp.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    )

    if not keys:
        raise FileNotFoundError(f"No parquet files found under {prefix}")

    frames = []
    for key in keys:
        print(f"Reading: s3://{BUCKET_NAME}/{key}")
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        frames.append(pd.read_parquet(BytesIO(obj["Body"].read())))

    return pd.concat(frames, ignore_index=True)


def transform_us_to_silver():
    """
    Convert U.S. diesel prices into CAD/L using exchange rate Silver data.
    """

    s3 = boto3.client("s3")

    bronze_key = "bronze/oil_prices/us/full_history/raw_2016_2025.json"
    print(f"Reading Bronze: s3://{BUCKET_NAME}/{bronze_key}")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=bronze_key)
    payload = json.loads(obj["Body"].read())

    us_df = pd.DataFrame(payload.get("response", {}).get("data", []))
    us_df = us_df[["period", "value"]].rename(
        columns={"period": "date", "value": "usd"}
    )
    us_df["usd"] = pd.to_numeric(us_df["usd"], errors="coerce")

    fx_df = read_partitioned_dataset(s3, "silver/exchange_rate/")
    fx_df = fx_df[["date", "USD_CAD"]].copy()
    fx_df["USD_CAD"] = pd.to_numeric(fx_df["USD_CAD"], errors="coerce")

    merged_df = pd.merge(us_df, fx_df, on="date", how="inner")
    merged_df["us_cad_l"] = merged_df["usd"] * merged_df["USD_CAD"] / 3.78541
    merged_df["year"] = merged_df["date"].str[:4]
    merged_df = merged_df[
        (merged_df["year"].astype(int) >= START_YEAR) &
        (merged_df["year"].astype(int) <= END_YEAR)
    ].copy()

    for year, group in merged_df.groupby("year"):
        buffer = BytesIO()
        group.drop(columns=["year"]).to_parquet(buffer, index=False)

        silver_key = f"silver/oil_prices/us/year={year}/data.parquet"
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=silver_key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )

        print(f"Uploaded Silver: s3://{BUCKET_NAME}/{silver_key}")


def transform_canada_to_silver():
    """
    Aggregate city-level Canada diesel prices into a national CAD/L series.
    """

    s3 = boto3.client("s3")

    bronze_key = "bronze/oil_prices/canada/full_history/raw_2016_2025.json"
    print(f"Reading Bronze: s3://{BUCKET_NAME}/{bronze_key}")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=bronze_key)
    rows = json.loads(obj["Body"].read())

    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = (
        df.groupby("date", as_index=False)["value"]
        .mean()
        .rename(columns={"value": "ca_cad_l"})
    )
    df["ca_cad_l"] = df["ca_cad_l"] / 100
    df["year"] = df["date"].str[:4]

    for year, group in df.groupby("year"):
        buffer = BytesIO()
        group.drop(columns=["year"]).to_parquet(buffer, index=False)

        silver_key = f"silver/oil_prices/canada/year={year}/data.parquet"
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=silver_key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )

        print(f"Uploaded Silver: s3://{BUCKET_NAME}/{silver_key}")


def combine_oil_prices_to_silver():
    """
    Combine U.S. and Canada oil prices into the integrated gas price series.
    """

    s3 = boto3.client("s3")

    us_df = read_partitioned_dataset(s3, "silver/oil_prices/us/")
    canada_df = read_partitioned_dataset(s3, "silver/oil_prices/canada/")

    combined_df = pd.merge(
        us_df[["date", "us_cad_l"]],
        canada_df[["date", "ca_cad_l"]],
        on="date",
        how="inner",
    )
    combined_df["integrated_gas_price"] = (
        0.8 * combined_df["us_cad_l"] + 0.2 * combined_df["ca_cad_l"]
    )
    combined_df["year"] = combined_df["date"].str[:4]

    for year, group in combined_df.groupby("year"):
        buffer = BytesIO()
        group.drop(columns=["year"]).to_parquet(buffer, index=False)

        silver_key = f"silver/oil_prices/integrated/year={year}/data.parquet"
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


# Airflow DAG definition (manual run only)
with DAG(
    dag_id="oil_prices_full_history",
    default_args=default_args,
    schedule=None,
    catchup=False,
) as dag:

    # Task 1: Fetch full U.S. history → Bronze
    us_fetch_task = PythonOperator(
        task_id="save_us_bronze_full_task",
        python_callable=save_us_bronze_full_task,
    )

    # Task 2: Fetch full Canada city-level history → Bronze
    canada_fetch_task = PythonOperator(
        task_id="save_canada_bronze_full_task",
        python_callable=save_canada_bronze_full_task,
    )

    # Task 3: Convert U.S. raw prices into CAD/L Silver
    us_transform_task = PythonOperator(
        task_id="transform_us_to_silver",
        python_callable=transform_us_to_silver,
    )

    # Task 4: Aggregate Canada city-level raw prices into Silver
    canada_transform_task = PythonOperator(
        task_id="transform_canada_to_silver",
        python_callable=transform_canada_to_silver,
    )

    # Task 5: Combine U.S. and Canada Silver datasets
    combine_task = PythonOperator(
        task_id="combine_oil_prices_to_silver",
        python_callable=combine_oil_prices_to_silver,
    )

    us_fetch_task >> us_transform_task
    canada_fetch_task >> canada_transform_task
    [us_transform_task, canada_transform_task] >> combine_task
