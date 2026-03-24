from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
from datetime import datetime
from io import BytesIO
import requests
import json
import boto3
import os
import pandas as pd


# S3 bucket name (configured via docker-compose environment variable)
BUCKET_NAME = os.environ["BUCKET_NAME"]

# EIA API endpoint
API_URL = "https://api.eia.gov/v2/petroleum/pri/gnd/data/"
CURRENT_YEAR = 2026

# Static request parameters
API_PARAMS = {
    "frequency": "monthly",
    "data[0]": "value",
    "facets[series][]": "EMD_EPD2D_PTE_NUS_DPG",
    "start": "2016-01",
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


def fetch_json(url, params=None):
    """
    Download a JSON payload from a public API endpoint.
    """

    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def save_us_bronze_task():
    """
    Fetch the current-year U.S. diesel price snapshot from the EIA API
    and store raw JSON into the S3 Bronze layer.
    """

    params = API_PARAMS.copy()
    params["end"] = datetime.now().strftime("%Y-%m")

    print("Requesting latest EIA oil price data...")

    # Call EIA API
    payload = fetch_json(API_URL, params=params)

    data = payload.get("response", {}).get("data", [])
    if not data:
        raise Exception("No oil price data returned from EIA API")

    # Keep only the current-year snapshot
    current_year_data = [
        row for row in data
        if row.get("period", "").startswith(f"{CURRENT_YEAR}-")
    ]

    if not current_year_data:
        raise Exception(f"No oil price data found for {CURRENT_YEAR}")

    latest_period = current_year_data[-1]["period"]
    print(
        f"Latest month detected for {CURRENT_YEAR}: {latest_period} "
        f"({len(current_year_data)} rows in current-year snapshot)"
    )

    bronze_payload = {
        "response": {
            "total": str(len(current_year_data)),
            "dateFormat": payload.get("response", {}).get("dateFormat"),
            "frequency": payload.get("response", {}).get("frequency"),
            "data": current_year_data,
        }
    }

    # Create S3 client
    s3 = boto3.client("s3")

    # Bronze storage path (partitioned by year)
    s3_key = f"bronze/oil_prices/us/year={CURRENT_YEAR}/raw.json"

    # Skip if the stored snapshot already contains the same latest month
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        existing_payload = json.loads(obj["Body"].read())
        existing_data = existing_payload.get("response", {}).get("data", [])
        existing_latest_period = (
            existing_data[-1]["period"] if existing_data else None
        )

        if existing_latest_period == latest_period:
            print(f"Bronze already up to date: {s3_key}")
            raise AirflowSkipException("Current-year snapshot already up to date")

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

    # Upload raw JSON to S3 Bronze layer
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json.dumps(bronze_payload),
        ContentType="application/json",
    )

    print(f"Uploaded Bronze: s3://{BUCKET_NAME}/{s3_key}")


def save_canada_bronze_task():
    """
    Fetch the latest monthly Canada city-level diesel price data from
    Statistics Canada and store raw JSON into a year/month Bronze partition.
    """

    vectors = ",".join(f'"{vector_id}"' for vector_id in CANADA_VECTOR_META.keys())
    params = {
        "vectorIds": vectors,
        "startRefPeriod": f"{CURRENT_YEAR}-01-01",
        "endReferencePeriod": datetime.now().strftime("%Y-%m-01"),
    }

    print("Requesting latest Statistics Canada oil price data for Canada cities...")

    payload = fetch_json(CANADA_API_URL, params=params)

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

            if not ref_per.startswith(f"{CURRENT_YEAR}-") or value is None:
                continue

            rows.append({
                "date": ref_per[:7],
                "city": city,
                "value": value,
            })

    if not rows:
        raise Exception(f"No Canada oil price data found for {CURRENT_YEAR}")

    latest_month = max(row["date"] for row in rows)
    year, month = latest_month.split("-")
    latest_rows = [row for row in rows if row["date"] == latest_month]
    latest_rows = (
        pd.DataFrame(latest_rows)
        .sort_values(["date", "city"])
        .to_dict(orient="records")
    )

    print(
        f"Latest Canada month detected: {latest_month} "
        f"({len(latest_rows)} rows in latest monthly snapshot)"
    )

    s3 = boto3.client("s3")
    s3_key = f"bronze/oil_prices/canada/year={year}/month={month}/raw.json"

    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=s3_key)
        print(f"Bronze already exists: {s3_key}")
        raise AirflowSkipException("Latest Canada month already ingested")
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in {"404", "NoSuchKey"}:
            print(f"New Canada month detected. Saving Bronze for {year}-{month}...")
        else:
            raise

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json.dumps(latest_rows, ensure_ascii=False),
        ContentType="application/json",
    )

    print(f"Uploaded Bronze: s3://{BUCKET_NAME}/{s3_key}")


def read_parquet_from_s3(s3, key):
    """
    Read a Parquet file from S3 into a DataFrame.
    """

    print(f"Reading: s3://{BUCKET_NAME}/{key}")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))


def s3_key_exists(s3, key):
    """
    Check whether an S3 object exists.
    """

    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=key)
        return True
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in {"404", "NoSuchKey"}:
            return False
        raise


def transform_us_to_silver():
    """
    Convert the current-year U.S. diesel price snapshot into CAD/L Silver.
    """

    s3 = boto3.client("s3")

    bronze_key = f"bronze/oil_prices/us/year={CURRENT_YEAR}/raw.json"
    fx_key = f"silver/exchange_rate/year={CURRENT_YEAR}/data.parquet"

    print(f"Reading Bronze: s3://{BUCKET_NAME}/{bronze_key}")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=bronze_key)
    payload = json.loads(obj["Body"].read())

    us_df = pd.DataFrame(payload.get("response", {}).get("data", []))
    us_df = us_df[["period", "value"]].rename(
        columns={"period": "date", "value": "usd"}
    )
    us_df["usd"] = pd.to_numeric(us_df["usd"], errors="coerce")

    fx_df = read_parquet_from_s3(s3, fx_key)
    fx_df["USD_CAD"] = pd.to_numeric(fx_df["USD_CAD"], errors="coerce")

    merged_df = pd.merge(
        us_df,
        fx_df[["date", "USD_CAD"]],
        on="date",
        how="inner",
    ).sort_values("date").reset_index(drop=True)
    merged_df["us_cad_l"] = merged_df["usd"] * merged_df["USD_CAD"] / 3.78541

    buffer = BytesIO()
    merged_df.to_parquet(buffer, index=False)

    silver_key = f"silver/oil_prices/us/year={CURRENT_YEAR}/data.parquet"
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=silver_key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream",
    )

    print(f"Uploaded Silver: s3://{BUCKET_NAME}/{silver_key}")


def transform_canada_to_silver():
    """
    Aggregate current-year Canada city-level raw partitions into CAD/L Silver.
    """

    s3 = boto3.client("s3")
    bronze_prefix = f"bronze/oil_prices/canada/year={CURRENT_YEAR}/"

    resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=bronze_prefix)
    bronze_keys = sorted(
        obj["Key"] for obj in resp.get("Contents", [])
        if obj["Key"].endswith("raw.json")
    )

    if not bronze_keys:
        raise FileNotFoundError(f"No Canada Bronze data found under {bronze_prefix}")

    frames = []
    for bronze_key in bronze_keys:
        print(f"Reading Bronze: s3://{BUCKET_NAME}/{bronze_key}")
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=bronze_key)
        frames.append(pd.DataFrame(json.loads(obj["Body"].read())))

    df = pd.concat(frames, ignore_index=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = (
        df.groupby("date", as_index=False)["value"]
        .mean()
        .rename(columns={"value": "ca_cad_l"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    df["ca_cad_l"] = df["ca_cad_l"] / 100

    buffer = BytesIO()
    df.to_parquet(buffer, index=False)

    silver_key = f"silver/oil_prices/canada/year={CURRENT_YEAR}/data.parquet"
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=silver_key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream",
    )

    print(f"Uploaded Silver: s3://{BUCKET_NAME}/{silver_key}")


def combine_oil_prices_to_silver():
    """
    Combine current-year U.S. and Canada oil prices into integrated Silver.
    """

    s3 = boto3.client("s3")

    us_key = f"silver/oil_prices/us/year={CURRENT_YEAR}/data.parquet"
    canada_key = f"silver/oil_prices/canada/year={CURRENT_YEAR}/data.parquet"

    if not s3_key_exists(s3, us_key):
        raise AirflowSkipException(
            f"U.S. Silver dataset does not exist yet for {CURRENT_YEAR}"
        )

    if not s3_key_exists(s3, canada_key):
        raise AirflowSkipException(
            f"Canada Silver dataset does not exist yet for {CURRENT_YEAR}"
        )

    us_df = read_parquet_from_s3(s3, us_key)
    canada_df = read_parquet_from_s3(s3, canada_key)

    combined_df = pd.merge(
        us_df[["date", "us_cad_l"]],
        canada_df[["date", "ca_cad_l"]],
        on="date",
        how="inner",
    ).sort_values("date").reset_index(drop=True)

    if combined_df.empty:
        raise AirflowSkipException(
            f"No overlapping U.S. and Canada oil price months found for {CURRENT_YEAR}"
        )

    latest_common_date = combined_df["date"].iloc[-1]
    combined_df["integrated_gas_price"] = (
        0.8 * combined_df["us_cad_l"] + 0.2 * combined_df["ca_cad_l"]
    )

    integrated_key = f"silver/oil_prices/integrated/year={CURRENT_YEAR}/data.parquet"

    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=integrated_key)
        existing_df = pd.read_parquet(BytesIO(obj["Body"].read()))
        existing_latest_date = (
            existing_df["date"].iloc[-1] if not existing_df.empty else None
        )

        if existing_latest_date == latest_common_date:
            print(f"Silver already up to date: s3://{BUCKET_NAME}/{integrated_key}")
            raise AirflowSkipException("Current-year integrated oil price already up to date")

        print(
            f"New month detected. Existing latest month: "
            f"{existing_latest_date}, new latest month: {latest_common_date}"
        )
    except s3.exceptions.NoSuchKey:
        print("No current-year integrated Silver snapshot found. Saving Silver...")
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in {"404", "NoSuchKey"}:
            print("No current-year integrated Silver snapshot found. Saving Silver...")
        else:
            raise

    buffer = BytesIO()
    combined_df.to_parquet(buffer, index=False)

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=integrated_key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream",
    )

    print(f"Uploaded Silver: s3://{BUCKET_NAME}/{integrated_key}")


# Default DAG configuration
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}


# Airflow DAG definition
with DAG(
    dag_id="oil_prices_monthly",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
) as dag:

    # Task 1: Fetch latest U.S. monthly data → Bronze
    us_fetch_task = PythonOperator(
        task_id="save_us_bronze_task",
        python_callable=save_us_bronze_task,
    )

    # Task 2: Fetch latest Canada city-level monthly data → Bronze
    canada_fetch_task = PythonOperator(
        task_id="save_canada_bronze_task",
        python_callable=save_canada_bronze_task,
    )

    # Task 3: Convert current-year U.S. raw prices into CAD/L Silver
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
        trigger_rule="none_failed_min_one_success",
    )

    us_fetch_task >> us_transform_task
    canada_fetch_task >> canada_transform_task
    [us_transform_task, canada_transform_task] >> combine_task
