from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime
from io import BytesIO
import boto3
import os
import pandas as pd


# S3 bucket name (configured via docker-compose environment variable)
BUCKET_NAME = os.environ["BUCKET_NAME"]

# Products to adjust with CPI
PRODUCTS = ["avocado", "tomato"]
BASE_DATE = "2017-01"
CURRENT_YEAR = 2026


def read_parquet_from_s3(s3, key):
    """
    Read a Parquet file from S3 into a DataFrame.
    """

    print(f"Reading: s3://{BUCKET_NAME}/{key}")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))


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
        frames.append(read_parquet_from_s3(s3, key))

    return pd.concat(frames, ignore_index=True)


def transform_to_silver():
    """
    Join current-year grocery price data with CPI data and overwrite the
    current-year adjusted price datasets in the Silver layer.
    """

    s3 = boto3.client("s3")

    for product in PRODUCTS:
        grocery_key = (
            f"silver/canadian_grocery_prices/{product}/"
            f"year={CURRENT_YEAR}/data.parquet"
        )
        cpi_key = (
            f"silver/consumer_price_index/{product}/"
            f"year={CURRENT_YEAR}/data.parquet"
        )
        adjusted_key = (
            f"silver/grocery_price_adjusted/{product}/"
            f"year={CURRENT_YEAR}/data.parquet"
        )

        grocery_df = read_parquet_from_s3(s3, grocery_key)
        current_cpi_df = read_parquet_from_s3(s3, cpi_key)
        full_cpi_df = read_partitioned_dataset(
            s3,
            f"silver/consumer_price_index/{product}/",
        )

        base_cpi_rows = full_cpi_df.loc[full_cpi_df["date"] == BASE_DATE, "value"]
        if base_cpi_rows.empty:
            raise FileNotFoundError(
                f"Base CPI date {BASE_DATE} was not found for {product}"
            )

        base_cpi = pd.to_numeric(base_cpi_rows, errors="coerce").iloc[0]

        grocery_df = grocery_df.rename(columns={"price": "price_before_adjustment"})
        current_cpi_df = current_cpi_df.rename(columns={"value": "cpi"})
        current_cpi_df["cpi"] = pd.to_numeric(current_cpi_df["cpi"], errors="coerce")
        grocery_df["price_before_adjustment"] = pd.to_numeric(
            grocery_df["price_before_adjustment"],
            errors="coerce",
        )

        merged_df = pd.merge(
            grocery_df,
            current_cpi_df,
            on="date",
            how="inner",
        ).sort_values("date").reset_index(drop=True)

        if merged_df.empty:
            raise FileNotFoundError(
                f"No overlapping grocery price and CPI data found for "
                f"{product} in {CURRENT_YEAR}"
            )

        latest_common_date = merged_df["date"].iloc[-1]
        print(
            f"Latest common month for {product} in {CURRENT_YEAR}: "
            f"{latest_common_date}"
        )

        try:
            obj = s3.get_object(Bucket=BUCKET_NAME, Key=adjusted_key)
            existing_df = pd.read_parquet(BytesIO(obj["Body"].read()))
            existing_latest_date = (
                existing_df["date"].iloc[-1] if not existing_df.empty else None
            )

            if existing_latest_date == latest_common_date:
                print(f"Silver already up to date: s3://{BUCKET_NAME}/{adjusted_key}")
                raise AirflowSkipException(
                    f"Current-year adjusted grocery price already up to date "
                    f"for {product}"
                )

            print(
                f"New month detected. Existing latest month: "
                f"{existing_latest_date}, new latest month: {latest_common_date}"
            )
        except s3.exceptions.NoSuchKey:
            print("No current-year adjusted Silver snapshot found. Saving Silver...")
        except s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] in {"404", "NoSuchKey"}:
                print("No current-year adjusted Silver snapshot found. Saving Silver...")
            else:
                raise

        merged_df["price_adjusted"] = (
            merged_df["price_before_adjustment"] * base_cpi / merged_df["cpi"]
        )
        merged_df["price_adjusted"] = merged_df["price_adjusted"].round(2)

        print(f"{product} adjusted sample:")
        print(merged_df.head())
        print(f"Total rows for {product}: {len(merged_df):,}")

        buffer = BytesIO()
        merged_df.to_parquet(buffer, index=False)

        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=adjusted_key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )

        print(f"Uploaded Silver: s3://{BUCKET_NAME}/{adjusted_key}")


# Default DAG configuration
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}


# Airflow DAG definition
with DAG(
    dag_id="grocery_price_adjusted_monthly",
    default_args=default_args,
    schedule=None,
    catchup=False,
) as dag:

    # Task 1: Join current-year grocery price and CPI datasets → adjusted Silver
    transform_task = PythonOperator(
        task_id="transform_to_silver",
        python_callable=transform_to_silver,
    )

    trigger_gold_task = TriggerDagRunOperator(
        task_id="trigger_gold_features_monthly",
        trigger_dag_id="gold_features_monthly",
        wait_for_completion=False,
    )

    transform_task >> trigger_gold_task
