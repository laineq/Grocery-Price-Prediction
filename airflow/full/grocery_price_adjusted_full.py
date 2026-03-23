from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
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
START_YEAR = 2017
END_YEAR = 2025


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


def transform_to_silver():
    """
    Join grocery price data with CPI data and store adjusted price datasets
    in the Silver layer.
    """

    s3 = boto3.client("s3")

    for product in PRODUCTS:
        grocery_prefix = f"silver/canadian_grocery_prices/{product}/"
        cpi_prefix = f"silver/consumer_price_index/{product}/"

        grocery_df = read_partitioned_dataset(s3, grocery_prefix)
        cpi_df = read_partitioned_dataset(s3, cpi_prefix)

        grocery_df = grocery_df.rename(columns={"price": "price_before_adjustment"})
        cpi_df = cpi_df.rename(columns={"value": "cpi"})

        merged_df = pd.merge(
            grocery_df,
            cpi_df,
            on="date",
            how="inner",
        )
        merged_df["year"] = merged_df["date"].str[:4].astype(int)
        merged_df = merged_df[
            (merged_df["year"] >= START_YEAR) &
            (merged_df["year"] <= END_YEAR)
        ].copy()

        base_cpi = merged_df.loc[
            merged_df["date"] == BASE_DATE,
            "cpi",
        ].iloc[0]

        merged_df["price_adjusted"] = (
            merged_df["price_before_adjustment"] * base_cpi / merged_df["cpi"]
        )
        merged_df["price_adjusted"] = merged_df["price_adjusted"].round(2)
        merged_df["year"] = merged_df["year"].astype(str)

        print(f"{product} adjusted sample:")
        print(merged_df.head())
        print(f"Total rows for {product}: {len(merged_df):,}")

        for year, group in merged_df.groupby("year"):
            buffer = BytesIO()
            group.drop(columns=["year"]).to_parquet(buffer, index=False)

            silver_key = (
                f"silver/grocery_price_adjusted/{product}/"
                f"year={year}/data.parquet"
            )

            s3.put_object(
                Bucket=BUCKET_NAME,
                Key=silver_key,
                Body=buffer.getvalue(),
                ContentType="application/octet-stream",
            )

            print(f"Uploaded: s3://{BUCKET_NAME}/{silver_key}")


# Default DAG configuration
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}


# Airflow DAG definition (manual run only)
with DAG(
    dag_id="grocery_price_adjusted_full_history",
    default_args=default_args,
    schedule=None,
    catchup=False,
) as dag:

    # Task 1: Join grocery price and CPI datasets → adjusted Silver
    transform_task = PythonOperator(
        task_id="transform_to_silver",
        python_callable=transform_to_silver,
    )
