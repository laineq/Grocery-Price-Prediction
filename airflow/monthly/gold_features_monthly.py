from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
from datetime import datetime
from io import BytesIO
import boto3
import os
import pandas as pd


# S3 bucket name (configured via docker-compose environment variable)
BUCKET_NAME = os.environ["BUCKET_NAME"]

# Fixed lag configuration from feature engineering results
AVOCADO_LAGS = {
    "MEAN_C": 2,
    "PRECIPITATION_MM": 4,
    "MXN_CAD": 0,
    "USD_CAD": 0,
    "integrated_gas_price": 0,
    "import_qty": 2,
}

TOMATO_LAGS = {
    "MEAN_C": 0,
    "PRECIPITATION_MM": 6,
    "MXN_CAD": 0,
    "USD_CAD": 0,
    "integrated_gas_price": 0,
    "import_qty": 0,
}

START_DATE = "2017-01"
CURRENT_YEAR = 2026


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


def build_weather_features(weather_df):
    """
    Build tomato and avocado weather feature tables from Mexico weather Silver.
    """

    weather_df = weather_df.copy()
    weather_df["date"] = weather_df["PERIOD"].str[:7]

    tomato_weather = (
        weather_df[weather_df["STATE"] == "Sinaloa"][
            ["date", "MEAN_C", "PRECIPITATION_MM"]
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )

    mean_pivot = weather_df.pivot(index="date", columns="STATE", values="MEAN_C")
    precip_pivot = weather_df.pivot(
        index="date",
        columns="STATE",
        values="PRECIPITATION_MM",
    )

    avocado_weather = pd.DataFrame(index=mean_pivot.index)
    avocado_weather["MEAN_C"] = (
        0.7 * mean_pivot["Michoacán"] +
        0.2 * mean_pivot["Jalisco"] +
        0.1 * mean_pivot["Estado de México"]
    )
    avocado_weather["PRECIPITATION_MM"] = (
        0.7 * precip_pivot["Michoacán"] +
        0.2 * precip_pivot["Jalisco"] +
        0.1 * precip_pivot["Estado de México"]
    )
    avocado_weather = avocado_weather.reset_index()

    return avocado_weather, tomato_weather


def apply_lag_table(df, lag_map):
    """
    Apply fixed lags to each feature column.
    """

    result = pd.DataFrame(index=df.index)

    for column, lag in lag_map.items():
        result[column] = df[column].shift(lag) if lag > 0 else df[column]

    return result


def build_avocado_gold_features(
    avocado_import_df,
    avocado_weather_df,
    gas_df,
    fx_df,
    avocado_price_df,
):
    """
    Build the final avocado Gold feature table.
    """

    avocado_import_df = avocado_import_df.rename(columns={"qty": "import_qty"})
    avocado_import_df = avocado_import_df[["date", "import_qty"]].copy()
    avocado_target = avocado_price_df[["date", "price_adjusted"]].copy()

    avocado_features = (
        avocado_import_df
        .merge(avocado_weather_df, on="date", how="inner")
        .merge(gas_df, on="date", how="inner")
        .merge(fx_df, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )

    avocado_lagged = apply_lag_table(
        avocado_features.set_index("date"),
        AVOCADO_LAGS,
    )
    avocado_final = avocado_lagged.loc[avocado_lagged.index >= START_DATE].dropna()
    avocado_final = avocado_final.join(
        avocado_target.set_index("date"),
        how="inner",
    ).reset_index()

    return avocado_final.rename(columns={
        "date": "Date",
        "MEAN_C": "Mean Temperature",
        "PRECIPITATION_MM": "Precipitation",
        "MXN_CAD": "MXNCAD Exchange Rate",
        "USD_CAD": "USDCAD Exchange Rate",
        "integrated_gas_price": "Integrated Gas Price",
        "import_qty": "Import Quantity",
        "price_adjusted": "Avocado Price",
    })


def build_tomato_gold_features(
    tomato_import_df,
    tomato_weather_df,
    gas_df,
    fx_df,
    tomato_price_df,
):
    """
    Build the final tomato Gold feature table.
    """

    tomato_import_df = (
        tomato_import_df.groupby("date", as_index=False)["qty"]
        .sum()
        .rename(columns={"qty": "import_qty"})
    )
    tomato_target = tomato_price_df[["date", "price_adjusted"]].copy()

    tomato_features = (
        tomato_import_df
        .merge(tomato_weather_df, on="date", how="inner")
        .merge(gas_df, on="date", how="inner")
        .merge(fx_df, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )

    tomato_lagged = apply_lag_table(
        tomato_features.set_index("date"),
        TOMATO_LAGS,
    )
    tomato_final = tomato_lagged.loc[tomato_lagged.index >= START_DATE].dropna()
    tomato_final = tomato_final.join(
        tomato_target.set_index("date"),
        how="inner",
    ).reset_index()

    return tomato_final.rename(columns={
        "date": "Date",
        "MEAN_C": "Mean Temperature",
        "PRECIPITATION_MM": "Precipitation",
        "MXN_CAD": "MXNCAD Exchange Rate",
        "USD_CAD": "USDCAD Exchange Rate",
        "integrated_gas_price": "Integrated Gas Price",
        "import_qty": "Import Quantity",
        "price_adjusted": "Tomato Price",
    })


def transform_to_gold():
    """
    Rebuild the final Gold feature tables using the latest monthly Silver data.
    """

    s3 = boto3.client("s3")

    required_keys = [
        f"silver/grocery_price_adjusted/avocado/year={CURRENT_YEAR}/data.parquet",
        f"silver/grocery_price_adjusted/tomato/year={CURRENT_YEAR}/data.parquet",
        f"silver/canadian_agricultural_import/avocado/year={CURRENT_YEAR}/data.parquet",
        f"silver/canadian_agricultural_import/tomato/year={CURRENT_YEAR}/data.parquet",
        f"silver/mexico_weather/year={CURRENT_YEAR}/data.parquet",
        f"silver/oil_prices/integrated/year={CURRENT_YEAR}/data.parquet",
        f"silver/exchange_rate/year={CURRENT_YEAR}/data.parquet",
    ]

    missing_keys = [key for key in required_keys if not s3_key_exists(s3, key)]
    if missing_keys:
        raise AirflowSkipException(
            f"Required current-year Silver datasets are not ready yet: {missing_keys}"
        )

    weather_df = read_partitioned_dataset(s3, "silver/mexico_weather/")
    fx_df = read_partitioned_dataset(s3, "silver/exchange_rate/")
    gas_df = read_partitioned_dataset(s3, "silver/oil_prices/integrated/")
    avocado_price_df = read_partitioned_dataset(
        s3,
        "silver/grocery_price_adjusted/avocado/",
    )
    tomato_price_df = read_partitioned_dataset(
        s3,
        "silver/grocery_price_adjusted/tomato/",
    )
    avocado_import_df = read_partitioned_dataset(
        s3,
        "silver/canadian_agricultural_import/avocado/",
    )
    tomato_import_df = read_partitioned_dataset(
        s3,
        "silver/canadian_agricultural_import/tomato/",
    )

    avocado_weather_df, tomato_weather_df = build_weather_features(weather_df)

    fx_df = fx_df[["date", "MXN_CAD", "USD_CAD"]].copy()
    gas_df = gas_df[["date", "integrated_gas_price"]].copy()

    avocado_final = build_avocado_gold_features(
        avocado_import_df=avocado_import_df,
        avocado_weather_df=avocado_weather_df,
        gas_df=gas_df,
        fx_df=fx_df,
        avocado_price_df=avocado_price_df,
    )
    tomato_final = build_tomato_gold_features(
        tomato_import_df=tomato_import_df,
        tomato_weather_df=tomato_weather_df,
        gas_df=gas_df,
        fx_df=fx_df,
        tomato_price_df=tomato_price_df,
    )

    latest_gold_date = max(
        avocado_final["Date"].iloc[-1],
        tomato_final["Date"].iloc[-1],
    )

    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key="gold/avocado_features.parquet")
        existing_avocado = pd.read_parquet(BytesIO(obj["Body"].read()))
        existing_latest_date = (
            existing_avocado["Date"].iloc[-1] if not existing_avocado.empty else None
        )

        if existing_latest_date == latest_gold_date:
            print("Gold features already up to date")
            raise AirflowSkipException("Gold features already up to date")

        print(
            f"New month detected. Existing latest month: "
            f"{existing_latest_date}, new latest month: {latest_gold_date}"
        )
    except s3.exceptions.NoSuchKey:
        print("No Gold feature snapshot found. Saving Gold...")
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in {"404", "NoSuchKey"}:
            print("No Gold feature snapshot found. Saving Gold...")
        else:
            raise

    avocado_buffer = BytesIO()
    avocado_final.to_parquet(avocado_buffer, index=False)
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key="gold/avocado_features.parquet",
        Body=avocado_buffer.getvalue(),
        ContentType="application/octet-stream",
    )
    print(f"Uploaded Gold: s3://{BUCKET_NAME}/gold/avocado_features.parquet")

    tomato_buffer = BytesIO()
    tomato_final.to_parquet(tomato_buffer, index=False)
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key="gold/tomato_features.parquet",
        Body=tomato_buffer.getvalue(),
        ContentType="application/octet-stream",
    )
    print(f"Uploaded Gold: s3://{BUCKET_NAME}/gold/tomato_features.parquet")


# Default DAG configuration
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}


# Airflow DAG definition
with DAG(
    dag_id="gold_features_monthly",
    default_args=default_args,
    schedule=None,
    catchup=False,
) as dag:

    # Task 1: Rebuild final avocado/tomato Gold feature tables
    transform_task = PythonOperator(
        task_id="transform_to_gold",
        python_callable=transform_to_gold,
    )
