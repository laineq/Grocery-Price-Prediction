from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
from io import BytesIO
import boto3
import os
import pandas as pd


# S3 bucket name (configured via docker-compose environment variable)
BUCKET_NAME = os.environ["BUCKET_NAME"]

# Fixed lag configuration from feature engineering results
AVOCADO_LAGS = {
    "MEAN_C": 0,
    "PRECIPITATION_MM": 5,
    "import_qty": 0,
    "USD_CAD": 0,
    "integrated_gas_price": 1,
    "MXN_CAD": 7,
}

TOMATO_LAGS = {
    "MEAN_C": 2,
    "PRECIPITATION_MM": 3,
    "import_qty": 3,
    "USD_CAD": 1,
    "integrated_gas_price": 1,
    "MXN_CAD": 1,
}

START_DATE = "2017-01"
END_DATE = "2025-12"


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
        lagged_column = f"{column}_lag_{lag}"
        result[lagged_column] = df[column].shift(lag) if lag > 0 else df[column]

    return result


def cast_feature_columns_to_float(df):
    """
    Cast all numeric feature/target columns to float for a consistent Gold schema.
    """

    result = df.copy()
    numeric_columns = [column for column in result.columns if column != "Date"]
    result[numeric_columns] = result[numeric_columns].astype(float)
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
    avocado_final = avocado_final[avocado_final["date"] <= END_DATE].copy()

    avocado_final = avocado_final.rename(columns={
        "date": "Date",
    })
    return cast_feature_columns_to_float(avocado_final)


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
    tomato_final = tomato_final[tomato_final["date"] <= END_DATE].copy()

    tomato_final = tomato_final.rename(columns={
        "date": "Date",
    })
    return cast_feature_columns_to_float(tomato_final)


def transform_to_gold():
    """
    Build final avocado and tomato feature tables and store them in Gold.
    """

    s3 = boto3.client("s3")

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

    print("Avocado Gold sample:")
    print(avocado_final.head())
    print("Tomato Gold sample:")
    print(tomato_final.head())

    avocado_buffer = BytesIO()
    avocado_final.to_csv(avocado_buffer, index=False)
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key="gold/avocado_features.csv",
        Body=avocado_buffer.getvalue(),
        ContentType="text/csv",
    )
    print(f"Uploaded Gold: s3://{BUCKET_NAME}/gold/avocado_features.csv")

    tomato_buffer = BytesIO()
    tomato_final.to_csv(tomato_buffer, index=False)
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key="gold/tomato_features.csv",
        Body=tomato_buffer.getvalue(),
        ContentType="text/csv",
    )
    print(f"Uploaded Gold: s3://{BUCKET_NAME}/gold/tomato_features.csv")


# Default DAG configuration
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}


# Airflow DAG definition (manual run only)
with DAG(
    dag_id="gold_features_full_history",
    default_args=default_args,
    schedule=None,
    catchup=False,
) as dag:

    # Task 1: Build final avocado/tomato feature tables → Gold
    transform_task = PythonOperator(
        task_id="transform_to_gold",
        python_callable=transform_to_gold,
    )
