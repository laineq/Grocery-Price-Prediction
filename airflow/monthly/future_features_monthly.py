from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime
from io import BytesIO
import boto3
import os
import pandas as pd


BUCKET_NAME = os.environ["BUCKET_NAME"]
START_DATE = "2017-01"

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


def read_partitioned_dataset(s3, prefix):
    """
    Read all parquet files under a partitioned S3 prefix into one DataFrame.
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


def read_gold_csv(s3, key):
    """
    Read a Gold CSV file from S3.
    """

    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return pd.read_csv(BytesIO(obj["Body"].read()))


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
        lagged_column = f"{column}_lag_{lag}"
        result[lagged_column] = df[column].shift(lag) if lag > 0 else df[column]

    return result


def cast_feature_columns_to_float(df):
    """
    Cast all numeric feature columns to float for a consistent schema.
    """

    result = df.copy()
    numeric_columns = [column for column in result.columns if column != "Date"]
    result[numeric_columns] = result[numeric_columns].astype(float)
    return result


def get_target_prediction_date():
    """
    Return the first day of next month as YYYY-MM.
    """

    now = pd.Timestamp.utcnow()
    next_month = now.replace(day=1) + pd.DateOffset(months=1)
    return next_month.strftime("%Y-%m")


def extend_and_fill_features(feature_df, target_prediction_date):
    """
    Extend the raw feature table through next month and fill missing values
    using seasonal fill first and forward fill second.
    """

    result = feature_df.copy()
    result.index = pd.to_datetime(result.index)

    future_timeline = pd.date_range(
        start=result.index.min(),
        end=pd.to_datetime(f"{target_prediction_date}-01"),
        freq="MS",
    )
    result = result.reindex(future_timeline)

    for column in ["MEAN_C", "PRECIPITATION_MM", "import_qty"]:
        for _ in range(2):
            result[column] = result[column].fillna(result[column].shift(12))

    return result.ffill()


def build_future_features(base_features_df, lag_map, last_feature_date):
    """
    Apply fixed lags and keep only rows after the last observed Gold feature month.
    """

    lagged_df = apply_lag_table(base_features_df, lag_map)
    lagged_df = lagged_df.loc[lagged_df.index >= pd.to_datetime(START_DATE)].dropna()
    future_df = lagged_df.loc[lagged_df.index > last_feature_date].copy()
    future_df = future_df.reset_index().rename(columns={"index": "Date"})
    future_df["Date"] = future_df["Date"].dt.strftime("%Y-%m")
    return cast_feature_columns_to_float(future_df)


def build_base_feature_table(import_df, weather_df, gas_df, fx_df):
    """
    Build the raw monthly feature table before lagging.
    """

    result = (
        import_df
        .merge(weather_df, on="date", how="inner")
        .merge(gas_df, on="date", how="inner")
        .merge(fx_df, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )
    result["date"] = pd.to_datetime(result["date"], format="%Y-%m")
    return result.set_index("date")


def transform_to_future_features():
    """
    Build future feature files for avocado and tomato and save them to Gold.
    """

    s3 = boto3.client("s3")

    required_keys = [
        "gold/avocado_features.csv",
        "gold/tomato_features.csv",
    ]
    missing_keys = [key for key in required_keys if not s3_key_exists(s3, key)]
    if missing_keys:
        raise AirflowSkipException(
            f"Required Gold feature files are not ready yet: {missing_keys}"
        )

    weather_df = read_partitioned_dataset(s3, "silver/mexico_weather/")
    fx_df = read_partitioned_dataset(s3, "silver/exchange_rate/")
    gas_df = read_partitioned_dataset(s3, "silver/oil_prices/integrated/")
    avocado_import_df = read_partitioned_dataset(
        s3,
        "silver/canadian_agricultural_import/avocado/",
    )
    tomato_import_df = read_partitioned_dataset(
        s3,
        "silver/canadian_agricultural_import/tomato/",
    )

    avocado_gold_df = read_gold_csv(s3, "gold/avocado_features.csv")
    tomato_gold_df = read_gold_csv(s3, "gold/tomato_features.csv")

    avocado_last_date = pd.to_datetime(avocado_gold_df["Date"].iloc[-1], format="%Y-%m")
    tomato_last_date = pd.to_datetime(tomato_gold_df["Date"].iloc[-1], format="%Y-%m")
    target_prediction_date = get_target_prediction_date()

    avocado_weather_df, tomato_weather_df = build_weather_features(weather_df)
    fx_df = fx_df[["date", "MXN_CAD", "USD_CAD"]].copy()
    gas_df = gas_df[["date", "integrated_gas_price"]].copy()

    avocado_import_df = avocado_import_df.rename(columns={"qty": "import_qty"})
    avocado_import_df = avocado_import_df[["date", "import_qty"]].copy()
    tomato_import_df = (
        tomato_import_df.groupby("date", as_index=False)["qty"]
        .sum()
        .rename(columns={"qty": "import_qty"})
    )

    avocado_features = build_base_feature_table(
        avocado_import_df,
        avocado_weather_df,
        gas_df,
        fx_df,
    )
    tomato_features = build_base_feature_table(
        tomato_import_df,
        tomato_weather_df,
        gas_df,
        fx_df,
    )

    avocado_features = extend_and_fill_features(avocado_features, target_prediction_date)
    tomato_features = extend_and_fill_features(tomato_features, target_prediction_date)

    avocado_future = build_future_features(
        avocado_features,
        AVOCADO_LAGS,
        avocado_last_date,
    )
    tomato_future = build_future_features(
        tomato_features,
        TOMATO_LAGS,
        tomato_last_date,
    )

    expected_avocado_start = (avocado_last_date + pd.DateOffset(months=1)).strftime("%Y-%m")
    expected_tomato_start = (tomato_last_date + pd.DateOffset(months=1)).strftime("%Y-%m")

    if not avocado_future.empty and avocado_future["Date"].iloc[0] != expected_avocado_start:
        raise ValueError(
            f"Avocado future features should start at {expected_avocado_start}, "
            f"but got {avocado_future['Date'].iloc[0]}"
        )
    if not tomato_future.empty and tomato_future["Date"].iloc[0] != expected_tomato_start:
        raise ValueError(
            f"Tomato future features should start at {expected_tomato_start}, "
            f"but got {tomato_future['Date'].iloc[0]}"
        )

    latest_future_month = target_prediction_date
    try:
        avocado_obj = s3.get_object(Bucket=BUCKET_NAME, Key="gold/avocado_future_features.csv")
        tomato_obj = s3.get_object(Bucket=BUCKET_NAME, Key="gold/tomato_future_features.csv")
        existing_avocado = pd.read_csv(BytesIO(avocado_obj["Body"].read()))
        existing_tomato = pd.read_csv(BytesIO(tomato_obj["Body"].read()))

        avocado_first = existing_avocado["Date"].iloc[0] if not existing_avocado.empty else None
        avocado_last = existing_avocado["Date"].iloc[-1] if not existing_avocado.empty else None
        tomato_first = existing_tomato["Date"].iloc[0] if not existing_tomato.empty else None
        tomato_last = existing_tomato["Date"].iloc[-1] if not existing_tomato.empty else None

        if (
            avocado_first == expected_avocado_start and
            avocado_last == latest_future_month and
            tomato_first == expected_tomato_start and
            tomato_last == latest_future_month
        ):
            print("Future feature files already up to date")
            raise AirflowSkipException("Future feature files already up to date")
    except s3.exceptions.NoSuchKey:
        print("No future feature snapshot found. Saving future features...")
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in {"404", "NoSuchKey"}:
            print("No future feature snapshot found. Saving future features...")
        else:
            raise

    avocado_buffer = BytesIO()
    avocado_future.to_csv(avocado_buffer, index=False)
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key="gold/avocado_future_features.csv",
        Body=avocado_buffer.getvalue(),
        ContentType="text/csv",
    )
    print(f"Uploaded Gold: s3://{BUCKET_NAME}/gold/avocado_future_features.csv")

    tomato_buffer = BytesIO()
    tomato_future.to_csv(tomato_buffer, index=False)
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key="gold/tomato_future_features.csv",
        Body=tomato_buffer.getvalue(),
        ContentType="text/csv",
    )
    print(f"Uploaded Gold: s3://{BUCKET_NAME}/gold/tomato_future_features.csv")


default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}


with DAG(
    dag_id="future_features_monthly",
    default_args=default_args,
    schedule="0 0 1 * *",
    catchup=False,
) as dag:

    transform_task = PythonOperator(
        task_id="transform_to_future_features",
        python_callable=transform_to_future_features,
    )

    trigger_prediction_task = TriggerDagRunOperator(
        task_id="trigger_prediction_monthly",
        trigger_dag_id="prediction_monthly",
        wait_for_completion=False,
    )

    transform_task >> trigger_prediction_task
