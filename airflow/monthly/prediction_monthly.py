from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime
from io import BytesIO
import boto3
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX


warnings.filterwarnings("ignore")

BUCKET_NAME = os.environ["BUCKET_NAME"]

PRODUCT_CONFIG = {
    "avocado": {
        "history_key": "gold/avocado_features.csv",
        "future_key": "gold/avocado_future_features.csv",
        "prediction_key": "prediction/avocado_predictions.csv",
        "drop_columns": ["MXN_CAD_lag_7"],
        "log_target": False,
        "order": (1, 1, 1),
        "seasonal_order": (1, 1, 1, 12),
        "enforce_stationarity": True,
        "enforce_invertibility": True,
    },
    "tomato": {
        "history_key": "gold/tomato_features.csv",
        "future_key": "gold/tomato_future_features.csv",
        "prediction_key": "prediction/tomato_predictions.csv",
        "drop_columns": [],
        "log_target": True,
        "order": (1, 0, 1),
        "seasonal_order": (0, 1, 1, 12),
        "enforce_stationarity": False,
        "enforce_invertibility": True,
    },
}


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


def read_csv_from_s3(s3, key):
    """
    Read a CSV file from S3 into a DataFrame.
    """

    print(f"Reading: s3://{BUCKET_NAME}/{key}")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return pd.read_csv(BytesIO(obj["Body"].read()))


def normalize_feature_dates(df):
    """
    Normalize Date values from either YYYY-MM or YYYY-MM-01 to month-start index.
    """

    result = df.copy()
    result["Date"] = pd.to_datetime(result["Date"])
    result = result.set_index("Date").sort_index()
    result.index = result.index.to_period("M").to_timestamp()
    return result.asfreq("MS")


def prepare_history_and_future(history_df, future_df, config):
    """
    Align historical and future feature matrices for SARIMAX training.
    """

    history_df = normalize_feature_dates(history_df)
    future_df = normalize_feature_dates(future_df)

    target_col = "price_adjusted"
    y_train = history_df[target_col].copy()
    X_train = history_df.drop(columns=[target_col]).copy()

    if config["drop_columns"]:
        X_train = X_train.drop(
            columns=[col for col in config["drop_columns"] if col in X_train.columns]
        )

    missing_future_columns = [col for col in X_train.columns if col not in future_df.columns]
    if missing_future_columns:
        raise ValueError(f"Future feature file missing columns: {missing_future_columns}")

    X_future = future_df[X_train.columns].copy()

    if config["log_target"]:
        y_train = np.log(y_train)

    scaler = StandardScaler()
    X_train.loc[:, X_train.columns] = scaler.fit_transform(X_train)
    X_future.loc[:, X_future.columns] = scaler.transform(X_future)

    return y_train, X_train, X_future


def build_prediction_df(predicted_mean, conf_int, log_target):
    """
    Format forecast values and confidence intervals for export.
    """

    if log_target:
        predicted_mean = np.exp(predicted_mean)
        conf_int = np.exp(conf_int)

    result = pd.DataFrame(
        {
            "Predicted_Price": predicted_mean.values,
            "Lower_CI": conf_int.iloc[:, 0].values,
            "Upper_CI": conf_int.iloc[:, 1].values,
        },
        index=predicted_mean.index,
    )
    result.index.name = "Date"
    result.index = result.index.strftime("%Y-%m")
    return result


def transform_to_prediction():
    """
    Train monthly SARIMAX models using Gold feature files and save prediction
    CSVs to the prediction/ S3 prefix.
    """

    s3 = boto3.client("s3")

    required_keys = []
    for config in PRODUCT_CONFIG.values():
        required_keys.extend([config["history_key"], config["future_key"]])

    missing_keys = [key for key in required_keys if not s3_key_exists(s3, key)]
    if missing_keys:
        raise AirflowSkipException(f"Required Gold files are not ready yet: {missing_keys}")

    for commodity, config in PRODUCT_CONFIG.items():
        history_df = read_csv_from_s3(s3, config["history_key"])
        future_df = read_csv_from_s3(s3, config["future_key"])

        if future_df.empty:
            raise AirflowSkipException(f"No future feature rows available for {commodity}")

        y_train, X_train, X_future = prepare_history_and_future(history_df, future_df, config)

        model = SARIMAX(
            endog=y_train,
            exog=X_train,
            order=config["order"],
            seasonal_order=config["seasonal_order"],
            enforce_stationarity=config["enforce_stationarity"],
            enforce_invertibility=config["enforce_invertibility"],
        )
        results = model.fit(disp=False)

        forecast_obj = results.get_forecast(steps=len(X_future), exog=X_future)
        prediction_df = build_prediction_df(
            forecast_obj.predicted_mean,
            forecast_obj.conf_int(alpha=0.50),
            config["log_target"],
        )

        buffer = BytesIO()
        prediction_df.to_csv(buffer)
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=config["prediction_key"],
            Body=buffer.getvalue(),
            ContentType="text/csv",
        )
        print(f"Uploaded prediction: s3://{BUCKET_NAME}/{config['prediction_key']}")


default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}


with DAG(
    dag_id="prediction_monthly",
    default_args=default_args,
    schedule=None,
    catchup=False,
) as dag:

    transform_task = PythonOperator(
        task_id="transform_to_prediction",
        python_callable=transform_to_prediction,
    )

    trigger_app_output_task = TriggerDagRunOperator(
        task_id="trigger_app_output_monthly",
        trigger_dag_id="app_output_monthly",
        wait_for_completion=False,
    )

    transform_task >> trigger_app_output_task
