from pathlib import Path

import numpy as np
import pandas as pd


DATE_COLUMN = "date"
TARGET_COLUMN = "price_adjusted"
LOG_COLUMN = "log_price"


def load_price_data(data_dir: Path, file_name: str, include_log: bool = False) -> tuple[pd.DataFrame, dict]:
    file_path = data_dir / file_name
    df = pd.read_csv(file_path)

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format="%Y-%m", errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    if include_log:
        df[LOG_COLUMN] = np.where(df[TARGET_COLUMN] > 0, np.log(df[TARGET_COLUMN]), np.nan)

    quality_report = {
        "rows": int(len(df)),
        "missing_date": int(df[DATE_COLUMN].isna().sum()),
        "missing_price_adjusted": int(df[TARGET_COLUMN].isna().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_dates": int(df.duplicated(subset=[DATE_COLUMN]).sum()),
    }

    return df, quality_report
