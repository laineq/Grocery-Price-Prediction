'''
Generates the final feature sets by applying selected lags and preparing both training and future datasets.
input: Same datasets as calculate_lag.py
output:
    avocado_final_selective_log.csv (lagged avocado features with target variable; last row corresponds to the final observed price)
    tomato_final_selective_log.csv (lagged tomato features with target variable; last row corresponds to the final observed price)
    avocado_future_features.csv (lagged avocado features for forecasting, covering periods after the last observed price to the prediction horizon)
    tomato_future_features.csv (lagged tomato features for forecasting, covering periods after the last observed price to the prediction horizon)
'''

import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. LOAD AND PREP DATA
# ==========================================
data_folder = Path("../AdjustedData")
output_folder = Path("../AdjustedData")
output_folder.mkdir(exist_ok=True)

# Load raw datasets
df_weather_raw = pd.read_csv(data_folder/'mexico_weather_adjusted.csv', parse_dates=['date'])
df_gas = pd.read_csv(data_folder/'gas_price.csv', parse_dates=['date'])[['date', 'integrated_gas_price']].set_index('date')
df_fx = pd.read_csv(data_folder/'xrate_adjusted.csv', parse_dates=['date'])[['date', 'MXN_CAD','USD_CAD']].set_index('date')

# --- Tomato Prep ---
df_weather_sinaloa = df_weather_raw[df_weather_raw['STATE'] == 'Sinaloa'].copy().set_index('date')
tom_weather_index = df_weather_sinaloa[['MEAN_C', 'PRECIPITATION_MM']]
df_tom_price = pd.read_csv(data_folder/'tomato_price_adjusted.csv')[['date', 'price_adjusted']]
df_tom_price['date'] = pd.to_datetime(df_tom_price['date'])
df_tom_price.set_index('date', inplace=True)
df_tom_imp = pd.read_csv(data_folder/'tomato_import.csv')
df_tom_imp = df_tom_imp.groupby('date')['qty'].sum().reset_index()
df_tom_imp['date'] = pd.to_datetime(df_tom_imp['date'])
df_tom_imp.set_index('date', inplace=True)
df_tom_imp.rename(columns={'qty': 'import_qty'}, inplace=True)

# --- Avocado Prep ---
df_mean_c = df_weather_raw.pivot(index='date', columns='STATE', values='MEAN_C')
df_precip = df_weather_raw.pivot(index='date', columns='STATE', values='PRECIPITATION_MM')
avo_weather_index = pd.DataFrame(index=df_mean_c.index)
avo_weather_index['MEAN_C'] = (0.7 * df_mean_c['Michoacán'] + 0.2 * df_mean_c['Jalisco'] + 0.1 * df_mean_c['Estado de México'])
avo_weather_index['PRECIPITATION_MM'] = (0.7 * df_precip['Michoacán'] + 0.2 * df_precip['Jalisco'] + 0.1 * df_precip['Estado de México'])
df_avo_price = pd.read_csv(data_folder/'avocado_price_adjusted.csv')[['date', 'price_adjusted']]
df_avo_price['date'] = pd.to_datetime(df_avo_price['date'])
df_avo_price.set_index('date', inplace=True)
df_avo_imp = pd.read_csv(data_folder/'avocado_import.csv')
df_avo_imp['date'] = pd.to_datetime(df_avo_imp['date'])
df_avo_imp.set_index('date', inplace=True)
df_avo_imp.rename(columns={'qty': 'import_qty'}, inplace=True)


# ==========================================
# 2. DEFINE HELPER FUNCTIONS (MUST BE FIRST)
# ==========================================

def apply_best_lags(df, lags_df):
    # Creates lagged feature columns based on a results dataframe.
    new_lagged_cols = {}
    for row in lags_df.itertuples():
        col = row.Feature
        lag = int(getattr(row, 'Optimal_Lag', 0))
        if col in df.columns:
            new_col_name = f"{col}_lag_{lag}"
            new_lagged_cols[new_col_name] = df[col].shift(lag)
    return pd.DataFrame(new_lagged_cols, index=df.index)

def prepare_final_features(feature_df, lags_df, target_start='2017-01-01'):
    # Orchestrates logging, lagging, and trimming of features.
    feature_df_logged = feature_df
    # Generate the lagged columns
    lagged_df = apply_best_lags(feature_df_logged, lags_df)
    # Trim to history and drop NaN rows created by lags
    final_features = lagged_df.loc[lagged_df.index >= pd.to_datetime(target_start)]
    return final_features.dropna()

# ==========================================
# 3. LOAD MANUAL LAGS & SETUP TIMELINE
# ==========================================

try:
    tom_results = pd.read_csv('tomato_lag_results_manual.csv')
    avo_results = pd.read_csv('avocado_lag_results_manual.csv')
except FileNotFoundError:
    print("Error: Manual lag CSV files not found. Run selection script first.")
    exit()

# Separate target from features
tom_features_only = df_tom_imp.join([tom_weather_index, df_gas, df_fx], how='outer')
avo_features_only = df_avo_imp.join([avo_weather_index, df_gas, df_fx], how='outer')

tom_target_only = df_tom_price
avo_target_only = df_avo_price

# Setup future prediction window
today = pd.Timestamp.now()
target_prediction_date = today.replace(day=1) + pd.DateOffset(months=1)
future_timeline = pd.date_range(start=tom_features_only.index.min(), end=target_prediction_date, freq='MS')

print(f"Today is: {today.strftime('%Y-%m')}")
print(f"Stretching dataset to predict: {target_prediction_date.strftime('%Y-%m')}")

# ==========================================
# 4. DATA FILLING AND EXECUTION
# ==========================================

tom_features_only = tom_features_only.reindex(future_timeline)
avo_features_only = avo_features_only.reindex(future_timeline)

# A. Seasonal Fill
for col in ['MEAN_C', 'PRECIPITATION_MM', 'import_qty']:
    for _ in range(2):
        tom_features_only[col] = tom_features_only[col].fillna(tom_features_only[col].shift(12))
        avo_features_only[col] = avo_features_only[col].fillna(avo_features_only[col].shift(12))

# B. Forward Fill
tom_features_only = tom_features_only.ffill()
avo_features_only = avo_features_only.ffill()

# Generate Final Feature Sets
tom_lagged_features = prepare_final_features(tom_features_only, tom_results, target_start='2017-01')
avo_lagged_features = prepare_final_features(avo_features_only, avo_results, target_start='2017-01')

# Merge and Export Training Logs
tom_final = tom_lagged_features.join(tom_target_only, how='inner')
avo_final = avo_lagged_features.join(avo_target_only, how='inner')

tom_final.to_csv('tomato_final_selective_log.csv', index=True, index_label='Date')
avo_final.to_csv('avocado_final_selective_log.csv', index=True, index_label='Date')

# Export Future Features
tom_lagged_features.index = pd.to_datetime(tom_lagged_features.index)
avo_lagged_features.index = pd.to_datetime(avo_lagged_features.index)

tom_future = tom_lagged_features.loc[tom_lagged_features.index > df_tom_price.index.max()]
avo_future = avo_lagged_features.loc[avo_lagged_features.index > df_avo_price.index.max()]

tom_future.to_csv('tomato_future_features.csv', index=True, index_label='Date')
avo_future.to_csv('avocado_future_features.csv', index=True, index_label='Date')

print(f"Tomato Future Rows exported: {len(tom_future)}")
print(f"Avocado Future Rows exported: {len(avo_future)}")