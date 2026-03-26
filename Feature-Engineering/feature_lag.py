import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import ccf
from pathlib import Path
from statsmodels.tsa.stattools import adfuller

# ==========================================
# 1. LOAD AND PREP DATA
# ==========================================
data_folder = Path("../AdjustedData")
output_folder = Path("../AdjustedData")
output_folder.mkdir(exist_ok=True)


df_weather_raw = pd.read_csv(data_folder/'mexico_weather_adjusted.csv')
df_weather_raw['date'] = pd.to_datetime(df_weather_raw['date'])

# Gas Price
df_gas = pd.read_csv(data_folder/'gas_price.csv')[['date', 'integrated_gas_price']]
df_gas['date'] = pd.to_datetime(df_gas['date'])
df_gas.set_index('date', inplace=True)

# Exchange Rate
df_fx = pd.read_csv(data_folder/'xrate_adjusted.csv')[['date', 'MXN_CAD','USD_CAD']]
df_fx['date'] = pd.to_datetime(df_fx['date'])
df_fx.set_index('date', inplace=True)

# ------------------------------------------
# A. TOMATO DATA PREP (Jalisco Weather + Combined Imports)
# ------------------------------------------
df_weather_jalisco = df_weather_raw[df_weather_raw['STATE'] == 'Sinaloa'].copy()
df_weather_jalisco.set_index('date', inplace=True)
tom_weather_index = df_weather_jalisco[['MEAN_C', 'PRECIPITATION_MM']]

df_tom_price = pd.read_csv(data_folder/'tomato_price_adjusted.csv')[['date', 'price_adjusted']]
df_tom_price['date'] = pd.to_datetime(df_tom_price['date'])
df_tom_price.set_index('date', inplace=True)

df_tom_imp = pd.read_csv(data_folder/'tomato_import.csv')
df_tom_imp = df_tom_imp.groupby('date')['qty'].sum().reset_index()
df_tom_imp['date'] = pd.to_datetime(df_tom_imp['date'])
df_tom_imp.set_index('date', inplace=True)
df_tom_imp.rename(columns={'qty': 'import_qty'}, inplace=True)



tom_data = df_tom_price.join(df_tom_imp, how='inner').join(tom_weather_index, how='inner').join(df_gas, how='inner').join(df_fx, how = 'inner')

# ------------------------------------------
# B. AVOCADO DATA PREP (Weighted Weather + Regular Imports)
# ------------------------------------------
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

avo_data = df_avo_price.join(df_avo_imp, how='inner').join(avo_weather_index, how='inner').join(df_gas, how='inner').join(df_fx, how = 'inner')


# ==========================================
# 2. SELECTIVE LOG TRANSFORMATION
# ==========================================
def apply_selective_log(df):
    df_transformed = df.copy()
    # ONLY log the price and the gas price
    df_transformed['price_adjusted'] = np.log(df_transformed['price_adjusted'])
    df_transformed['integrated_gas_price'] = np.log(df_transformed['integrated_gas_price'])
    # import_qty and PRECIPITATION_MM remain completely raw
    return df_transformed

tom_data_transformed = apply_selective_log(tom_data)
avo_data_transformed = apply_selective_log(avo_data)


# ==========================================
# 3. MIXED METHODOLOGY LAG CALCULATION
# ==========================================
def calculate_mixed_lags(df, target_col='price_adjusted', max_lag=6):
    results = []
    df_clean = df.dropna()
    target_series = df_clean[target_col]
    result = adfuller(target_series.dropna())
    if result[1] > 0.05:  # p-value > 0.05 means non-stationary
        target_series = target_series.diff().dropna()

    # 1. Calculate the dynamic threshold (95% Confidence)
    n = len(target_series)
    threshold = 2 / np.sqrt(n)

    # A. CCF for agricultural features
    ccf_cols = ['MEAN_C','PRECIPITATION_MM','MXN_CAD','USD_CAD']
    for col in ccf_cols:
        exog_series = df_clean[col]

        method_note = 'ccf'
        if col in ['integrated_gas_price', 'import_qty']:
            result = adfuller(exog_series.dropna())
            if result[1] > 0.05:  # p-value > 0.05 means non-stationary
                exog_series = exog_series.diff().dropna()
                method_note = 'ccf (differenced)'
            else:
                method_note = 'ccf (levels)'


        ccf_vals = ccf(exog_series,target_series,adjusted=False)[:max_lag + 1]
        best_abs_lag = int(np.argmax(np.abs(ccf_vals)))
        best_corr = ccf_vals[best_abs_lag]

        if abs(best_corr) >= threshold: # ONLY keep if it breaks the threshold
            results.append({
                'Feature': col,
                'Method': method_note,
                'Optimal_Lag': best_abs_lag,
                'Correlation': round(best_corr, 3)
            })
        else:
            results.append({
                'Feature': col,
                'Method': method_note,
                'Optimal_Lag': 0,
                'Correlation': round(best_corr, 3)
            })
    
    # B. Manual Pandas Shift 
    manual_cols = ['integrated_gas_price','import_qty']
    for col in manual_cols:
        exog_series = df_clean[col]
        manual_corrs = []

        method_note = 'manual'

        if col in ['integrated_gas_price', 'import_qty']:
            result = adfuller(exog_series.dropna())
            if result[1] > 0.05:  # p-value > 0.05 means non-stationary
                exog_series = exog_series.diff().dropna()
                method_note = 'manual (differenced)'
            else:
                method_note = 'manual (levels)'

        for lag in range(max_lag + 1):
            if lag == 0:
                corr = df[target_col].corr(exog_series)
            else:
                corr = df[target_col].corr(exog_series.shift(lag))
            manual_corrs.append(corr)
        
        best_manual_lag = int(np.argmax(np.abs(manual_corrs)))
        best_manual_corr = manual_corrs[best_manual_lag]

        if abs(best_manual_corr) >= threshold: # ONLY keep if it breaks the threshold
            results.append({
                'Feature': col,
                'Method': method_note,
                'Optimal_Lag': best_manual_lag,
                'Correlation': round(best_manual_corr, 3)
            })
        else:
            results.append({
                'Feature': col,
                'Method': method_note,
                'Optimal_Lag': 0,
                'Correlation': round(best_manual_corr, 3)
            })

 
    return pd.DataFrame(results)



tom_results = calculate_mixed_lags(tom_data_transformed)
avo_results = calculate_mixed_lags(avo_data_transformed)

tom_results.to_csv('tomato_lag_results.csv', index=False)
avo_results.to_csv('avocado_lag_results.csv', index=False)

print("\n--- AVOCADO MODEL (No Log on Volume/Precip) ---")
print(avo_results.to_markdown(index=False))
print("--- TOMATO MODEL (No Log on Volume/Precip) ---")
print(tom_results.to_markdown(index=False))


# ============================
# 4. Function to apply lags
# ============================
def apply_best_lags(df, lags_df):
    """
    Creates lagged features efficiently using pd.concat.
    Original columns are preserved.
    Lagged columns align by date (same index as df).
    """
    new_lagged_cols = {}
    
    for row in lags_df.itertuples():
        col = row.Feature
        lag = getattr(row, 'Optimal_Lag', 0)
        
        if col in df.columns:
            new_col_name = f"{col}_lag_{lag}"
            if lag > 0:
                new_lagged_cols[new_col_name] = df[col].shift(lag)
            else:
                new_lagged_cols[new_col_name] = df[col]

    # Concatenate original + lagged features along columns
    df_lagged = pd.concat([df, pd.DataFrame(new_lagged_cols, index=df.index)], axis=1)
    
    # Drop rows with any NaNs created by shifting
    df_lagged = df_lagged
    return df_lagged

def prepare_final_features(feature_df, lags_df, target_start='2017-01-01'):
    
    max_lag = lags_df['Optimal_Lag'].max()
    
    # Apply lags on FULL feature history (from 2016-01)
    lagged_features = apply_best_lags(feature_df, lags_df)
    
    # NOW trim to target start
    final_features = lagged_features.loc[lagged_features.index >= pd.to_datetime(target_start)]
    
    # Drop original unlagged columns
    cols_to_drop = [row.Feature for row in lags_df.itertuples() 
                    if row.Feature in final_features.columns]
    final_features = final_features.drop(columns=cols_to_drop)

    return final_features.dropna()

# ============================
# 5. Execute and Save (The Bulletproof Method)
# ============================

# 1. Separate your target from your features (if they are currently in the same dataframe)
tom_features_only = df_tom_imp.join(tom_weather_index, how='outer')\
                              .join(df_gas, how='outer')\
                              .join(df_fx, how='outer')

avo_features_only = df_avo_imp.join(avo_weather_index, how='outer')\
                              .join(df_gas, how='outer')\
                              .join(df_fx, how='outer')


tom_target_only = df_tom_price
avo_target_only = df_avo_price

# 2. Get today's actual, real-world date
today = pd.Timestamp.now()
target_prediction_date = today.replace(day=1) + pd.DateOffset(months=1)

print(f"Today is: {today.strftime('%Y-%m')}")
print(f"Stretching dataset to predict: {target_prediction_date.strftime('%Y-%m')}")

# 3. Create a strict monthly calendar that forces the timeline to reach next month
future_timeline = pd.date_range(start=tom_features_only.index.min(), 
                                end=target_prediction_date, 
                                freq='MS')

# 4. Apply the new calendar and Fill the missing data
tom_features_only = tom_features_only.reindex(future_timeline)
avo_features_only = avo_features_only.reindex(future_timeline)

# A. Seasonal Fill (Same month, last year) for Weather and Imports
# (Update these strings to exactly match your dataframe column names!)
tom_seasonal_cols = ['MEAN_C', 'PRECIPITATION_MM', 'import_qty'] 
avo_seasonal_cols = ['MEAN_C', 'PRECIPITATION_MM', 'import_qty']

for col in avo_seasonal_cols:
    if col in avo_features_only.columns:
        # Apply shift(12) up to 2 times to cover any remaining gaps
        for _ in range(2):
            avo_features_only[col] = avo_features_only[col].fillna(
                avo_features_only[col].shift(12)
            )
for col in tom_seasonal_cols:
    if col in tom_features_only.columns:
        # Apply shift(12) up to 2 times to cover any remaining gaps
        for _ in range(2):
            tom_features_only[col] = tom_features_only[col].fillna(
                tom_features_only[col].shift(12)
            )

# B. Forward-fill ONLY remaining gaps in non-seasonal columns (gas, FX)
non_seasonal_cols = [c for c in tom_features_only.columns if c not in tom_seasonal_cols]
tom_features_only[non_seasonal_cols] = tom_features_only[non_seasonal_cols].ffill()

non_seasonal_cols = [c for c in avo_features_only.columns if c not in avo_seasonal_cols]
avo_features_only[non_seasonal_cols] = avo_features_only[non_seasonal_cols].ffill()

# 5. Apply the lag function ONLY to the features. 
tom_lagged_features = prepare_final_features(tom_features_only, tom_results, target_start='2017-01')
avo_lagged_features = prepare_final_features(avo_features_only, avo_results, target_start='2017-01')

# Merge the target variable back onto the safely lagged features
tom_final = tom_lagged_features.join(tom_target_only, how='inner')
avo_final = avo_lagged_features.join(avo_target_only, how='inner')

tom_final.index = tom_final.index.strftime('%Y-%m')
avo_final.index = avo_final.index.strftime('%Y-%m')

# Save to CSV without the date index
tom_final.to_csv('tomato_final_selective_log.csv', index=True, index_label='Date')
avo_final.to_csv('avocado_final_selective_log.csv', index=True, index_label='Date')

# ==========================================
# 6. EXPORT FUTURE FEATURES FOR PREDICTION
# ==========================================
# Format Datetime so the comparison works perfectly
tom_lagged_features.index = pd.to_datetime(tom_lagged_features.index)
tom_target_only.index = pd.to_datetime(tom_target_only.index)
avo_lagged_features.index = pd.to_datetime(avo_lagged_features.index)
avo_target_only.index = pd.to_datetime(avo_target_only.index)

tom_future = tom_lagged_features.loc[tom_lagged_features.index > tom_target_only.index.max()].copy()
avo_future = avo_lagged_features.loc[avo_lagged_features.index > avo_target_only.index.max()].copy()

tom_future.index = tom_future.index.strftime('%Y-%m')
avo_future.index = avo_future.index.strftime('%Y-%m')

# Save to CSV
tom_future.to_csv('tomato_future_features.csv', index=True, index_label='Date')
avo_future.to_csv('avocado_future_features.csv', index=True, index_label='Date')

print(f" Tomato Future Rows exported: {len(tom_future)}")
print(f" Avocado Future Rows exported: {len(avo_future)}")
