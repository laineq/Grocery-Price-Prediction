'''
Computes correlations between variables across all lag values up to a maximum lag of 12 months.
input: Cleaned datasets from the AdjustedData folder:
    avocado_price_adjusted.csv,
    tomato_price_adjusted.csv,
    avocado_import.csv,
    tomato_import.csv,
    mexico_weather_adjusted.csv,
    gas_price.csv,
    xrate_adjusted.csv
output:
    avocado_lag_results_manual.csv (selected lag features of avocado)
    tomato_lag_results_manual.csv (selected lag featutures of tomato)
    avocado_correlations_lag12.csv (full lag correlation results of avocado)
    tomato_correlations_lag12.csv (full lag correlation results of tomato)
'''

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
    # df_transformed['price_adjusted'] = np.log(df_transformed['price_adjusted'])
    # df_transformed['integrated_gas_price'] = np.log(df_transformed['integrated_gas_price'])
    # df_transformed['import_qty'] = np.log(df_transformed['import_qty'])
    # import_qty and PRECIPITATION_MM remain completely raw
    return df_transformed

tom_data_transformed = apply_selective_log(tom_data)
avo_data_transformed = apply_selective_log(avo_data)

# ==========================================
# 3. MODIFIED: HYBRID CCF CALCULATION (MAX_LAG=12)
# ==========================================
def calculate_feature_correlations(df, target_col='price_adjusted', max_lag=12):

    results = []
    # Drop NaNs to ensure statsmodels ccf and adfuller don't crash
    df_clean = df.dropna().copy()
    
    # 1. Prepare Target (Check Stationarity)
    target_series = df_clean[target_col]
    _, p_val, *_ = adfuller(target_series)
    if p_val > 0.05:
        target_series = target_series.diff().dropna()
        target_note = "Differenced"
    else:
        target_note = "Level"

    # Define Feature Groups
    ccf_tool_cols = [ 'MXN_CAD', 'USD_CAD']
    manual_cols = ['integrated_gas_price' ,'PRECIPITATION_MM','import_qty','MEAN_C']

    # --- A. USE STATSMODELS CCF FUNCTION ---
    for col in ccf_tool_cols:
        if col not in df_clean.columns: continue
        
        exog_series = df_clean[col]
        # Pre-process exog stationarity
        _, f_p_val, *_ = adfuller(exog_series)
        if f_p_val > 0.05:
            exog_series = exog_series.diff().dropna()

        # Align lengths after differencing
        common_idx = target_series.index.intersection(exog_series.index)
        y = target_series.loc[common_idx]
        x = exog_series.loc[common_idx]

        # ccf(x, y) returns correlation of x_lagged with y
        ccf_values = ccf(x, y, adjusted=False)[:max_lag + 1]
        
        for lag, val in enumerate(ccf_values):
            results.append({
                'Feature': col,
                'Lag': lag,
                'Correlation': round(val, 4),
                'Method': 'statsmodels_ccf'
            })

    # --- B. USE MANUAL PANDAS SHIFT ---
    for col in manual_cols:
        if col not in df_clean.columns: continue
        
        exog_series = df_clean[col]
        _, f_p_val, *_ = adfuller(exog_series)
        if f_p_val > 0.05:
            exog_series = exog_series.diff().dropna()

        for lag in range(max_lag + 1):
            shifted_feat = exog_series.shift(lag)
            common_idx = target_series.index.intersection(shifted_feat.index)
            
            # Calculate correlation only on overlapping non-NaN rows
            if len(common_idx) > 0:
                corr = target_series.loc[common_idx].corr(shifted_feat.loc[common_idx])
            else:
                corr = np.nan
            
            results.append({
                'Feature': col,
                'Lag': lag,
                'Correlation': round(corr, 4) if not np.isnan(corr) else 0,
                'Method': 'manual_shift'
            })

    return pd.DataFrame(results)

# ==========================================
# 4. EXECUTION
# ==========================================

# Calculate for Tomato and Avocado with 12-month window
tom_all_lags = calculate_feature_correlations(tom_data_transformed, max_lag=12)
avo_all_lags = calculate_feature_correlations(avo_data_transformed, max_lag=12)

# Save the full 13-row (0 to 12) correlation matrices
tom_all_lags.to_csv('tomato_correlations_lag12.csv', index=False)
avo_all_lags.to_csv('avocado_correlations_lag12.csv', index=False)

# Display Pivot Tables for quick inspection
print("--- TOMATO CORRELATION MATRIX (Lags 0-12) ---")
print(tom_all_lags.pivot(index='Lag', columns='Feature', values='Correlation').to_markdown())

print("\n--- AVOCADO CORRELATION MATRIX (Lags 0-12) ---")
print(avo_all_lags.pivot(index='Lag', columns='Feature', values='Correlation').to_markdown())

import pandas as pd

# ==========================================
# 5. DEFINE THE CHOSEN REASONABLE LAGS
# ==========================================
avo_data = [
    {'Feature': 'MEAN_C', 'Optimal_Lag': 0, 'Correlation': 0.3628, 'Logic': 'Immediate harvest/heat stress'},
    {'Feature': 'PRECIPITATION_MM', 'Optimal_Lag': 5, 'Correlation': -0.3826, 'Logic': 'Mid-stage fruit bulking'},
    {'Feature': 'import_qty', 'Optimal_Lag': 0, 'Correlation': -0.1741, 'Logic': 'Immediate supply/demand'},
    {'Feature': 'USD_CAD', 'Optimal_Lag': 0, 'Correlation': 0.2084, 'Logic': 'Real-time import costs'},
    {'Feature': 'integrated_gas_price', 'Optimal_Lag': 1, 'Correlation': 0.1806, 'Logic': 'Logistics cost pass-through'},
    {'Feature': 'MXN_CAD', 'Optimal_Lag': 7, 'Correlation': 0.2002, 'Logic': 'Seasonal contract renewals'}
]
avo_lags_df = pd.DataFrame(avo_data)
avo_lags_df.to_csv('avocado_lag_results_manual.csv', index=False)

tom_data = [
    {'Feature': 'MEAN_C', 'Optimal_Lag': 2, 'Correlation': 0.3059, 'Logic': 'Flowering/fruit set stage'},
    {'Feature': 'PRECIPITATION_MM', 'Optimal_Lag': 3, 'Correlation': 0.4029, 'Logic': 'Planting/early growth moisture'},
    {'Feature': 'import_qty', 'Optimal_Lag': 3, 'Correlation': -0.3329, 'Logic': 'Supply chain pipeline lag'},
    {'Feature': 'USD_CAD', 'Optimal_Lag': 1, 'Correlation': -0.1643, 'Logic': 'Delayed currency adjustment'},
    {'Feature': 'integrated_gas_price', 'Optimal_Lag': 1, 'Correlation': 0.1534, 'Logic': 'Logistics cost pass-through'},
    {'Feature': 'MXN_CAD', 'Optimal_Lag': 1, 'Correlation': -0.1144, 'Logic': 'Short-term trade fluctuation'}
]
tom_lags_df = pd.DataFrame(tom_data)
tom_lags_df.to_csv('tomato_lag_results_manual.csv', index=False)

print("Files saved: 'avocado_lag_results_manual.csv' and 'tomato_lag_results_manual.csv'")