'''
Predicts the target price using the SARIMAX model
input: Outputs generated from feature_lag.py in the Feature-Engineering module
    avocado_final_selective_log.csv
    tomato_final_selective_log.csv
    avocado_future_features.csv
    tomato_future_features.csv
output: Predicted prices along with confidence intervals
    avocado_sarima_predictions.csv
    tomato_sarima_predictions.csv
'''

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path
from metrics import compute_metrics
from cv import create_expanding_window_folds
import matplotlib.pyplot as plt  
import warnings
from sklearn.preprocessing import StandardScaler
import numpy as np

# Ignore statsmodels warnings for a cleaner terminal
warnings.filterwarnings("ignore")

# Setup Folders
data_folder = Path("../Feature-Engineering")
output_folder = Path("../sarimax-model-output")
output_folder.mkdir(exist_ok=True)

commodities = ['tomato', 'avocado']

for commodity in commodities:
    print(f"\n{'='*40}")
    print(f"PREDICTING FUTURE FOR: {commodity.upper()}")
    print(f"{'='*40}")
    
    # 1. Load Historical Data (Train on EVERYTHING)
    hist_path = data_folder / f"{commodity}_final_selective_log.csv"
    df_hist = pd.read_csv(hist_path, parse_dates=['Date'], index_col='Date')
    df_hist = df_hist.asfreq('MS').interpolate(method='linear')
    
    # 2. Load Future Features
    future_path = data_folder / f"{commodity}_future_features.csv"
    df_future = pd.read_csv(future_path, parse_dates=['Date'], index_col='Date')
    df_future = df_future.asfreq('MS') # Ensure dates are recognized

    # 3. Using log 
    if commodity in ['tomato']:
        if 'price_adjusted' in df_hist.columns:
            df_hist['price_adjusted'] = np.log(df_hist['price_adjusted'])
        if 'price_adjusted' in df_future.columns:
            df_future['price_adjusted'] = np.log(df_future['price_adjusted'])

    # 4. Define Target and Features
    target_col = 'price_adjusted'
    y_train = df_hist[target_col]
    X_train = df_hist.drop(columns=[target_col])

    # 5. Addressing Multicollinearity for Avocado 
    if commodity == 'avocado':
        # We drop MXN_CAD because USD_CAD is the primary driver for imports 
        # and having both can confuse the model.
        cols_to_drop = ['MXN_CAD_lag_7'] # Use the exact column name in your lagged dataset
        X_train = X_train.drop(columns=[c for c in cols_to_drop if c in X_train.columns])
        
    # 4. Train the Model on ALL historical data   
    scaler = StandardScaler()
    X_future = df_future[X_train.columns] # Ensure column order matches exactly
    X_train[X_train.columns] = scaler.fit_transform(X_train)
    X_future[X_future.columns] = scaler.transform(X_future)
    if commodity == 'avocado':
        model = SARIMAX(
            endog=y_train, 
            exog=X_train,
            order=(1,1,1),
            seasonal_order=(1,1,1,12),
            enforce_stationarity=True,
            enforce_invertibility=True
        )
    if commodity == 'tomato':
        model = SARIMAX(
            endog=y_train, 
            exog=X_train,
            order=(1, 1, 1),              
            seasonal_order=(1, 1, 1, 12), 
            enforce_stationarity=True,
            enforce_invertibility=True
        )
    results = model.fit(disp=False)
        
    # 5. Make the True Future Prediction
    forecast_obj = results.get_forecast(steps=len(X_future), exog=X_future)
    future_predictions = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int(alpha=0.50)
      
    # 6. EXPORT PREDICTIONS TO CSV (Convert back from log scale if needed)
    if commodity in ['tomato']:
        # Reverse log transform
        future_predictions = np.exp(future_predictions)
        conf_int = np.exp(conf_int)

    print(f"\n{commodity.upper()} FUTURE PRICE FORECAST:")
    for date, pred, lower, upper in zip(future_predictions.index, future_predictions.values, conf_int.iloc[:, 0], conf_int.iloc[:, 1]):
        date_str = date.strftime('%Y-%m')
        print(f"   {date_str} | Estimated Price: ${pred:.2f} (Expected Range: ${lower:.2f} to ${upper:.2f})")

    # 7. EXPORT PREDICTIONS TO CSV
    final_output = pd.DataFrame({
        'Predicted_Price': future_predictions.values,
        'Lower_CI': conf_int.iloc[:, 0].values,
        'Upper_CI': conf_int.iloc[:, 1].values
    }, index=future_predictions.index)
    
    # Format the date index to look like 'YYYY-MM'
    final_output.index.name = 'Date'
    final_output.index = final_output.index.strftime('%Y-%m')
    
    # Save the file to your Feature-Engineering folder
    output_filename = output_folder / f"{commodity}_sarima_predictions.csv"
    final_output.to_csv(output_filename)
    print(f" [SUCCESS] Predictions saved to: {output_filename}")

   