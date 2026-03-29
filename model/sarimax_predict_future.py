import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path
from metrics import compute_metrics
from cv import create_expanding_window_folds
import matplotlib.pyplot as plt  # <--- NEW: Imported for plotting
import warnings


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
    
    # 3. Define Target and Features
    target_col = 'price_adjusted'
    y_train = df_hist[target_col]
    X_train = df_hist.drop(columns=[target_col])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # The exogenous variables for the future
    X_future = df_future[X_train.columns] # Ensure column order matches exactly

    # This elegantly squashes all large/tiny numbers into a safe mathematical range
    X_train[X_train.columns] = scaler.fit_transform(X_train)
    X_future[X_future.columns] = scaler.transform(X_future)
   
    # 4. Train the Model on ALL historical data
    print("Training model on all historical data...")
    model = SARIMAX(
        endog=y_train, 
        exog=X_train,
        order=(1, 1, 1),              
        seasonal_order=(1, 1, 1, 12), 
        enforce_stationarity=True,
        enforce_invertibility=True
    )
    results = model.fit(disp=False)
    
    # ==========================================
    # IMPROVEMENT 2: CHECK P-VALUES
    # ==========================================
    print(f"\n📊 --- {commodity.upper()} MODEL SUMMARY ---")
    print(results.summary())
    print("-" * 50)
    
    # 5. Make the True Future Prediction
    forecast_obj = results.get_forecast(steps=len(X_future), exog=X_future)
    future_predictions = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int(alpha=0.50)
  
    print(f"\n{commodity.upper()} FUTURE PRICE FORECAST:")
    for date, pred, lower, upper in zip(future_predictions.index, future_predictions.values, conf_int.iloc[:, 0], conf_int.iloc[:, 1]):
        date_str = date.strftime('%Y-%m')
        print(f"   {date_str} | Estimated Price: ${pred:.2f} (Expected Range: ${lower:.2f} to ${upper:.2f})")

    # 6. EXPORT PREDICTIONS TO CSV
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

    # ==========================================
    # 7. Using metrics.py and cv.py to evaluate the model
    # ==========================================

    print(f"\n Running Cross-Validation to evaluate {commodity} model accuracy...")
    
    # Create the expanding window folds (Train on 5 years, test 1 month, expand)
    folds = create_expanding_window_folds(n_samples=len(df_hist), initial_window=60, horizon=1)
    
    all_actuals = []
    all_predictions = []
    cv_dates = [] # <--- Initialized here
    
    for train_idx, test_idx in folds:
        # Split data based on the fold indices
        y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]
        X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
        
        # Train a temporary model for this specific historical window
        cv_model = SARIMAX(
            endog=y_train_fold, 
            exog=X_train_fold,
            order=(1, 1, 1),              
            seasonal_order=(1, 1, 1, 12), 
            enforce_stationarity=True,
            enforce_invertibility=True
        )
        
        try:
            # FIX 1 & 2: Use the robust 'nm' solver and give it a "Warm Start" using the main model's weights
            cv_results = cv_model.fit(start_params=results.params, method='nm', maxiter=200, disp=False)
            cv_forecast = cv_results.get_forecast(steps=len(X_test_fold), exog=X_test_fold)
            pred_price = cv_forecast.predicted_mean.values[0]
            
            # FIX 3: The Sanity Check. If the model predicts an impossible price (< $0 or > $10), catch it!
            if pred_price < 0 or pred_price > 10:
                pred_price = y_train_fold.iloc[-1] # Fallback to last known price
                
        except:
            # If the statsmodels solver completely crashes and throws an error, fallback to last known price
            pred_price = y_train_fold.iloc[-1]
            
        # ALL THREE of these must be indented to match exactly inside the loop
        all_actuals.append(y_test_fold.values[0])
        all_predictions.append(pred_price)
        cv_dates.append(y_test_fold.index[0]) # <--- Now safely inside the loop!

    # Convert the lists to Pandas Series to feed into your metrics function
    actuals_series = pd.Series(all_actuals)
    predictions_series = pd.Series(all_predictions)
    
    # Calculate the final scores!
    final_scores = compute_metrics(actuals_series, predictions_series)
    print(f" FINAL EVALUATION SCORES ({final_scores['n_obs']} historical months tested):")
    print(f"   MAE  (Mean Abs Error):  {final_scores['mae']:.4f}")
    print(f"   RMSE (Root Mean Sq):    {final_scores['rmse']:.4f}")
    print(f"   MAPE (Mean % Error):    {final_scores['mape']:.2f}%")

    # 8. EXPORT METRICS TO CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Months Tested', 'MAE', 'RMSE', 'MAPE (%)'],
        'Score': [
            final_scores['n_obs'], 
            round(final_scores['mae'], 4), 
            round(final_scores['rmse'], 4), 
            round(final_scores['mape'], 2)
        ]
    })
    
    metrics_filename = output_folder / f"{commodity}_evaluation_metrics.csv"
    metrics_df.to_csv(metrics_filename, index=False)
    print(f" [SUCCESS] Metrics saved to: {metrics_filename}")
    # ==========================================
    # 9. AUTOMATIC PLOTTING (Actuals vs CV vs Future)
    # ==========================================
    print(f"📊 Generating and saving charts for {commodity}...")
    
    # Create a clean dataframe purely for the historical plotting
    df_cv = pd.DataFrame({
        'Actual': all_actuals,
        'Predicted': all_predictions
    }, index=cv_dates)
    df_cv['Error'] = abs(df_cv['Actual'] - df_cv['Predicted'])

    # Set up the 2-panel figure (stacked vertically)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

    # --- TOP PLOT: Price Forecast & History ---
    # 1. The baseline actual historical prices
    ax1.plot(df_hist.index, df_hist[target_col], color='black', linewidth=2.5, label='Actual Historical Price')
    
    # 2. How well the model guessed the past (Cross-Validation)
    ax1.plot(df_cv.index, df_cv['Predicted'], color='tab:blue', linewidth=2, alpha=0.9, label='SARIMAX CV Prediction')
    
    # 3. Where the model says we are going (Future Forecast)
    ax1.plot(future_predictions.index, future_predictions.values, color='tab:red', linestyle='--', linewidth=2.5, label='Future Forecast (2026)')
    # Shaded confidence interval
    ax1.fill_between(future_predictions.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='tab:red', alpha=0.15, label='Expected Range (50% CI)')

    ax1.set_title(f'{commodity.upper()} - SARIMAX Model vs Actual & Future Forecast', fontsize=15, fontweight='bold')
    ax1.set_ylabel('Price (CAD)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=11)

    # --- BOTTOM PLOT: Absolute Error Tracking ---
    ax2.plot(df_cv.index, df_cv['Error'], color='purple', linewidth=2, label='Absolute Prediction Error ($)')
    ax2.fill_between(df_cv.index, 0, df_cv['Error'], color='purple', alpha=0.1)
    
    ax2.set_title(f'{commodity.upper()} - Cross-Validation Error Analysis', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Absolute Error ($)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=11)

    # Format perfectly and save
    plt.tight_layout()
    plot_filename = output_folder / f"{commodity}_sarimax_forecast_plot.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close() # Closes the image in memory so the loop can start fresh for the next commodity
    
    print(f" [SUCCESS] Chart saved to: {plot_filename}")

print("\n ALL PIPELINES COMPLETE!")


