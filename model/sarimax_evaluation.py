import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path
from metrics import compute_metrics
from cv import create_expanding_window_folds
from metrics import compute_directional_accuracy
import matplotlib.pyplot as plt  
import warnings
from sklearn.preprocessing import StandardScaler
import numpy as np

warnings.filterwarnings("ignore")

# Setup Folders
data_folder = Path("../Feature-Engineering")
output_folder = Path("../sarimax-model-output")
output_folder.mkdir(exist_ok=True)

commodities = ['tomato', 'avocado']

seasonal_fill_features = ['MEAN_C', 'PRECIPITATION_MM', 'import_qty']

for commodity in commodities:

    # ==========================================
    # I. Predict the target price
    # ==========================================
    print(f"\n{'='*40}")
    print(f"PREDICTING FUTURE FOR: {commodity.upper()}")
    print(f"{'='*40}")
    
    # 1. Load Historical Data
    hist_path = data_folder / f"{commodity}_final_selective_log.csv"
    df_hist = pd.read_csv(hist_path, parse_dates=['Date'], index_col='Date')
    df_hist = df_hist.asfreq('MS').interpolate(method='linear')
    
    # 2. Load Future Features
    future_path = data_folder / f"{commodity}_future_features.csv"
    df_future = pd.read_csv(future_path, parse_dates=['Date'], index_col='Date')
    df_future = df_future.asfreq('MS')

    # 3. Log Transform
    if commodity in ['tomato']:
        if 'price_adjusted' in df_hist.columns:
            df_hist['price_adjusted'] = np.log(df_hist['price_adjusted'])
        if 'price_adjusted' in df_future.columns:
            df_future['price_adjusted'] = np.log(df_future['price_adjusted'])

    # 4. Define Target and Features
    target_col = 'price_adjusted'
    y_train = df_hist[target_col]
    X_train = df_hist.drop(columns=[target_col])

    # 5. Drop multicollinear columns for Avocado
    if commodity == 'avocado':
        cols_to_drop = ['MXN_CAD_lag_7']
        X_train = X_train.drop(columns=[c for c in cols_to_drop if c in X_train.columns])
        
    # 6. Scale & Train
    scaler = StandardScaler()
    X_future = df_future[X_train.columns]
    X_train[X_train.columns] = scaler.fit_transform(X_train)
    X_future[X_future.columns] = scaler.transform(X_future)

    if commodity == 'avocado':
        model = SARIMAX(endog=y_train, exog=X_train,
                        order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=True, enforce_invertibility=True)
    if commodity == 'tomato':
        model = SARIMAX(endog=y_train, exog=X_train,
                        order=(1,0,1), seasonal_order=(0,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=True)

    results = model.fit(disp=False)

    # 6.5 Calculate Training RMSE
    # compare the model's fitted values to the actual training data
    train_fitted = results.fittedvalues[13:]
    y_actual_subset = y_train[13:]
    
    if commodity == 'tomato':
        # Convert both to Real CAD before comparing
        t_rmse = np.sqrt(np.mean((np.exp(y_actual_subset) - np.exp(train_fitted))**2))
    else:
        t_rmse = np.sqrt(np.mean((y_actual_subset - train_fitted)**2))
        
    print(f" >>> Adjusted Training RMSE: {t_rmse:.4f}")


    # 7. Future Forecast
    forecast_obj = results.get_forecast(steps=len(X_future), exog=X_future)
    future_predictions = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int(alpha=0.50)
      
    # 8. Inverse log transform for tomato
    if commodity in ['tomato']:
        future_predictions = np.exp(future_predictions)
        conf_int = np.exp(conf_int)

    print(f"\n{commodity.upper()} FUTURE PRICE FORECAST:")
    for date, pred, lower, upper in zip(
        future_predictions.index, future_predictions.values,
        conf_int.iloc[:, 0], conf_int.iloc[:, 1]
    ):
        print(f"   {date.strftime('%Y-%m')} | Estimated Price: ${pred:.2f} "
              f"(Expected Range: ${lower:.2f} to ${upper:.2f})")

    # 9. Export Forecast CSV
    final_output = pd.DataFrame({
        'Predicted_Price': future_predictions.values,
        'Lower_CI':        conf_int.iloc[:, 0].values,
        'Upper_CI':        conf_int.iloc[:, 1].values
    }, index=future_predictions.index)
    final_output.index.name = 'Date'
    final_output.index = final_output.index.strftime('%Y-%m')
    output_filename = output_folder / f"{commodity}_sarima_predictions.csv"
    final_output.to_csv(output_filename)
    print(f" [SUCCESS] Predictions saved to: {output_filename}")


    # ==========================================
    # II. Evaluation with metrics.py and cv.py
    # ==========================================
    print(f"\n --- {commodity.upper()} MODEL SUMMARY ---")
    print(results.summary())
    print("-" * 50)
    print(f"\n Running Cross-Validation for {commodity}...")

    folds = create_expanding_window_folds(
        n_samples=len(df_hist), initial_window=48, horizon=1
    )

    all_actuals     = []
    all_predictions = []
    cv_dates        = []
    fold_records    = []   # store per-fold details for CSV export

    for train_idx, test_idx in folds:
        y_train_fold = y_train.iloc[train_idx]
        y_test_fold  = y_train.iloc[test_idx]
        X_train_fold = X_train.iloc[train_idx]
        X_test_fold  = X_train.iloc[test_idx]

        if commodity == 'avocado':
            cv_model = SARIMAX(endog=y_train_fold, exog=X_train_fold,
                               order=(1,1,1), seasonal_order=(1,1,1,12),
                               enforce_stationarity=True, enforce_invertibility=True)
        if commodity == 'tomato':
            cv_model = SARIMAX(endog=y_train_fold, exog=X_train_fold,
                               order=(1,0,1), seasonal_order=(0,1,1,12),
                               enforce_stationarity=False, enforce_invertibility=True)

        try:
            cv_results  = cv_model.fit(
                start_params=results.params, method='nm', maxiter=200, disp=False
            )
            cv_forecast = cv_results.get_forecast(steps=len(X_test_fold), exog=X_test_fold)
            pred_price  = cv_forecast.predicted_mean.values[0]
            cv_conf     = cv_forecast.conf_int(alpha=0.50)
            cv_lower    = cv_conf.iloc[0, 0]
            cv_upper    = cv_conf.iloc[0, 1]

            if pred_price < 0 or pred_price > 10:
                pred_price = y_train_fold.iloc[-1]
                cv_lower   = np.nan
                cv_upper   = np.nan

        except:
            # Model crashed — fall back to last known price, no CI available
            pred_price = y_train_fold.iloc[-1]
            cv_lower   = np.nan
            cv_upper   = np.nan

        actual_price = y_test_fold.values[0]

        # Inverse log for tomato so metrics are in real CAD
        if commodity == 'tomato':
            actual_display = np.exp(actual_price)
            pred_display   = np.exp(pred_price)
        else:
            actual_display = actual_price
            pred_display   = pred_price

        all_actuals.append(actual_display)
        all_predictions.append(pred_display)
        cv_dates.append(y_test_fold.index[0])
    
        # Record per-fold details
        fold_records.append({
            'Date':            y_test_fold.index[0].strftime('%Y-%m'),
            'Actual_Price':    round(actual_display, 4),
            'Predicted_Price': round(pred_display,   4),
            'Lower_CI':        round(np.exp(cv_lower) if commodity == 'tomato' else cv_lower, 4),
            'Upper_CI':        round(np.exp(cv_upper) if commodity == 'tomato' else cv_upper, 4),
            'Abs_Error':       round(abs(actual_display - pred_display), 4),
            'Pct_Error':       round(
                abs(actual_display - pred_display) / (actual_display + 1e-8) * 100, 2
            ),
        })
       

    actuals_series     = pd.Series(all_actuals)
    predictions_series = pd.Series(all_predictions)

    # ---- Enhanced metrics ----
    n    = len(actuals_series)
    mae  = np.mean(np.abs(actuals_series - predictions_series))
    rmse = np.sqrt(np.mean((actuals_series - predictions_series) ** 2))
    mape = np.mean(np.abs((actuals_series - predictions_series) /
                          (actuals_series + 1e-8))) * 100
    ss_res = np.sum((actuals_series - predictions_series) ** 2)
    ss_tot = np.sum((actuals_series - actuals_series.mean()) ** 2)
    r2   = 1 - ss_res / ss_tot
    bias = np.mean(predictions_series - actuals_series)

    print(f" FINAL EVALUATION SCORES ({n} historical months tested):")
    print(f"   MAE   (Mean Abs Error):   {mae:.4f}")
    print(f"   RMSE  (Root Mean Sq):     {rmse:.4f}")
    print(f"   MAPE  (Mean % Error):     {mape:.2f}%")
    print(f"   R2    (Explained Var):    {r2:.4f}")
    print(f"   Bias  (Over/Under-est):   {bias:.4f} "
          f"{'(overestimating)' if bias > 0 else '(underestimating)'}")
    
    da = compute_directional_accuracy(actuals_series, predictions_series)
    print(f"   DA    (Directional Acc):   {da:.2%}")

    # ---- Export summary metrics CSV ----
    metrics_df = pd.DataFrame({
        'Metric': ['Months Tested', 'MAE', 'RMSE', 'MAPE (%)', 'R2', 'Bias','DA'],
        'Score':  [n,
                   round(mae,  4),
                   round(rmse, 4),
                   round(mape, 2),
                   round(r2,   4),
                   round(bias, 4),
                   round(da,   4)]
    })
    metrics_filename = output_folder / f"{commodity}_evaluation_metrics.csv"
    metrics_df.to_csv(metrics_filename, index=False)
    print(f" [SUCCESS] Metrics saved to: {metrics_filename}")

    # ---- Export per-fold CV results CSV ----
    fold_df = pd.DataFrame(fold_records)
    fold_filename = output_folder / f"{commodity}_sarimax_cv_results.csv"
    fold_df.to_csv(fold_filename, index=False)
    print(f" [SUCCESS] Per-fold CV results saved to: {fold_filename}")

    # ---- Print 5 worst predicted months ----
    print(f"\n Top 5 Worst Predicted Months:")
    print(fold_df.nlargest(5, 'Abs_Error')[
        ['Date', 'Actual_Price', 'Predicted_Price', 'Abs_Error', 'Pct_Error']
    ].to_string(index=False))


    # ==========================================
    # II-B. 2025 Price Impact Simulation (Real Difference)
    # Goal: Compare price predicted with "Imputed" features against the "Actual" historical price.
    # ==========================================
    print(f"\n{'─'*50}")
    print(f" 2025 REAL DIFFERENCE SIMULATION ({commodity.upper()})")
    print(f"{'─'*50}")
 
    try:
        # 1. Load ground truth from the raw selective log
        df_raw = pd.read_csv(data_folder / f"{commodity}_final_selective_log.csv", 
                             parse_dates=['Date'], index_col='Date').asfreq('MS')
        
        # 2. Get actual prices for the 2025 stress-test window
        # These are the "Real Prices" that actually happened.
        actual_prices_raw = df_raw.loc['2025-01':'2025-04', target_col]
        
        # 3. Create the "Imputed" Feature set
        # We act as if we are in 2026 and don't have these values.
        X_imputed_raw = df_raw.loc['2025-01':'2025-04', X_train.columns].copy()
        ref_2024 = df_raw.loc['2024-01':'2024-04', X_train.columns]
        ref_dec2024 = df_raw.loc['2024-12', X_train.columns].iloc[0]

        for i, month in enumerate(X_imputed_raw.index):
            for col in X_train.columns:
                if col in seasonal_fill_features:
                    # Guessed via last year's pattern
                    X_imputed_raw.loc[month, col] = ref_2024.iloc[i][col]
                else:
                    # Guessed via last month's stability
                    X_imputed_raw.loc[month, col] = ref_dec2024[col]

        # 4. Scale and Predict using the existing model 'results'
        X_imputed_scaled = pd.DataFrame(scaler.transform(X_imputed_raw), 
                                        index=X_imputed_raw.index, 
                                        columns=X_imputed_raw.columns)
        
        sim_forecast = results.get_forecast(steps=len(X_imputed_scaled), exog=X_imputed_scaled)
        sim_pred_log = sim_forecast.predicted_mean
        
        # 5. Convert back to Real CAD (exp for Tomato)
        if commodity == 'tomato':
            sim_pred_final = np.exp(sim_pred_log)
            actual_final = np.exp(actual_prices_raw)
        else:
            sim_pred_final = sim_pred_log
            actual_final = actual_prices_raw

        # 6. Calculate the "Real Difference" in Dollars
        sim_records = []
        for date, a_p, s_p in zip(sim_pred_final.index, actual_final, sim_pred_final):
            diff = s_p - a_p
            pct_err = (abs(diff) / (a_p + 1e-8)) * 100
            sim_records.append({
                'Month': date.strftime('%Y-%m'),
                'Real_Price': round(a_p, 2),
                'Sim_Pred_Price': round(s_p, 2),
                'Dollar_Diff': round(diff, 2),
                'Impact_Error_%': round(pct_err, 2)
            })

        sim_results_df = pd.DataFrame(sim_records)
        print("\n [RESULT] Price Bias caused by Data Imputation:")
        print(sim_results_df.to_string(index=False))

    except Exception as e:
        print(f" [ERROR] Could not calculate real difference: {e}")

    # ==========================================
    # III. Plotting — 4-panel Dashboard
    # ==========================================
    print(f"\n Generating charts for {commodity}...")

    # Historical prices in real scale
    if commodity == 'tomato':
        hist_display = np.exp(df_hist[target_col])
    else:
        hist_display = df_hist[target_col]

    df_cv = pd.DataFrame({
        'Actual':    all_actuals,
        'Predicted': all_predictions,
    }, index=cv_dates)
    df_cv['Error']     = np.abs(df_cv['Actual'] - df_cv['Predicted'])
    df_cv['Pct_Error'] = df_cv['Error'] / (df_cv['Actual'] + 1e-8) * 100

    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(18, 12),
        gridspec_kw={'hspace': 0.4, 'wspace': 0.3}
    )
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ── Panel 1: Full Forecast (History + CV + Future) ──────────────────────
    ax1.plot(hist_display.index, hist_display.values,
             color='black', linewidth=2, label='Actual Historical Price')
    ax1.plot(df_cv.index, df_cv['Predicted'],
             color='tab:blue', linewidth=1.8, alpha=0.85,
             label='CV Predicted Price')
    ax1.plot(future_predictions.index, future_predictions.values,
             color='tab:red', linestyle='--', linewidth=2.2,
             label='Future Forecast (2026)')
    ax1.fill_between(future_predictions.index,
                     conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                     color='tab:red', alpha=0.15, label='50% Confidence Interval')
    ax1.set_title(f'{commodity.upper()} — Price Forecast vs Actual',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('Price (CAD $)', fontsize=11)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ── Panel 2: Actual vs Predicted Scatter ────────────────────────────────
    ax2.scatter(df_cv['Actual'], df_cv['Predicted'],
                alpha=0.65, color='tab:blue', edgecolors='navy',
                s=45, zorder=3)
    min_val = min(df_cv['Actual'].min(), df_cv['Predicted'].min()) * 0.95
    max_val = max(df_cv['Actual'].max(), df_cv['Predicted'].max()) * 1.05
    ax2.plot([min_val, max_val], [min_val, max_val],
             color='red', linestyle='--', linewidth=1.5,
             label='Perfect Prediction (y = x)')
    ax2.set_title(f'{commodity.upper()} — Actual vs Predicted  (R2 = {r2:.4f})',
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel('Actual Price ($)', fontsize=11)
    ax2.set_ylabel('Predicted Price ($)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # ── Panel 3: Absolute Error Over Time ───────────────────────────────────
    ax3.plot(df_cv.index, df_cv['Error'],
             color='purple', linewidth=1.8, label='Absolute Error ($)')
    ax3.fill_between(df_cv.index, 0, df_cv['Error'],
                     color='purple', alpha=0.10)
    ax3.axhline(y=mae, color='red', linestyle='--', linewidth=1.3,
                label=f'MAE = ${mae:.4f}')
    ax3.axhline(y=rmse, color='orange', linestyle=':', linewidth=1.3,
                label=f'RMSE = ${rmse:.4f}')
    ax3.set_title(f'{commodity.upper()} — Absolute Error Over Time',
                  fontsize=13, fontweight='bold')
    ax3.set_ylabel('Absolute Error ($)', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.legend(fontsize=9, loc='upper left')
    ax3.grid(True, linestyle='--', alpha=0.5)

    # ── Panel 4: % Error Distribution Histogram ─────────────────────────────
    ax4.hist(df_cv['Pct_Error'], bins=20,
             color='steelblue', edgecolor='white', alpha=0.85)
    ax4.axvline(x=mape, color='red', linestyle='--', linewidth=1.5,
                label=f'MAPE = {mape:.2f}%')
    ax4.set_title(f'{commodity.upper()} — % Error Distribution',
                  fontsize=13, fontweight='bold')
    ax4.set_xlabel('Percentage Error (%)', fontsize=11)
    ax4.set_ylabel('Frequency (Months)', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle(
        f'{commodity.upper()} SARIMAX — Full Evaluation Dashboard',
        fontsize=15, fontweight='bold', y=1.01
    )
    plot_filename = output_folder / f"{commodity}_sarimax_forecast_plot.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [SUCCESS] Chart saved to: {plot_filename}")

print("\n ALL PIPELINES COMPLETE!")
