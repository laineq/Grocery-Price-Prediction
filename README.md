SFU BigData Lab2 Project

**Internal Document:**  
https://docs.google.com/document/d/1qe57lMZQkCWW0wimnmd1cKWC8teOTCj8oKzUkfZY7gU/edit?tab=t.0

---

# Grocery Price Prediction (Canada) (Ongoing)

This project forecasts monthly Canadian grocery prices for **avocado and tomato**. It combines historical retail prices with external indicators such as **import volumes, gas prices, weather (temperature and precipitation), exchange rates, and CPI** to improve prediction accuracy.

The system is designed as an **end-to-end automated pipeline**, where data is updated monthly and predictions are generated automatically.

---

## Project Goal

The goal of this project is to:

- predict next-month grocery prices for avocado and tomato in Canada  
- compare baseline, time series, and machine learning approaches  
- incorporate external supply chain and economic signals  
- build an automated forecasting pipeline  
- develop an application and dashboard for visualization  

---

## Data Sources

This project integrates multiple monthly datasets:

- **Canadian retail grocery prices** (target variable)  
- **Agricultural import data** (avocado and tomato)  
- **Gas price signals** (U.S., Canada, Mexico → weighted fuel cost proxy)  
- **Exchange rates** (CAD/USD, CAD/MXN)  
- **Weather data (temperature and precipitation from Mexican production regions)**  
- **Canadian CPI** (inflation adjustment)  
- **FAO food price index** (supplementary signal)  

---

## Automated Pipeline (Airflow)

This project leverages **Apache Airflow** to automate the end-to-end workflow:

- monthly data ingestion  
- data validation and cleaning  
- feature engineering  
- model training  
- price prediction generation  

This enables continuous updates and **automated monthly forecasting**.

---

## Feature Engineering

Key feature engineering steps include:

- **Lag features**: target and external variables with 1–12 month lags  
- **CPI adjustment**: normalize prices to real values (base year)  
- **Fuel cost proxy**: weighted combination of U.S., Canada, and Mexico gas prices  
- **Weather features**: temperature and precipitation from production regions  
- **Log transformation**: applied to stabilize variance for certain models  

These features are designed to capture **seasonality, supply chain delays, and macroeconomic effects**.

---

## Modeling Approach

We evaluate multiple forecasting approaches:

### Baselines
- Naive  
- Seasonal Naive  

### Time Series Models
- SARIMA  
- SARIMAX (with external indicators)  

### Machine Learning
- XGBoost  

To better reflect real-world supply chain dynamics, we incorporate **lagged external features (1–6 months)** to capture delays between production, transportation, and retail pricing.

---

## Evaluation

Models are evaluated using:

- **MAE (Mean Absolute Error)**  
- **RMSE (Root Mean Squared Error)**  
- **MAPE (Mean Absolute Percentage Error)**  

We use **expanding window cross-validation with one-step-ahead forecasting**, which simulates a real-world prediction setting where only past data is available at each step.

---

## Application & Dashboard (Planned)

We plan to build:

- a lightweight **application** for accessing model predictions  
- an interactive **dashboard** to visualize:
  - price trends  
  - forecasts  
  - feature importance / impacts  

This will enable users to easily explore model outputs and gain insights into price dynamics.

## Team Members
- Joohyun Park
- Jiayi Li
- Hongrui Qu
- Tracy Cui

## Code List
- Feature-Engineering
  -  calculate_lag.py: Computes correlations between variables across all lag values up to a maximum lag of 12 months.
     -  input: Cleaned datasets from the AdjustedData folder:
         -  avocado_price_adjusted.csv, 
         -  tomato_price_adjusted.csv, 
         -  avocado_import.csv,              
         -  tomato_import.csv, 
         -  mexico_weather_adjusted.csv, 
         -  gas_price.csv, 
         -  xrate_adjusted.csv
     -  output: 
         -  avocado_lag_results_manual.csv (selected lag features of avocado)
         -  tomato_lag_results_manual.csv (selected lag featutures of tomato)
         -  avocado_correlations_lag12.csv (full lag correlation results of avocado)
         -  tomato_correlations_lag12.csv (full lag correlation results of tomato)
  -  feature_lag.py: Generates the final feature sets by applying selected lags and preparing both training and future datasets.
     -  input: Same datasets as calculate_lag.py
     -  output: 
         -  avocado_final_selective_log.csv (lagged avocado features with target variable; last row corresponds to the final observed price)
         -  tomato_final_selective_log.csv (lagged tomato features with target variable; last row corresponds to the final observed price)
         -  avocado_future_features.csv (lagged avocado features for forecasting, covering periods after the last observed price to the prediction horizon)
         -  tomato_future_features.csv (lagged tomato features for forecasting, covering periods after the last observed price to the prediction horizon)
- model: tTraining, evaluation, and prediction pipeline
   -   sarimax_predict_future.py:Predicts the target price using the SARIMAX model
     -  input: Outputs generated from feature_lag.py in the Feature-Engineering module
        -  avocado_final_selective_log.csv
        -  tomato_final_selective_log.csv
        -  avocado_future_features.csv
        -  tomato_future_features.csv
     - output: Predicted prices along with confidence intervals
        - avocado_sarima_predictions.csv
        - tomato_sarima_predictions.csv
- 
- 
- 

