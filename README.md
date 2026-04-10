SFU BigData Lab2 Project

## requirement  

In your repository, please include a file README.txt (or README.md if you prefer) indicating how we can actually test your project as well as other notes about things we should look for. If you created some kind of web frontend, please include a URL in the README.md as well.

**Internal Document:**   
https://docs.google.com/document/d/1qe57lMZQkCWW0wimnmd1cKWC8teOTCj8oKzUkfZY7gU/edit?tab=t.0

---

# Grocery Price Prediction (Canada) (Ongoing)

This project forecasts monthly Canadian grocery prices for **avocado and tomato**. It combines historical retail prices with external indicators such as **import volumes, gas prices, weather (temperature and precipitation), exchange rates, and CPI** to improve prediction accuracy.

The system is designed as an **end-to-end automated pipeline**, where data is updated monthly and predictions are generated automatically.

## Final Outputs

- **Web application (GroceryCast):** [http://35.91.193.142:3000](http://35.91.193.142:3000)
- **Interactive dashboard:** [https://jli624.shinyapps.io/grocerypriceprediction/](https://jli624.shinyapps.io/grocerypriceprediction/)

**GroceryCast** is our main web app for grocery price forecasting. It is designed to be simple and easy to use. Users can check predicted prices for upcoming periods, see whether prices are expected to go up or down, and view historical price trends.

We also provide a separate interactive dashboard for deeper analysis. The dashboard includes model comparisons, predicted vs. actual price plots, historical trends, and lag analysis. The web app is meant for quick and simple use, while the dashboard gives more detail for users who want to explore the results further.

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

At a high level, the monthly workflow begins by checking public data sources for new grocery price, CPI, import, exchange-rate, oil-price, and weather data. These datasets are ingested into the pipeline, standardized into monthly analytical tables, merged into product-specific feature sets, and then passed to the forecasting stage. The final DAG converts model outputs into JSON files that are consumed directly by the GroceryCast web application.

### Airflow DAG Flow

```mermaid
flowchart TD
    A["canadian_grocery_prices_monthly"] --> B["grocery_price_adjusted_monthly"]
    C["consumer_price_index_monthly"] --> B
    D["canadian_agricultural_import_monthly"] --> E["gold_features_monthly"]
    F["exchange_rate_monthly"] --> E
    G["oil_prices_monthly"] --> E
    H["mexico_weather_monthly"] --> E
    B --> E
    E --> I["future_features_monthly"]
    I --> J["prediction_monthly"]
    J --> K["app_output_monthly"]
```

The DAG structure is modular rather than monolithic. Source-specific ingestion DAGs independently collect and clean monthly data for grocery prices, CPI, imports, exchange rates, oil prices, and weather. These upstream outputs feed into `grocery_price_adjusted_monthly` and `gold_features_monthly`, where inflation-adjusted price targets and model-ready feature tables are created. Next, `future_features_monthly` extends the exogenous variables to the next forecast month, `prediction_monthly` runs the SARIMAX forecasting step, and `app_output_monthly` prepares the final application payloads.

### Data Layers

```mermaid
flowchart LR
    A["Bronze<br/>Raw source snapshots"] --> B["Silver<br/>Cleaned and standardized monthly tables"]
    B --> C["Gold<br/>Product-specific feature tables"]
    C --> D["Prediction<br/>SARIMAX forecasts + confidence intervals"]
    D --> E["App Output<br/>JSON for GroceryCast"]
```

The pipeline follows a **Bronze-Silver-Gold** architecture. In the Bronze layer, raw snapshots from external data sources are stored without modification so the original inputs are preserved. In the Silver layer, those raw files are cleaned, filtered, and standardized into monthly datasets that can be merged consistently across sources. In the Gold layer, the Silver datasets are combined into product-specific feature tables that include the variables needed for forecasting. These Gold outputs are then used by the prediction stage to generate next-month forecasts and confidence intervals, which are finally transformed into JSON files for the GroceryCast frontend.

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

---

## Code Structure
- **Feature-Engineering**
  -  **calculate_lag.py: Computes correlations between variables across all lag values up to a maximum lag of 12 months.**
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
  -  **feature_lag.py: Generates the final feature sets by applying selected lags and preparing both training and future datasets.**
     -  input: Same datasets as calculate_lag.py, and the following lag datasets:
         -  tomato_lag_results_manual.csv 
         -  avocado_lag_results_manual.csv
     -  output: 
         -  avocado_final_selective_log.csv (lagged avocado features with target variable; last row corresponds to the final observed price)
         -  tomato_final_selective_log.csv (lagged tomato features with target variable; last row corresponds to the final observed price)
         -  avocado_future_features.csv (lagged avocado features for forecasting, covering periods after the last observed price to the prediction horizon)
         -  tomato_future_features.csv (lagged tomato features for forecasting, covering periods after the last observed price to the prediction horizon)
- **model: Training, evaluation, and prediction pipeline**
   -   **sarimax_predict_future.py:Predicts the target price using the SARIMAX model**
        -  input: Outputs generated from feature_lag.py in the Feature-Engineering module
            -  avocado_final_selective_log.csv
            -  tomato_final_selective_log.csv
            -  avocado_future_features.csv
            -  tomato_future_features.csv
        - output: Predicted prices along with confidence intervals, stored in **sarimax-model-output** module
          - avocado_sarima_predictions.csv
          - tomato_sarima_predictions.csv
    -   **sarimax_evaluation.py:Evaluate the SARIMAX model**
        -  input: Same datasets as sarimax_predict_future.py
        - output: Historical predictions versus actual prices (last five years), including confidence intervals, stored in **sarimax-model-output** module.
          - avocado_sarimax_cv_results.csv
          - tomato_sarimax_cv_results.csv
    
