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
