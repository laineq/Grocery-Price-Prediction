# Airflow DAG Structure

This folder contains the Airflow DAGs for the data pipeline.

There are two main folders:

- `full/`: one-time backfill DAGs for historical data
- `monthly/`: update DAGs for new monthly data


## Full DAGs

These DAGs are used to load and transform historical data.

- `canadian_agricultural_import_full.py`
  Loads historical Canada agricultural import data and saves Silver data for avocado and tomato.

- `canadian_grocery_prices_full.py`
  Loads historical Canada grocery prices and saves Silver data for avocado and tomato.

- `consumer_price_index_full.py`
  Loads historical CPI proxy data for avocado and tomato.

- `exchange_rate_full.py`
  Loads historical exchange rates and saves yearly Silver files.

- `grocery_price_adjusted_full.py`
  Combines grocery prices and CPI data to create adjusted price datasets.

- `mexico_weather_full.py`
  Loads historical Mexico weather data and keeps only the states used in the model.

- `oil_prices_full.py`
  Loads historical U.S. and Canada oil price data, converts them, and creates integrated oil price Silver data.

- `gold_features_full.py`
  Builds the final avocado and tomato feature tables for modeling and stores them in Gold.


## Monthly DAGs

These DAGs are used for new monthly updates.

- `canadian_agricultural_import_monthly.py`
  Updates current-year agricultural import data.

- `canadian_grocery_prices_monthly.py`
  Updates current-year Canada grocery prices.

- `consumer_price_index_monthly.py`
  Updates current-year CPI proxy data.

- `exchange_rate_monthly.py`
  Updates current-year exchange rate data.

- `grocery_price_adjusted_monthly.py`
  Rebuilds adjusted grocery price Silver data when grocery prices or CPI are updated.

- `mexico_weather_monthly.py`
  Updates current-year Mexico weather Silver data.

- `oil_prices_monthly.py`
  Updates current-year U.S. and Canada oil prices and rebuilds integrated oil price Silver data.

- `gold_features_monthly.py`
  Rebuilds the final Gold feature tables after upstream monthly Silver data changes.


## Simple Data Flow

The pipeline follows this order:

1. Raw source data goes to `bronze/`
2. Cleaned and structured data goes to `silver/`
3. Final model-ready tables go to `gold/`


## Monthly Trigger Flow

Some monthly DAGs trigger downstream DAGs after Silver is updated.

- `canadian_grocery_prices_monthly`
- `consumer_price_index_monthly`

These trigger:

- `grocery_price_adjusted_monthly`

Then these Silver-producing DAGs can trigger:

- `grocery_price_adjusted_monthly`
- `canadian_agricultural_import_monthly`
- `mexico_weather_monthly`
- `exchange_rate_monthly`
- `oil_prices_monthly`

These trigger:

- `gold_features_monthly`


## Note

`full/` DAGs are mainly for backfill.

`monthly/` DAGs are mainly for regular updates and overwrite the latest yearly or final datasets when needed.
