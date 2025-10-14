# Copy and paste this code into a new cell in your notebook to define all variables

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
import pandas as pd

# Function to prepare train/test split
def prepare_data(df, train_ratio=0.8):
    data = df["Adj Close"].dropna()
    train_size = int(len(data) * train_ratio)
    train, test = data[:train_size], data[train_size:]
    return train, test

# Function to fit MA model
def fit_ma_model(train, test, q=20):
    model = ARIMA(train, order=(0, 0, q))
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test))
    forecast.index = test.index
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    return fitted, forecast, rmse, mae

# Function to fit AR model
def fit_ar_model(train, test, p=20):
    model = ARIMA(train, order=(p, 0, 0))
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test))
    forecast.index = test.index
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    return fitted, forecast, rmse, mae

# Function to fit ARIMA model
def fit_arima_model(train, test, p=20, d=1, q=20):
    model = ARIMA(train, order=(p, d, q))
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test))
    forecast.index = test.index
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    return fitted, forecast, rmse, mae

# Get stock data (if not already defined)
if 'aapl' not in locals():
    def get_stock_data(symbol: str, start="2020-01-01", end=None):
        if end is None:
            end = pd.Timestamp.today().strftime('%Y-%m-%d')
        print(f"Fetching {symbol}...")
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
        if df.empty:
            print(f"⚠️ Error fetching {symbol}: No data returned")
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        print(f"✓ Fetched {len(df)} days of data for {symbol}")
        return df
    
    aapl = get_stock_data("AAPL")
    nvda = get_stock_data("NVDA")
    lyft = get_stock_data("LYFT")

# Prepare data for all stocks
print("Preparing train/test data...")
aapl_train, aapl_test = prepare_data(aapl)
nvda_train, nvda_test = prepare_data(nvda)
lyft_train, lyft_test = prepare_data(lyft)

print(f"AAPL - Training: {len(aapl_train)} days, Test: {len(aapl_test)} days")
print(f"NVDA - Training: {len(nvda_train)} days, Test: {len(nvda_test)} days")
print(f"LYFT - Training: {len(lyft_train)} days, Test: {len(lyft_test)} days")

# Fit MA models
print("\nFitting MA models...")
aapl_ma_fitted, aapl_ma_forecast, aapl_ma_rmse, aapl_ma_mae = fit_ma_model(aapl_train, aapl_test)
nvda_ma_fitted, nvda_ma_forecast, nvda_ma_rmse, nvda_ma_mae = fit_ma_model(nvda_train, nvda_test)
lyft_ma_fitted, lyft_ma_forecast, lyft_ma_rmse, lyft_ma_mae = fit_ma_model(lyft_train, lyft_test)

# Fit AR models
print("Fitting AR models...")
aapl_ar_fitted, aapl_ar_forecast, aapl_ar_rmse, aapl_ar_mae = fit_ar_model(aapl_train, aapl_test)
nvda_ar_fitted, nvda_ar_forecast, nvda_ar_rmse, nvda_ar_mae = fit_ar_model(nvda_train, nvda_test)
lyft_ar_fitted, lyft_ar_forecast, lyft_ar_rmse, lyft_ar_mae = fit_ar_model(lyft_train, lyft_test)

# Fit ARIMA models
print("Fitting ARIMA models...")
aapl_arima_fitted, aapl_arima_forecast, aapl_arima_rmse, aapl_arima_mae = fit_arima_model(aapl_train, aapl_test)
nvda_arima_fitted, nvda_arima_forecast, nvda_arima_rmse, nvda_arima_mae = fit_arima_model(nvda_train, nvda_test)
lyft_arima_fitted, lyft_arima_forecast, lyft_arima_rmse, lyft_arima_mae = fit_arima_model(lyft_train, lyft_test)

print("\nAll models fitted successfully!")
print("Variables are now defined and ready for plotting.")

# Print some results
print(f"\nAAPL MA(20) - RMSE: ${aapl_ma_rmse:.2f}, MAE: ${aapl_ma_mae:.2f}")
print(f"AAPL AR(20) - RMSE: ${aapl_ar_rmse:.2f}, MAE: ${aapl_ar_mae:.2f}")
print(f"AAPL ARIMA(20,1,20) - RMSE: ${aapl_arima_rmse:.2f}, MAE: ${aapl_arima_mae:.2f}")
