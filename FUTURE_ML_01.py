# FUTURE_ML_01.py
# This script uses the Prophet library to forecast sales data from a CSV file.
# It includes data preprocessing, model training, evaluation, and visualization of results with error handling.

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load CSV
try:
    df = pd.read_csv("sales_data_sample.csv", encoding='ISO-8859-1')
except FileNotFoundError:
    print("Error: 'sales_data_sample.csv' file not found.")
    exit()
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

# Data Preprocessing
try:
    df = df[['ORDERDATE', 'SALES']]
    df = df.dropna()
    df = df[df['SALES'] > 0]
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    df['ORDERDATE'] = df['ORDERDATE'].dt.tz_localize('UTC')
    df['ORDERDATE'] = df['ORDERDATE'].dt.tz_localize(None)
    df['ORDERDATE'] = df['ORDERDATE'].dt.date
    df = df.drop_duplicates(subset=['ORDERDATE'])
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    df = df.sort_values('ORDERDATE')
    df = df.reset_index(drop=True)
    df['SALES'] = df['SALES'].astype(float)
    sales_data = df.groupby('ORDERDATE')['SALES'].sum().reset_index()
    sales_data = sales_data.rename(columns={'ORDERDATE': 'ds', 'SALES': 'y'})
    sales_data['ds'] = pd.to_datetime(sales_data['ds'])
    sales_data = sales_data.sort_values('ds').reset_index(drop=True)
    sales_data['y'] = sales_data['y'].astype(float)
except Exception as e:
    print(f"Data preprocessing error: {e}")
    exit()

# Split data
try:
    train = sales_data[:-30]
    test = sales_data[-30:]
    if train.empty or test.empty:
        raise ValueError("Training or testing data is empty after splitting.")
    if len(train) < 30 or len(test) < 30:
        raise ValueError("Not enough data points for training or testing.")
except ValueError as ve:
    print(f"ValueError: {ve}")
    exit()
except Exception as e:
    print(f"Data splitting error: {e}")
    exit()

# Train model
try:
    model = Prophet()
    model.add_country_holidays(country_name='US')
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_seasonality(name='weekly', period=7, fourier_order=3)
    model.fit(train)
except ValueError as ve:
    print(f"ValueError: {ve}")
    exit()
except ImportError as ie:
    print(f"ImportError: {ie}")
    exit()
except KeyError as ke:
    print(f"KeyError: {ke}")
    exit()
except Exception as e:
    print(f"Model training error: {e}")
    exit()

# Forecast future
try:
    future = model.make_future_dataframe(periods=30)
    future['ds'] = pd.to_datetime(future['ds'])
    forecast = model.predict(future)  # Do not limit columns
except Exception as e:
    print(f"Forecasting error: {e}")
    exit()

# Evaluate model
try:
    predicted = forecast[['ds', 'yhat']].tail(30).reset_index(drop=True)
    predicted.columns = ['ds', 'yhat']
    actual = test.reset_index(drop=True)
    if actual.empty or predicted.empty:
        raise ValueError("Actual or predicted data is empty.")
    if len(actual) != len(predicted):
        raise ValueError("Length of actual and predicted data do not match.")
    predicted['y'] = actual['y'].astype(float)

    mae = mean_absolute_error(actual['y'], predicted['yhat'])
    rmse = np.sqrt(mean_squared_error(actual['y'], predicted['yhat']))
    mape = np.mean(np.abs((actual['y'] - predicted['yhat']) / actual['y'])) * 100

    print("\nForecast Accuracy Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
except Exception as e:
    print(f"Evaluation error: {e}")
    exit()

# Plot forecast
try:
    model.plot(forecast)
    plt.axvline(x=actual['ds'].iloc[0], color='r', linestyle='--')
    plt.title("Sales Forecast (Prophet)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Plotting forecast error: {e}")

# Plot forecast components
try:
    model.plot_components(forecast)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Plotting components error: {e}")
    exit()
