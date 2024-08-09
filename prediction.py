# LORD AMDAL - Ahmed Ali 
# Basic Stock Market Prediction using Prophet
# Install required libraries: pip install yfinance prophet plotly pandas numpy matplotlib seaborn
# Disclaimer: This code is for educational purposes only and should not be used for malicious purposes or as financial advisor.
# This code still needs to be adjusted according to your specific requirements.

import yfinance as yf
from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta

# Stock symbol and date range definition
stock_symbol = 'NKE'
start_date = '2020-01-01'  # Extended data range for better model training
end_date = datetime.now().strftime('%Y-%m-%d')

# Download stock data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Preprocess data
stock_data.reset_index(inplace=True)
data = stock_data[['Date', 'Close']]
data.columns = ['ds', 'y']

# Create and train the Prophet model
model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True,
                changepoint_prior_scale=0.05, seasonality_prior_scale=10)
model.fit(data)

# Generate future dates for prediction
future_dates = model.make_future_dataframe(periods=180)
forecast = model.predict(future_dates)

# Calculate prediction errors
y_true = data['y'].values
y_pred = forecast['yhat'][:len(y_true)]
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

# Create an elaborate dashboard
fig = make_subplots(rows=3, cols=2, subplot_titles=('Stock Price and Prediction', 'Components', 
                                                    'Prediction Error Distribution', 'Weekly Seasonality',
                                                    'Yearly Seasonality', 'Trend and Changepoints'))

# Stock Price and Prediction
fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Actual', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prediction', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill=None, mode='lines', line_color='rgba(0,0,255,0.2)', name='Lower Bound'), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines', line_color='rgba(0,0,255,0.2)', name='Upper Bound'), row=1, col=1)

# Components
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='Trend'), row=1, col=2)

# Prediction Error Distribution
error = y_true - y_pred
fig.add_trace(go.Histogram(x=error, nbinsx=50, name='Error Distribution'), row=2, col=1)

# Weekly Seasonality
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_seasonality = forecast['weekly'].iloc[0:7].values
fig.add_trace(go.Bar(x=days, y=weekly_seasonality, name='Weekly Seasonality'), row=2, col=2)

# Yearly Seasonality
yearly_seasonality = forecast['yearly'].iloc[0:365].values
months = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
fig.add_trace(go.Scatter(x=months, y=yearly_seasonality, mode='lines', name='Yearly Seasonality'), row=3, col=1)

# Trend and Changepoints
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='Trend'), row=3, col=2)
changepoints = model.changepoints
for changepoint in changepoints:
    fig.add_vline(x=changepoint, line_dash="dash", line_color="red", row=3, col=2)

# Update layout
fig.update_layout(height=1200, width=1200, title_text=f"{stock_symbol} Stock Analysis Dashboard")
fig.update_xaxes(title_text="Date", row=1, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_xaxes(title_text="Date", row=1, col=2)
fig.update_yaxes(title_text="Trend", row=1, col=2)
fig.update_xaxes(title_text="Error", row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)
fig.update_xaxes(title_text="Day of Week", row=2, col=2)
fig.update_yaxes(title_text="Effect", row=2, col=2)
fig.update_xaxes(title_text="Month", row=3, col=1)
fig.update_yaxes(title_text="Effect", row=3, col=1)
fig.update_xaxes(title_text="Date", row=3, col=2)
fig.update_yaxes(title_text="Trend", row=3, col=2)

# Show the dashboard
fig.show()

# Print prediction errors
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Save forecast (optional)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('forecasted_table.csv', index=False)