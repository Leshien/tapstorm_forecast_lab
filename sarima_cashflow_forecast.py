
# SARIMA-based Cashflow Forecast for Tapstorm Studios
# Author: Tapstorm Financial Simulation Suite

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Load dataset (only actuals)
df = pd.read_csv("tapstorm_cashflow_forecast.csv")
df = df[df['Type'] == 'Actual']
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df = df.asfreq('MS')  # Ensure monthly frequency

# Optional: log-transform to stabilize variance
series = df['Cash Collected (€)']

# Fit SARIMA model (example order, should be tuned for production)
model = sm.tsa.statespace.SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
results = model.fit(disp=False)

# Forecast next 12 months
forecast = results.get_forecast(steps=12)
forecast_df = forecast.summary_frame()
forecast_df = forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']]
forecast_df.index.name = 'Month'
forecast_df.columns = ['yhat', 'yhat_lower', 'yhat_upper']

# Combine historical + forecast
combined = pd.concat([series.rename("Actual"), forecast_df], axis=1)

# Save to CSV
combined.to_csv("tapstorm_sarima_forecast_output.csv")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(combined.index, combined['Actual'], label="Actual")
plt.plot(combined.index, combined['yhat'], label="Forecast", linestyle='--')
plt.fill_between(combined.index, combined['yhat_lower'], combined['yhat_upper'], color='gray', alpha=0.2)
plt.title("SARIMA Cashflow Forecast - Tapstorm Studios")
plt.xlabel("Month")
plt.ylabel("Cash Collected (€)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
