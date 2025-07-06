
# Prophet Forecast for Tapstorm Studios
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv("tapstorm_cashflow_forecast.csv")
df = df[df['Type'] == 'Actual']
df = df[['Month', 'Cash Collected (€)']].rename(columns={'Month': 'ds', 'Cash Collected (€)': 'y'})
df['ds'] = pd.to_datetime(df['ds'])

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=12, freq='MS')
forecast = model.predict(future)

forecast[['ds', 'yhat']].to_csv("tapstorm_prophet_forecast_output.csv", index=False)

fig = model.plot(forecast)
plt.title("Tapstorm Studios - Prophet Forecast")
plt.xlabel("Month")
plt.ylabel("Cash Collected (€)")
plt.grid(True)
plt.tight_layout()
plt.show()
