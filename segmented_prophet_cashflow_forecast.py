
# Segmented Prophet Forecast by Segment
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv("tapstorm_cashflow_forecast.csv")
df = df[df['Type'] == 'Actual']
segments = df['Segment'].unique()

all_forecasts = []

for seg in segments:
    df_seg = df[df['Segment'] == seg]
    df_seg = df_seg[['Month', 'Cash Collected (€)']].rename(columns={'Month': 'ds', 'Cash Collected (€)': 'y'})
    df_seg['ds'] = pd.to_datetime(df_seg['ds'])

    model = Prophet()
    model.fit(df_seg)

    future = model.make_future_dataframe(periods=12, freq='MS')
    forecast = model.predict(future)
    forecast['Segment'] = seg
    all_forecasts.append(forecast[['ds', 'yhat', 'Segment']])

segmented_df = pd.concat(all_forecasts)
segmented_df.to_csv("tapstorm_segmented_prophet_forecast_output.csv", index=False)
