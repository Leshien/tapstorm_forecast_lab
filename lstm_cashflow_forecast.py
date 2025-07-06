
# LSTM Cashflow Forecast for Tapstorm Studios
# Author: Tapstorm Financial Simulation Suite

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load dataset
df = pd.read_csv("tapstorm_cashflow_forecast.csv")
df = df[df['Type'] == 'Actual']
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df = df.asfreq('MS')

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(df[['Cash Collected (€)']].values)

# Create sequences (6 months → 1 prediction)
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

SEQ_LENGTH = 6
X, y = create_sequences(data, SEQ_LENGTH)

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=200, verbose=0)

# Forecast next 12 months
predictions = []
current_input = X[-1]

for _ in range(12):
    pred = model.predict(current_input.reshape(1, SEQ_LENGTH, 1), verbose=0)
    predictions.append(pred[0][0])
    current_input = np.append(current_input[1:], [[pred]], axis=0)

# Inverse transform predictions
pred_scaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Build forecast DataFrame
forecast_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
forecast_df = pd.DataFrame({
    'Month': forecast_dates,
    'yhat': pred_scaled
})

# Combine with actuals for plot
actual_df = df[['Cash Collected (€)']].reset_index().rename(columns={'Cash Collected (€)': 'Actual'})
combined = pd.concat([actual_df, forecast_df], ignore_index=True)

# Save forecast
forecast_df.to_csv("tapstorm_lstm_forecast_output.csv", index=False)

# Plot
plt.figure(figsize=(12,6))
plt.plot(actual_df['Month'], actual_df['Actual'], label="Actual")
plt.plot(forecast_df['Month'], forecast_df['yhat'], label="LSTM Forecast", linestyle='--')
plt.title("LSTM Cashflow Forecast - Tapstorm Studios")
plt.xlabel("Month")
plt.ylabel("Cash Collected (€)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
