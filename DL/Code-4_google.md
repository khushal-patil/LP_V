# Practical-4 (Recurrent Neural Network - Google Stock Price Dataset)

Problem Statement: Recurrent neural network (RNN): Use the Google stock prices dataset and design a time series analysis and prediction system using RNN.

> [!NOTE]
> Dataset available in [Datasets](../Datasets/GOOG.csv) directory.
> In the code, dataset is downloaded directly from Keras/TensorFlow in 2nd step (Load Dataset)

---

## Pre-requisities

1. Install packages using `pip`: `pip install tensorflow keras numpy pandas matplotlib scikit-learn yfinance` (`tensorflow` requires Python 3.9 - 3.12)

## Steps

1. Import Libraries
2. Load Dataset
3. Exploratory Data Analysis (EDA)
4. Visualize Closing Price Over Time
5. Preprocess Data - Normalize Closing Price
6. Create Sequences for RNN Input
7. Build the RNN Model
8. Train the Model
9. Plot Training vs Validation Loss
10. Make Predictions and Inverse Scale
11. Evaluate the Model
12. Plot Actual vs Predicted Stock Price
13. Forecast Next 30 Days

---

## Code

### 1. Import Libraries:

```python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, SimpleRNN, Dropout
from tensorflow.keras.callbacks import EarlyStopping
```

### 2. Load Dataset:

```python3
# Downloads GOOGL stock data from Yahoo Finance for the given date range
ticker = "GOOGL"
df = yf.download(ticker, start="2018-01-01", end="2024-01-01")

# --- Offline alternative (comment out the yf.download above and use this instead if using local dataset) ---
# df = pd.read_csv('GOOGL.csv', index_col='Date', parse_dates=True)
# df = df.sort_index()  # ensure chronological order

# yfinance returns MultiIndex columns — flatten to single level
df.columns = df.columns.get_level_values(0)

print(f"Dataset Shape: {df.shape}")
print(f"Date Range: {df.index.min().date()} to {df.index.max().date()}")
print(df.head())
```

### 3. Exploratory Data Analysis (EDA):

```python3
print("=== Dataset Info ===")
print(df.info())
print("\n=== Statistical Summary ===")
print(df.describe())
print("\n=== Missing Values ===")
print(df.isnull().sum())
```

### 4. Visualize Closing Price Over Time:

```python3
plt.figure(figsize=(16, 6))
plt.plot(df.index, df['Close'], color='steelblue', linewidth=1.5, label='Close Price')
plt.title('Google (GOOGL) Stock Closing Price (2018–2024)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

### 5. Preprocess Data - Normalize Closing Price:

```python3
data = df[['Close']].values   # use only Close price for prediction

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)  # scale values to [0, 1]

print(f"Original data range: [{data.min():.2f}, {data.max():.2f}]")
print(f"Scaled data range:   [{data_scaled.min():.4f}, {data_scaled.max():.4f}]")
print(f"Total data points:   {len(data_scaled)}")
```

### 6. Create Sequences for RNN Input:

```python3
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i, 0])  # window of past `time_steps` days
        y.append(data[i, 0])                  # next day's price
    return np.array(X), np.array(y)

TIME_STEPS = 60  # use past 60 days to predict the next day

# 80/20 train-test split (manual, to preserve time order)
train_size = int(len(data_scaled) * 0.80)
train_data = data_scaled[:train_size]
test_data  = data_scaled[train_size - TIME_STEPS:]  # overlap ensures test sequences start correctly

X_train, y_train = create_sequences(train_data, TIME_STEPS)
X_test,  y_test  = create_sequences(test_data,  TIME_STEPS)

# Reshape to [samples, time_steps, features] — required format for RNN layers
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test  = X_test.reshape((X_test.shape[0],   X_test.shape[1],  1))

print(f"Training samples: {X_train.shape}")
print(f"Testing samples:  {X_test.shape}")
```

### 7. Build the RNN Model:

```python3
model = Sequential()

model.add(Input(shape=(TIME_STEPS, 1)))                               # input: sequence of 60 days
model.add(SimpleRNN(units=64, return_sequences=True))                 # first RNN layer, passes output to next
model.add(Dropout(0.2))                                               # drop 20% neurons to reduce overfitting
model.add(SimpleRNN(units=64, return_sequences=False))                # second RNN layer, outputs single vector
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))                         # fully connected layer
model.add(Dense(units=1))                                             # output: single predicted price

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()
```

### 8. Train the Model:

```python3
# EarlyStopping stops training if val_loss doesn't improve for 10 consecutive epochs
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    validation_split=0.1,   # use 10% of training data for validation
    callbacks=[early_stop],
    verbose=1
)
print(f"\nTraining stopped at epoch: {len(history.history['loss'])}")
```

### 9. Plot Training vs Validation Loss:

```python3
plt.plot(history.history['loss'], label='Train Loss', color='royalblue')
plt.plot(history.history['val_loss'], label='Val Loss', color='tomato')
plt.title('Model Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

### 10. Make Predictions and Inverse Scale:

```python3
y_pred_scaled = model.predict(X_test)

# Convert scaled predictions back to original USD price range
y_pred   = scaler.inverse_transform(y_pred_scaled)
y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

print(f"Sample predictions (first 5): {y_pred[:5].flatten().round(2)}")
print(f"Actual values      (first 5): {y_actual[:5].flatten().round(2)}")
```

### 11. Evaluate the Model:

```python3
mse  = mean_squared_error(y_actual, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_actual, y_pred)
mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100  # mean absolute percentage error

print("=" * 40)
print("     MODEL EVALUATION METRICS")
print("=" * 40)
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  MAPE : {mape:.2f}%")
print("=" * 40)
```

### 12. Plot Actual vs Predicted Stock Price:

```python3
test_dates = df.index[train_size:]   # align dates with test predictions

plt.figure(figsize=(16, 6))
plt.plot(test_dates, y_actual, label='Actual Price',    color='steelblue', linewidth=1.5)
plt.plot(test_dates, y_pred,   label='Predicted Price', color='tomato',    linewidth=1.5, linestyle='--')
plt.title('Google Stock Price: Actual vs Predicted (RNN)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

### 13. Forecast Next 30 Days:

```python3
n_future = 30  # number of future days to predict

# Seed the forecast with the last TIME_STEPS days of known data
future_input       = data_scaled[-TIME_STEPS:].reshape(1, TIME_STEPS, 1)
future_predictions = []

for _ in range(n_future):
    pred = model.predict(future_input, verbose=0)
    future_predictions.append(pred[0, 0])
    # Slide the window: drop oldest day, append new prediction
    future_input = np.append(future_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

# Inverse scale forecasted prices back to USD
future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Generate business day dates starting from the day after last known date
last_date    = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_future, freq='B')

plt.figure(figsize=(16, 6))
plt.plot(df.index[-120:], scaler.inverse_transform(data_scaled[-120:]),
         label='Historical', color='steelblue', linewidth=1.5)
plt.plot(future_dates, future_prices,
         label='30-Day Forecast', color='orange', linewidth=1.5)
plt.axvline(x=last_date, color='gray', linestyle='--', label='Forecast Start')
plt.title('Google Stock — 30-Day Future Price Forecast (RNN)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nForecasted price range: {future_prices.min():.2f} USD - {future_prices.max():.2f} USD")
```

---

## Miscellaneous

- [Dataset source](https://www.kaggle.com/datasets/henryshan/google-stock-price)

---

