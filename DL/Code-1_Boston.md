# Practical-1 (Linear Regression using Deep Neural Network)

Problem Statement: Linear regression by using Deep Neural network: Implement Boston housing price prediction problem by Linear regression using Deep Neural network. Use Boston House price prediction dataset.

> [!NOTE]
> Dataset available in [Datasets](../Datasets/boston.csv) directory.

---

## Pre-requisities

1. Install packages using `pip`: `pip install tensorflow keras pandas numpy scikit-learn matplotlib seaborn` (`tensorflow` requires Python 3.9 - 3.12)
2. Copy the `boston.csv` dataset in the same directory as the Jupyter notebook.

## Steps

1. Import Libraries
2. Load Dataset
3. Exploratory Data Analysis (EDA)
4. Check for Missing Values
5. Correlation Heatmap
6. Separate Features and Target
7. Split into Training and Testing Sets
8. Feature Scaling (Standardization)
9. Build the Neural Network Model
10. Compile the Model
11. Train the Model
12. Evaluate the Model on Test Data
13. Make Predictions
14. Plot Training vs Validation Loss
15. Plot Predicted vs Actual Prices

---

## Code

### 1. Import Libraries:

```python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import Input
from keras.models import Sequential
from keras.layers import Dense
```

### 2. Load Dataset:

```python3
data = pd.read_csv('boston.csv')
print(data.head())
```

### 3. Exploratory Data Analysis (EDA):

```python3
print("Shape:", data.shape)          # number of rows and columns
print("\nData Types:\n", data.dtypes)
print("\nStatistical Summary:\n", data.describe())  # min, max, mean, std, etc.
```

### 4. Check for Missing Values:

```python3
print("Missing values per column:\n", data.isnull().sum())

# Drop rows with missing values (if any)
data = data.dropna()
print("\nShape after dropping nulls:", data.shape)
```

### 5. Correlation Heatmap:

```python3
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")  # show correlation between all feature pairs
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
```

### 6. Separate Features and Target:

```python3
X = data.drop('MEDV', axis=1)   # all columns except house price
y = data['MEDV']                 # target: median house price
```

### 7. Split into Training and Testing Sets:

```python3
# 80% train, 20% test; random_state=42 ensures reproducible split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 8. Feature Scaling (Standardization):

```python3
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)  # learn mean/std from train, then scale
X_test = scaler.transform(X_test)        # apply same mean/std to test (no leakage)
```

### 9. Build the Neural Network Model:

```python3
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))  # input shape = number of features
model.add(Dense(64, activation='relu'))       # hidden layer 1: 64 neurons
model.add(Dense(32, activation='relu'))       # hidden layer 2: 32 neurons
model.add(Dense(1, activation='linear'))      # output layer: single value (house price)

model.summary()
```

### 10. Compile the Model:

```python3
# adam: adaptive optimizer; mse: standard regression loss; mae: human-readable error metric
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
```

### 11. Train the Model:

```python3
# validation_split=0.2 reserves 20% of training data to monitor val loss each epoch
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
```

### 12. Evaluate the Model on Test Data:

```python3
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {loss:.4f}")
print(f"Test Mean Absolute Error: {mae:.4f}")
```

### 13. Make Predictions:

```python3
predictions = model.predict(X_test)
print("First 5 Predicted Prices:", predictions[:5].flatten())
print("First 5 Actual Prices:   ", y_test.values[:5])
```

### 14. Plot Training vs Validation Loss:

```python3
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()
```

### 15. Plot Predicted vs Actual Prices:

```python3
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.7)                        # each point = one test sample
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')   # diagonal = perfect prediction
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Miscellaneous

- [Dataset source](https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data)

---

