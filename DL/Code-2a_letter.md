# Practical-2a (Classification using Deep Neural Network - OCR Letter Recognition)

Problem Statement: Multiclass classification using Deep Neural Networks: Example: Use the OCR letter recognition dataset.

> [!NOTE]
> Dataset available in [Datasets](../Datasets/letter+recognition.zip) directory.

---

## Pre-requisities

1. Install packages using `pip`: `pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn` (`tensorflow` requires Python 3.9 - 3.12)
2. Download and unzip the `letter+recognition.zip` dataset in the same directory as the Jupyter notebook.

## Steps

1. Import Libraries
2. Load Dataset
3. Exploratory Data Analysis (EDA)
4. Visualize Class Distribution
5. Encode Labels and Separate Features
6. Split into Training and Testing Sets
7. Feature Scaling (Standardization)
8. One-Hot Encode Labels
9. Build the Deep Neural Network Model
10. Compile the Model
11. Train the Model
12. Evaluate the Model on Test Data
13. Plot Training vs Validation Accuracy
14. Plot Training vs Validation Loss
15. Confusion Matrix and Classification Report

---

## Code

### 1. Import Libraries:

```python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import to_categorical
```

### 2. Load Dataset:

```python3
# Dataset has no header row — define column names manually based on UCI documentation
col_names = ['letter', 'x-box', 'y-box', 'width', 'high', 'onpix',
             'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar',
             'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx']

data = pd.read_csv('./letter+recognition/letter-recognition.data', header=None, names=col_names)
print("Shape:", data.shape)
print(data.head())
```

### 3. Exploratory Data Analysis (EDA):

```python3
print("Data Types:\n", data.dtypes)
print("\nMissing Values:\n", data.isnull().sum())
print("\nStatistical Summary:\n", data.describe())
```

### 4. Visualize Class Distribution:

```python3
plt.figure(figsize=(14, 4))
data['letter'].value_counts().sort_index().plot(kind='bar')
plt.title("Number of Samples per Letter Class")
plt.xlabel("Letter")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
```

### 5. Encode Labels and Separate Features:

```python3
label_encoder = LabelEncoder()
data['letter'] = label_encoder.fit_transform(data['letter'])  # A=0, B=1, ..., Z=25

X = data.drop('letter', axis=1).values   # 16 numeric features
y = data['letter'].values                 # class index 0–25
num_classes = len(label_encoder.classes_)
print("Classes:", label_encoder.classes_)
print("Number of classes:", num_classes)
```

### 6. Split into Training and Testing Sets:

```python3
# 80% train, 20% test; stratify ensures balanced class distribution in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Train samples:", X_train.shape[0])
print("Test samples: ", X_test.shape[0])
```

### 7. Feature Scaling (Standardization):

```python3
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # learn mean/std from train, then scale
X_test  = scaler.transform(X_test)       # apply same mean/std to test (no leakage)
```

### 8. One-Hot Encode Labels:

```python3
# e.g. class 2 of 26 -> [0, 0, 1, 0, ..., 0]
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat  = to_categorical(y_test,  num_classes)
```

### 9. Build the Deep Neural Network Model:

```python3
model = Sequential()

model.add(Input(shape=(X_train.shape[1],)))    # input: 16 features
model.add(Dense(256, activation='relu'))        # hidden layer 1: 256 neurons
model.add(Dropout(0.3))                         # drop 30% neurons to reduce overfitting
model.add(Dense(128, activation='relu'))        # hidden layer 2: 128 neurons
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))         # hidden layer 3: 64 neurons
model.add(Dense(num_classes, activation='softmax'))  # output: probability for each of 26 letters

model.summary()
```

### 10. Compile the Model:

```python3
# categorical_crossentropy: standard loss for multi-class one-hot classification
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 11. Train the Model:

```python3
history = model.fit(
    X_train, y_train_cat,
    epochs=50,
    batch_size=32,
    validation_split=0.2   # use 20% of training data to monitor val loss each epoch
)
```

### 12. Evaluate the Model on Test Data:

```python3
loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"Test Loss:     {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")
```

### 13. Plot Training vs Validation Accuracy:

```python3
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
```

### 14. Plot Training vs Validation Loss:

```python3
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
```

### 15. Confusion Matrix and Classification Report:

```python3
y_pred = np.argmax(model.predict(X_test), axis=1)  # predicted class index

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
```

---

## Miscellaneous

- [Dataset source](https://archive.ics.uci.edu/ml/datasets/letter%2Brecognition)

---

