# Practical-3b (Convolutional Neural Network - MNIST Fashion Dataset)

Problem Statement: Convolutional Neural Network (CNN): Use MNIST Fashion Dataset and create a classifier to classify fashion clothing into categories.

> [!NOTE]
> Download dataset directly from [source](https://www.kaggle.com/datasets/zalando-research/fashionmnist).
> Dataset available in [Datasets](../Datasets/fashionmnist.zip) directory.
> In the code, dataset is downloaded directly from Keras/TensorFlow in 2nd step (Load Dataset)

---

## Pre-requisities

1. Install packages using `pip`: `pip install tensorflow keras numpy matplotlib seaborn scikit-learn` (`tensorflow` requires Python 3.9 - 3.12)

## Steps

---

## Code

### 1. Import Libraries:

```python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, AvgPool2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
```

### 2. Load Dataset:

```python3
# Fashion MNIST is built into Keras, downloads automatically on first run
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# --- Offline alternative (comment out tf.keras line above and use this instead) ---
# import pandas as pd
# train_df = pd.read_csv('fashion-mnist_train.csv')
# test_df  = pd.read_csv('fashion-mnist_test.csv')
# y_train = train_df['label'].values
# y_test  = test_df['label'].values
# X_train = train_df.drop('label', axis=1).values.reshape(-1, 28, 28)  # unflatten pixels to 28x28
# X_test  = test_df.drop('label', axis=1).values.reshape(-1, 28, 28)

print("Training set shape:", X_train.shape)   # (60000, 28, 28)
print("Test set shape:    ", X_test.shape)    # (10000, 28, 28)
print("Classes:", np.unique(y_train))
```

### 3. Exploratory Data Analysis (EDA):

```python3
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Class distribution
unique, counts = np.unique(y_train, return_counts=True)
plt.figure(figsize=(10, 4))
plt.bar([class_names[i] for i in unique], counts)
plt.xticks(rotation=45, ha='right')
plt.title("Training Set Class Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Sample images (one per class)
plt.figure(figsize=(15, 3))
for i, cls in enumerate(class_names):
    idx = np.where(y_train == i)[0][0]   # index of first image for this class
    plt.subplot(1, 10, i + 1)
    plt.imshow(X_train[idx], cmap='gray')
    plt.title(cls, fontsize=7)
    plt.axis('off')
plt.suptitle("Sample Image per Class")
plt.tight_layout()
plt.show()
```

### 4. Preprocess Data:

```python3
# Reshape to add channel dimension: (samples, 28, 28) -> (samples, 28, 28, 1)
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0  # normalize to [0,1]
X_test  = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# One-hot encode labels: e.g. class 3 of 10 -> [0,0,0,1,0,0,0,0,0,0]
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat  = to_categorical(y_test,  num_classes=10)

print("X_train shape:", X_train.shape)
print("y_train_cat shape:", y_train_cat.shape)
```

### 5. Build the CNN Model:

```python3
model = Sequential()

model.add(Input(shape=(28, 28, 1)))                                          # input: 28x28 grayscale image
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')) # 64 filters, extract features
model.add(AvgPool2D(pool_size=(2, 2)))                                        # downsample to 14x14

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')) # 32 filters, refine features
model.add(AvgPool2D(pool_size=(2, 2)))                                        # downsample to 7x7

model.add(GlobalAveragePooling2D())                                           # average each feature map to single value
model.add(Dense(10, activation='softmax'))                                    # output: probability for each of 10 classes

model.summary()
```

### 6. Compile the Model:

```python3
# categorical_crossentropy: standard loss for multi-class one-hot classification
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### 7. Train the Model:

```python3
# validation_data uses test set to monitor performance after each epoch
history = model.fit(X_train, y_train_cat, epochs=10, validation_data=(X_test, y_test_cat))
```

### 8. Evaluate the Model on Test Data:

```python3
loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"Test Loss:     {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")
```

### 9. Plot Training vs Validation Accuracy:

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

### 10. Plot Training vs Validation Loss:

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

### 11. Confusion Matrix and Classification Report:

```python3
y_pred = np.argmax(model.predict(X_test), axis=1)  # predicted class index

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=class_names))
```

### 12. Visualize Sample Predictions:

```python3
# batch predict all test images, then pick 10 random ones to display
all_preds = np.argmax(model.predict(X_test), axis=1)
random_indices = np.random.choice(len(X_test), 10, replace=False)

plt.figure(figsize=(20, 4))
for i, idx in enumerate(random_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    predicted = class_names[all_preds[idx]]
    actual    = class_names[y_test[idx]]
    color = 'green' if predicted == actual else 'red'   # green = correct, red = wrong
    plt.title(f"P: {predicted}\nA: {actual}", fontsize=8, color=color)
    plt.axis('off')
plt.suptitle("Sample Predictions (Green=Correct, Red=Wrong)")
plt.tight_layout()
plt.show()
```

---

## Miscellaneous

- [Dataset source](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

---

