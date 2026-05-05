# Practical-3a (Convolutional Neural Network - Plant Diseases)

Problem Statement: Convolutional Neural Network (CNN): Use any dataset of plant disease and design a plant disease detection system using CNN.

> [!NOTE]
> Download dataset directly from [source](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data).
> Haven't added it to the `/Datasets` directory due to its large size.
> tbh the dataset doesn't really matter in this case, you just need to ensure dataset directory contains `train` and `valid` sub-directories.
> Refer the above dataset to understand the required directory structure.

---

## Pre-requisities

1. Install packages using `pip`: `pip install tensorflow keras numpy opencv-python matplotlib seaborn scikit-learn` (`tensorflow` requires Python 3.9 - 3.12)
2. Download and unzip the dataset in the same directory as the Jupyter notebook.
3. Ensure your unzipped dataset has the required directory structure:

```shell
New Plant Diseases Dataset(Augmented)/
├── train
│   ├── Apple___Apple_scab
│   ├── Apple___Black_rot
│   ├── Apple___Cedar_apple_rust
├── valid
│   ├── Apple___Apple_scab
│   ├── Apple___Black_rot
│   ├── Apple___Cedar_apple_rust
```

## Steps

1. Import Libraries
2. Load Dataset
3. Exploratory Data Analysis (EDA)
4. Split into Training and Testing Sets
5. Build the CNN Model
6. Compile the Model
7. Train the Model
8. Evaluate the Model on Test Data
9. Plot Training vs Validation Accuracy
10. Plot Training vs Validation Loss
11. Confusion Matrix and Classification Report

---

## Code

### 1. Import Libraries:

```python3
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
```

### 2. Load Dataset:

```python3
data = []
labels = []

# Path to dataset folder containing one subfolder per disease class
path = './Datasets/plant/train/'
categories = sorted(os.listdir(path))  # sort for consistent label ordering

# Map each category name to a numeric index
label_dict = {category: idx for idx, category in enumerate(categories)}
print("Classes found:", len(categories))

max_per_class = 200  # cap images per class to avoid RAM overflow on large datasets

for category in categories:
    folder = os.path.join(path, category)
    count = 0
    for img_name in os.listdir(folder):
        if count >= max_per_class:
            break
        img_path = os.path.join(folder, img_name)
        img_array = cv2.imread(img_path)
        if img_array is not None:                          # skip unreadable files
            img_array = cv2.resize(img_array, (64, 64))   # resize to fixed 64x64 pixels
            data.append(img_array)
            labels.append(label_dict[category])
            count += 1

data = np.array(data) / 255.0   # normalize pixel values from [0,255] to [0,1]
labels = np.array(labels)
print("Dataset shape:", data.shape)
print("Labels shape:", labels.shape)
```

### 3. Exploratory Data Analysis (EDA):

```python3
print("Total images:", len(data))
print("Image shape:", data[0].shape)
print("Number of classes:", len(categories))

# Class distribution bar chart
class_counts = {cat: int((labels == idx).sum()) for cat, idx in label_dict.items()}
plt.figure(figsize=(14, 5))
plt.bar(class_counts.keys(), class_counts.values())
plt.xticks(rotation=90)
plt.title("Number of Images per Disease Class")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Sample images from first 5 classes
plt.figure(figsize=(15, 3))
for i, category in enumerate(categories[:5]):
    idx = np.where(labels == label_dict[category])[0][0]  # index of first image in class
    plt.subplot(1, 5, i + 1)
    plt.imshow(cv2.cvtColor((data[idx] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title(category[:15], fontsize=8)
    plt.axis('off')
plt.suptitle("Sample Images per Class")
plt.show()
```

### 4. Split into Training and Testing Sets:

```python3
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

num_classes = len(categories)
# One-hot encode labels: e.g. class 2 of 5 → [0, 0, 1, 0, 0]
y_train = to_categorical(y_train, num_classes)
y_test  = to_categorical(y_test,  num_classes)
print("Train samples:", X_train.shape[0])
print("Test samples: ", X_test.shape[0])
```

### 5. Build the CNN Model:

```python3
model = Sequential()

model.add(Input(shape=(64, 64, 3)))               # input: 64x64 RGB image
model.add(Conv2D(32, (3, 3), activation='relu'))  # 32 filters, detect basic features
model.add(MaxPooling2D(2, 2))                      # downsample by 2x

model.add(Conv2D(64, (3, 3), activation='relu'))  # 64 filters, detect complex features
model.add(MaxPooling2D(2, 2))

model.add(Flatten())                              # convert 2D feature maps to 1D vector

model.add(Dense(128, activation='relu'))          # fully connected layer
model.add(Dropout(0.5))                           # randomly drop 50% neurons to reduce overfitting

model.add(Dense(num_classes, activation='softmax'))  # output: probability for each class

model.summary()
```

### 6. Compile the Model:

```python3
# categorical_crossentropy: standard loss for multi-class classification with one-hot labels
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 7. Train the Model:

```python3
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
```

### 8. Evaluate the Model on Test Data:

```python3
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")
```

### 9. Plot Training vs Validation Accuracy:

```python3
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Model Accuracy Over Epochs')
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
plt.title('CNN Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
```

### 11. Confusion Matrix and Classification Report:

```python3
y_pred = np.argmax(model.predict(X_test), axis=1)  # predicted class index
y_true = np.argmax(y_test, axis=1)                  # actual class index (from one-hot)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=categories, yticklabels=categories)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=categories))
```

---

## Miscellaneous

- [Dataset source](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

---

