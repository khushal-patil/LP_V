# Practical-2b (Classification using Deep Neural Network - IMDB Dataset)

Problem Statement: Binary classification using Deep Neural Networks Example: Classify movie reviews into positive" reviews and "negative" reviews, just based on the text content of the reviews. Use IMDB dataset

> [!NOTE]
> Dataset available in [Datasets](../Datasets/IMDB%20Dataset.csv) directory.

---

## Pre-requisities

1. Install packages using `pip`: `pip install tensorflow keras pandas numpy scikit-learn matplotlib seaborn` (`tensorflow` requires Python 3.9 - 3.12)
2. Copy the `IMDB Dataset.csv` dataset in the same directory as the Jupyter notebook.

## Steps

1. Import Libraries
2. Load Dataset
3. Exploratory Data Analysis (EDA)
4. Data Cleaning - Strip HTML Tags
5. Encode Labels and Separate Features
6. Tokenize and Pad Text Sequences
7. Split into Training and Testing Sets
8. Build the Neural Network Model
9. Compile the Model
10. Train the Model
11. Evaluate the Model on Test Data
12. Plot Training vs Validation Accuracy
13. Plot Training vs Validation Loss
14. Confusion Matrix and Classification Report

---

## Code

### 1. Import Libraries:

```python3
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

### 2. Load Dataset:

```python3
data = pd.read_csv('IMDB Dataset.csv')
print(data.head())
```

### 3. Exploratory Data Analysis (EDA):

```python3
print("Shape:", data.shape)
print("\nMissing Values:\n", data.isnull().sum())
print("\nClass Distribution:\n", data['sentiment'].value_counts())

# Visualize class distribution
sns.countplot(x='sentiment', data=data)
plt.title('Sentiment Class Distribution')
plt.show()

# Sample reviews
print("\nSample positive review:\n", data[data['sentiment'] == 'positive']['review'].iloc[0][:300])
print("\nSample negative review:\n", data[data['sentiment'] == 'negative']['review'].iloc[0][:300])
```

### 4. Data Cleaning - Strip HTML Tags:

```python3
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)   # remove HTML tags like <br />
    text = text.lower().strip()         # lowercase and trim whitespace
    return text

data['review'] = data['review'].apply(clean_text)
print("Sample cleaned review:\n", data['review'].iloc[0][:300])
```

### 5. Encode Labels and Separate Features:

```python3
label_encoder = LabelEncoder()
data['sentiment'] = label_encoder.fit_transform(data['sentiment'])  # positive=1, negative=0

X = data['review'].values    # input: review text
y = data['sentiment'].values # output: 0 or 1
```

### 6. Tokenize and Pad Text Sequences:

```python3
vocab_size = 10000   # keep only top 10,000 most frequent words
max_length = 200     # truncate/pad all reviews to 200 words

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')  # <OOV> handles unknown words
tokenizer.fit_on_texts(X)                        # build word index from training text

sequences = tokenizer.texts_to_sequences(X)      # convert each word to its integer index
padded_sequences = pad_sequences(sequences, maxlen=max_length,
                                 padding='post', truncating='post')  # pad/truncate to fixed length
```

### 7. Split into Training and Testing Sets:

```python3
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)
```

### 8. Build the Neural Network Model:

```python3
model = Sequential()
model.add(Embedding(vocab_size, 16))        # maps each word index to a 16-dim vector
model.add(GlobalAveragePooling1D())          # averages all word vectors into one vector
model.add(Dense(24, activation='relu'))      # hidden layer: 24 neurons
model.add(Dense(1, activation='sigmoid'))    # output: probability between 0 and 1 (binary)

model.summary()
```

### 9. Compile the Model:

```python3
# binary_crossentropy: standard loss for binary classification; sigmoid output
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### 10. Train the Model:

```python3
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
```

### 11. Evaluate the Model on Test Data:

```python3
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")
```

### 12. Plot Training vs Validation Accuracy:

```python3
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()
```

### 13. Plot Training vs Validation Loss:

```python3
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()
```

### 14. Confusion Matrix and Classification Report:

```python3
y_pred = (model.predict(X_test) > 0.5).astype(int)  # threshold 0.5: prob > 0.5 = positive

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
```

---

## Miscellaneous

- [Dataset source](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

