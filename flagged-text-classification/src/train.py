import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import re
import os
import pickle

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC

# Load dataset
df = pd.read_csv("data/train.csv")

print("Dataset Loaded Successfully")
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Keep only required columns
data = df[['class', 'response_text']]

# Convert labels to numeric
data['class'] = data['class'].map({
    'not_flagged': 0,
    'flagged': 1
})

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text))
    text = text.lower()

    words = text.split()

    words = [
        w for w in words
        if w not in ENGLISH_STOP_WORDS
    ]

    return " ".join(words)

# Apply preprocessing
description_list = []

for text in data['response_text']:
    cleaned_text = preprocess_text(text)
    description_list.append(cleaned_text)

print("\nText preprocessing completed.")

# Features and labels
X = description_list
y = data['class']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split Completed")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),
    min_df=2
)

x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

print("\nTF-IDF Vectorization Completed")

# Logistic Regression Model
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

# Train model
model.fit(x_train_vec, y_train)

print("\nModel Training Completed")

# Predictions
y_pred = model.predict(x_test_vec)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Cross Validation
scores = cross_val_score(
    model,
    vectorizer.transform(X),
    y,
    cv=5,
    scoring='f1_macro'
)

print("\nCross Validation F1 Macro Mean:", scores.mean())

# Save model and vectorizer
os.makedirs("models", exist_ok=True)

pickle.dump(
    model,
    open("models/model.pkl", "wb")
)

pickle.dump(
    vectorizer,
    open("models/vectorizer.pkl", "wb")
)

print("\nModel and Vectorizer Saved Successfully")

# Example prediction
new_text = "I do care about anyone's problems and help anyone"

clean_text = preprocess_text(new_text)

X_new = vectorizer.transform([clean_text])

prediction = model.predict(X_new)

label_map = {
    0: "Supportive / empathetic response",
    1: "Non-supportive / other response"
}

print("\nExample Prediction:")
print(label_map[prediction[0]])
