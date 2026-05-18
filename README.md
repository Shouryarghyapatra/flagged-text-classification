# 🚩 NLP Text Classification – Flagged vs Not Flagged Detection

## 📌 Project Overview

This project is an end-to-end Natural Language Processing (NLP) classification pipeline designed to automatically detect whether a text response should be classified as:

* **Flagged (1)**
* **Not Flagged (0)**

The system preprocesses raw textual responses, converts them into numerical representations using TF-IDF vectorization, and trains machine learning models for automated moderation and text classification.

The project also integrates a production-style workflow structure with model saving and GitHub Actions automation support.

---

# 🎯 Problem Statement

Text moderation is an important challenge across:

* Social media platforms
* Public discussion forums
* Online communities
* Review systems
* Mental health support platforms

The objective of this project is to build a machine learning pipeline capable of automatically identifying whether a textual response should be flagged or not.

This is a **binary text classification problem**.

---

# 📂 Dataset Information

### Dataset Source

* Kaggle Dataset

### Features Used

| Column          | Description                              |
| --------------- | ---------------------------------------- |
| `response_text` | Raw textual response                     |
| `class`         | Target label (`flagged` / `not_flagged`) |

---

# ⚙️ Project Workflow

```text
Raw Text Data
      ↓
Text Cleaning
      ↓
Stopword Removal
      ↓
TF-IDF Vectorization
      ↓
Model Training
      ↓
Prediction
      ↓
Evaluation
      ↓
Model Saving
      ↓
GitHub Actions Automation
```

---

# 🛠 Tech Stack

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* NLTK
* GitHub Actions

---

# 🔍 Data Preprocessing

The following preprocessing steps were applied:

✔ Lowercase conversion
✔ Removal of non-alphabetic characters using Regular Expressions
✔ Tokenization
✔ Stopword removal using `ENGLISH_STOP_WORDS`
✔ Text normalization

Example:

```python
"I LOVE helping people!!!"
↓
"love helping people"
```

---

# 🧠 Feature Engineering

Used:

## TF-IDF Vectorization

Configuration:

```python
max_features = 2000
ngram_range = (1,2)
min_df = 2
```

The TF-IDF representation converts textual responses into machine-readable numerical vectors.

---

# 🤖 Machine Learning Models

The project experimented with multiple machine learning algorithms including:

* Gaussian Naive Bayes
* Logistic Regression
* Linear SVM (LinearSVC)

Final primary model:

## Logistic Regression

Configured with:

```python
class_weight='balanced'
max_iter=1000
```

---

# 📊 Model Performance

## Final Evaluation Metrics

### Accuracy

```text
0.75
```

### Classification Report

| Class           | Precision | Recall | F1-Score |
| --------------- | --------- | ------ | -------- |
| Not Flagged (0) | 0.80      | 0.86   | 0.83     |
| Flagged (1)     | 0.60      | 0.50   | 0.55     |

### Overall Accuracy

```text
75%
```

### Cross Validation Score

```text
F1 Macro Mean: 0.5767
```

---

# 💾 Model Persistence

The trained model and vectorizer are saved using `pickle`:

```python
models/model.pkl
models/vectorizer.pkl
```

This enables:

* Deployment
* Reusability
* Fast inference
* CI/CD integration

---

# 🚀 GitHub Actions Integration

This project supports GitHub Actions workflows for:

* Automated model training
* Automated evaluation
* Dependency installation
* Continuous Integration (CI)

Example workflow capabilities:

✔ Install dependencies
✔ Run training pipeline
✔ Execute evaluation scripts
✔ Validate workflow automatically on push

---

# 📁 Project Structure

```text
flagged-text-classification/
│
├── .github/
│   └── workflows/
│       └── nlp-pipeline.yaml
│
├── data/
│   └── train.csv
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ▶️ How to Run

## Clone Repository

```bash
git clone <your-repo-link>
cd flagged-text-classification
```

---

## Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Training Script

```bash
python src/train.py
```

---

# 📌 Future Improvements

* FastAPI deployment
* Docker containerization
* Streamlit UI
* Cloud deployment
* Deep Learning models
* Real-time moderation API

---

# 📬 Author

**Shouryarghya Patra**

Machine Learning | NLP | GitHub Actions | Python
