# Twitter-Data-Analysis

**Sentiment Analysis of Twitter Data Using Machine Learning**

## 📄 Abstract

This project explores sentiment analysis on Twitter data using the [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140/data) dataset, which contains 1.6 million labeled tweets. The objective is to classify tweet sentiments as **positive**, **negative**, or **neutral** using various machine learning models:

- Logistic Regression  
- Random Forest  
- Long Short-Term Memory (LSTM)  
- BERT (Bidirectional Encoder Representations from Transformers)

BERT achieved the best performance with **82.77% accuracy**, balanced across precision, recall, and F1-score. Feature engineering further improved model accuracy, positioning BERT as the top choice for Twitter sentiment analysis.

---

## 📌 Table of Contents

- [Introduction](#-introduction)
- [Related Works](#-related-works)
- [Dataset](#-dataset)
- [Key Concepts](#-key-concepts)
- [Methodology](#-methodology)
- [Numerical Experiments](#-numerical-experiments)
- [Conclusion](#-conclusion)
- [References](#-references)

---

## 🧠 Introduction

In the age of social media, Twitter serves as a major platform for public expression. Analyzing sentiments in tweets helps:

- Understand public opinion  
- Forecast trends  
- Assist businesses, researchers, and policymakers

This project uses machine learning to classify sentiments in tweets from the Sentiment140 dataset.

---

## 🔍 Related Works

Some foundational studies in sentiment analysis:

- **Pang & Lee (2008)** – Applied SVM and Naïve Bayes for movie reviews [1]  
- **Sarlan & Nadam (2015)** – Used VADER scores and engineered features for tweets [3]  
- **Ramadhani & Goo (2020)** – Used LSTM and DCNN for sentiment classification [2]  
- **Vateekul & Koomsubha (2021)** – Applied LSTM/DCNN to Thai Twitter data [4]

---

## 📊 Dataset

- **Name**: Sentiment140  
- **Size**: 1.6 million tweets  
- **Classes**:  
  - `0` = Negative  
  - `2` = Neutral  
  - `4` = Positive  
- **Fields**: `target`, `ids`, `date`, `flag`, `user`, `text`

[View Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140/data)

---

## 🔑 Key Concepts

- **Sentiment Analysis**: Classifying emotions in text (positive/negative/neutral)
- **Machine Learning Models**: Logistic Regression, Random Forest, LSTM, BERT
- **Tokenization & Lemmatization**: Preprocessing steps to normalize text
- **TF-IDF**: Text feature extraction method
- **Word Embeddings**: GloVe, BERT for semantic representation of words

---

## 🛠️ Methodology

### 1. Data Collection  
- Source: Sentiment140 (Kaggle)

### 2. Data Preprocessing  
- Cleaning: Remove URLs, mentions, hashtags, symbols  
- Tokenization: Done using NLTK  
- Feature Extraction:  
  - Traditional Models: TF-IDF  
  - Deep Learning: GloVe, BERT  
  - Extra Features: VADER scores for Random Forest

### 3. Model Selection  
- **Logistic Regression**: Simple baseline  
- **Random Forest**: Ensemble-based, good for feature interaction  
- **LSTM**: Captures sequence/context in text  
- **BERT**: Pretrained transformer for deep contextual understanding

### 4. Training & Hyperparameter Tuning  
- **Logistic Regression**: Default parameters  
- **Random Forest**: Default + engineered features  
- **LSTM**: Tuned, trained for 10 epochs  
- **BERT**: Fine-tuned using Hugging Face Transformers  
  - Learning rate: `2e-5`  
  - Epochs: `3`

### 5. Evaluation Metrics  
- **Accuracy**, **Precision**, **Recall**, **F1-score**, **MAP**  
- Emphasis placed on **recall**

---

## 📈 Numerical Experiments

### ✅ **Overall Performance**

| Model              | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| **BERT**          | **82.77%** | 0.83      | 0.83   | 0.83     |
| Logistic Regression | 77.23%   | 0.77      | 0.77   | 0.77     |
| LSTM              | 76.72%   | ~0.51     | ~0.50  | ~0.50    |
| Random Forest     | 75.01%   | 0.75      | 0.75   | 0.75     |

---

## 🧾 Conclusion

- **BERT** is the best-performing model, with robust results across all metrics.
- **Traditional models** (Logistic Regression, Random Forest) still provide solid baselines.
- **Deep learning models** (LSTM, BERT) excel when fine-tuned and supported by embeddings.
- **Future work**: Improve preprocessing, further fine-tune BERT, and manage class imbalance.

---

## 📚 References

1. B. Pang and L. Lee. *Opinion mining and sentiment analysis*, 2008.  
2. A. M. Ramadhani and H. S. Goo. *Twitter sentiment analysis using deep learning methods*, 2017.  
3. A. Sarlan, C. Nadam, S. Basri. *Twitter sentiment analysis*, 2014.  
4. P. Vateekul and T. Koomsubha. *Sentiment analysis on Thai Twitter using deep learning*, 2016.
