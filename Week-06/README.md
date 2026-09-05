# ₿ Bitcoin Market Trend Prediction Using Machine Learning

## Week 6 Capstone Project | NeuroFive Solutions Machine Learning Internship

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red)

---

## 📌 Project Overview

This project is an end-to-end Machine Learning application designed to predict the short-term direction of Bitcoin's market price.

Using historical Bitcoin market data and technical indicators, the system predicts whether Bitcoin's closing price is likely to move **UP 📈 or DOWN 📉 on the following day**.

The project covers the complete Machine Learning lifecycle, including data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, hyperparameter tuning, model selection, and deployment.

---

## 🎯 Problem Statement

Cryptocurrency markets are highly volatile and influenced by multiple factors. Predicting short-term market direction is challenging.

The objective of this project is to build a binary classification model capable of analyzing historical Bitcoin market data and technical indicators to predict:

- **UP (1):** Bitcoin closing price increases on the following day.
- **DOWN (0):** Bitcoin closing price decreases or remains unchanged on the following day.

> ⚠️ This project is created for educational and research purposes only and should not be considered financial or investment advice.

---

## 📊 Dataset

The project uses historical daily Bitcoin market data covering:

**September 2014 – September 2023**

The dataset contains the following original variables:

- Date
- Open
- High
- Low
- Close
- Adjusted Close
- Volume

---

## 🔄 Machine Learning Workflow

The project follows a complete end-to-end Machine Learning pipeline:

1. Data Collection
2. Data Understanding
3. Data Cleaning
4. Exploratory Data Analysis
5. Feature Engineering
6. Technical Indicator Creation
7. Target Variable Creation
8. Chronological Train-Test Split
9. Model Training
10. Model Evaluation
11. Hyperparameter Tuning
12. Final Model Selection
13. Feature Importance Analysis
14. Model Saving
15. Streamlit Deployment

---

## ⚙️ Feature Engineering

Several technical and market-based features were engineered from historical price data.

Examples include:

- Daily Return
- Price Range
- Volume Change
- SMA 7
- SMA 30
- EMA
- RSI
- MACD
- Volatility
- Additional trend-based features

These features were used as input variables for the Machine Learning models.

---

## 🤖 Machine Learning Models

The following classification models were trained and evaluated:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

The models were evaluated using chronological data splitting to respect the time-series nature of financial market data.

---

## 📈 Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### Initial Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.4632 | 0.4562 | 0.7296 | 0.5614 | 0.4893 |
| Random Forest | 0.4985 | 0.4814 | 0.8436 | 0.6130 | 0.5036 |
| XGBoost | **0.5046** | **0.4848** | 0.8339 | **0.6132** | 0.5233 |

---

## 🔧 Hyperparameter Tuning

The XGBoost model was selected for hyperparameter tuning using:

- RandomizedSearchCV
- TimeSeriesSplit Cross-Validation
- ROC-AUC optimization

### Best Parameters

```text
subsample: 0.8
n_estimators: 200
min_child_weight: 3
max_depth: 3
learning_rate: 0.01
colsample_bytree: 0.7