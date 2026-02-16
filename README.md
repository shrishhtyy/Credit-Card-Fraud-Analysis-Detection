
# Credit Card Fraud Analysis & Detection

A machine learning project that detects fraudulent credit card transactions from a highly imbalanced dataset using Logistic Regression

## Problem

Credit card fraud is rare. Only **0.17%** of transactions in this dataset are fraudulent. Standard models would simply predict "not fraud" for everything and still get 99.8% accuracy. This project tackles the class imbalance problem.

## Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Features:** 30 (V1 to V28 are PCA-transformed, plus Time and Amount)
- **Target:** Class (0 = Legitimate, 1 = Fraud)
- **Fraud rate:** 492 out of 284,807 transactions (0.17%)

## Approach

1. **Data exploration** | Analysed distribution of legitimate vs fraudulent transactions, compared statistical measures (mean Amount for fraud vs legit)
2. **Under-sampling** | Sampled 492 legitimate transactions to match the 492 fraud cases, creating a balanced dataset of 984 transactions
3. **Train/test split** | 80/20 stratified split to maintain class balance
4. **Model training** | Logistic Regression on the balanced dataset
5. **Evaluation** | Accuracy on both training and test data

## Results

| Metric | Score |
|--------|-------|
| Training Accuracy | ~94% |
| Test Accuracy | ~92% |

## Tech Stack

Python, NumPy, Pandas, scikit-learn, Logistic Regression

## How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Open the notebook in Google Colab
3. Update the file path to your dataset location
4. Run all cells

