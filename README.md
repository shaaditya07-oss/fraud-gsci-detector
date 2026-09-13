# Fraud Detection using GSCI Ensemble (Generalized Shapley Choquet Integral)

A credit card fraud detection system that combines Logistic Regression, Random
Forest, and XGBoost using a Fuzzy Measure and Choquet Integral ensemble
technique — with full explainability via Shapley values (model-level) and
SHAP (feature-level).

## Live Demo
https://fraud-gsci-detector-8znszyq9nd9rahpbghxiqp.streamlit.app/

## Problem
Credit card fraud detection is a highly imbalanced classification problem
(~0.17% fraud in the dataset). Simple ensembling (averaging/voting) treats
all models as equally reliable and independent, which ignores how models
interact and complement each other.

## Approach
1. **Preprocessing** — Separate scaling for `Amount` and `Time`; `V1`–`V28`
   are already PCA-transformed.
2. **SMOTE** — Synthetic oversampling of the minority (fraud) class, applied
   only to training data.
3. **Base Models** — Logistic Regression, Random Forest, and XGBoost trained
   independently.
4. **Fuzzy Measure (μ)** — Learned from validation performance of every model
   coalition (single models and pairs), capturing synergy between models
   rather than assuming independence.
5. **Choquet Integral** — Combines the three models' fraud probabilities
   using the fuzzy measure, instead of a simple average.
6. **Shapley Value Analysis** — Quantifies each base model's fair contribution
   to the ensemble's performance.
7. **SHAP Explainability** — Explains individual fraud predictions at the
   feature level (which transaction attributes drove the flag).

## Results

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.9729 | 0.4242 | 0.9106 | 0.5788 |
| Random Forest | 0.9785 | 0.8992 | 0.8699 | 0.8843 |
| XGBoost | 0.9840 | 0.9153 | 0.8780 | 0.8963 |
| **GSCI Ensemble** | *combines all three via Choquet Integral* | | | |

## Tech Stack
- Python, scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- SHAP for explainability
- Streamlit for deployment

## Run Locally
\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`
Upload a CSV with the `creditcard.csv` schema (`Time`, `V1`...`V28`, `Amount`,
optional `Class`) to get fraud predictions and explanations.

## Dataset
[mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) on Kaggle.

## Author
Aditya — B.Tech CSE (Data Science), GNIOT
