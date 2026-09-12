import math
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Fraud Detection — GSCI Ensemble", layout="wide")

@st.cache_resource
def load_bundle():
    return joblib.load("fraud_gsci_bundle.joblib")

bundle = load_bundle()
trained_models = bundle["trained_models"]
mu = bundle["mu"]
model_names = bundle["model_names"]
feature_order = bundle["feature_order"]
amount_scaler = bundle["amount_scaler"]
time_scaler = bundle["time_scaler"]


def choquet_integral(sample_scores: dict, mu: dict) -> float:
    sorted_models = sorted(sample_scores.items(), key=lambda kv: kv[1])
    names_sorted = [m for m, _ in sorted_models]
    vals_sorted = [v for _, v in sorted_models]
    result, prev_val = 0.0, 0.0
    for i in range(len(vals_sorted)):
        coalition = frozenset(names_sorted[i:])
        result += (vals_sorted[i] - prev_val) * mu[coalition]
        prev_val = vals_sorted[i]
    return result


def gsci_predict_proba(X: pd.DataFrame) -> np.ndarray:
    base_probs = {m: trained_models[m].predict_proba(X)[:, 1] for m in model_names}
    out = np.zeros(len(X))
    for i in range(len(X)):
        out[i] = choquet_integral({m: base_probs[m][i] for m in model_names}, mu)
    return out


def shapley_value(model_name, mu):
    others = [m for m in model_names if m != model_name]
    n = len(model_names)
    total = 0.0
    for r in range(len(others) + 1):
        for subset in combinations(others, r):
            S, S_with = frozenset(subset), frozenset(subset) | {model_name}
            weight = (math.factorial(len(S)) * math.factorial(n - len(S) - 1)) / math.factorial(n)
            total += weight * (mu[S_with] - mu[S])
    return total


st.title("Credit Card Fraud Detection — GSCI Ensemble")
st.caption("Logistic Regression + Random Forest + XGBoost combined via a Fuzzy Measure / Choquet Integral, with SHAP explainability.")

with st.expander("Model contribution (Shapley values over the ensemble)"):
    sv = {m: shapley_value(m, mu) for m in model_names}
    st.bar_chart(pd.Series(sv, name="Shapley value"))

st.subheader("Upload transactions")
st.write("CSV with the original `creditcard.csv` schema: `Time`, `V1`...`V28`, `Amount` (and optionally `Class`).")
uploaded = st.file_uploader("Upload CSV", type="csv")

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    has_labels = "Class" in raw.columns
    df = raw.copy()
    df["Amount_scaled"] = amount_scaler.transform(df[["Amount"]])
    df["Time_scaled"] = time_scaler.transform(df[["Time"]])
    X = df.drop(columns=[c for c in ["Amount", "Time", "Class"] if c in df.columns])
    X = X[feature_order]

    probs = gsci_predict_proba(X)
    result = raw.copy()
    result["fraud_probability"] = probs
    result["flagged"] = probs >= 0.5

    st.subheader("Results")
    st.dataframe(result.sort_values("fraud_probability", ascending=False), use_container_width=True)
    st.write(f"Flagged {int(result['flagged'].sum())} of {len(result)} transactions as likely fraud.")

    if has_labels:
        from sklearn.metrics import roc_auc_score
        st.write(f"AUC on this file (labels present): {roc_auc_score(raw['Class'], probs):.4f}")

    st.subheader("Explain a transaction")
    row_idx = st.number_input("Row index to explain", min_value=0, max_value=len(X) - 1, value=0, step=1)
    explainer = shap.TreeExplainer(trained_models["XGBoost"])
    shap_values = explainer.shap_values(X.iloc[[row_idx]])
    st.write(f"XGBoost fraud probability for row {row_idx}: "
             f"{trained_models['XGBoost'].predict_proba(X.iloc[[row_idx]])[:, 1][0]:.4f}")
    contrib = pd.DataFrame({
        "feature": feature_order,
        "shap_value": shap_values[0],
    }).sort_values("shap_value", key=abs, ascending=False).head(10)
    st.bar_chart(contrib.set_index("feature"))
else:
    st.info("Upload a CSV to run fraud predictions. A sample slice of `creditcard.csv` works well for a demo.")
