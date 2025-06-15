import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Fraud Detection App", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")  # Replace with your actual file path
    return df

df = load_data()

# --------------------------------------------
# Introduction
# --------------------------------------------
st.title("💳 Fraud Detection Using Machine Learning")
st.markdown("""
Welcome to the **Fraud Detection Project**. This app showcases how machine learning can help detect fraudulent credit card transactions.

### 🔍 Project Goals:
- Explore and understand transaction data
- Build a model to predict fraud
- Visualize performance
""")

# --------------------------------------------
# EDA Section
# --------------------------------------------
st.header("📊 Exploratory Data Analysis")
if st.checkbox("Show raw data"):
    st.dataframe(df.head(100))

st.subheader("Class Distribution")
class_counts = df['Class'].value_counts()
st.bar_chart(class_counts)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='coolwarm', ax=ax)
st.pyplot(fig)

# --------------------------------------------
# Model Training and Prediction
# --------------------------------------------
st.header("🤖 Model: XGBoost Classifier")

# Data prep
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
st.subheader("Model Performance")
st.write("**ROC AUC Score:**", round(roc_auc_score(y_test, y_pred), 3))

cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_title('Confusion Matrix')
st.pyplot(fig_cm)

# --------------------------------------------
# Conclusion
# --------------------------------------------
st.header("📌 Conclusion")
st.markdown("""
- Fraud detection is a highly imbalanced classification problem.
- XGBoost performed well on this dataset with decent ROC AUC.
- Further improvements can be made using SMOTE, anomaly detection, or advanced ensembles.
""")
