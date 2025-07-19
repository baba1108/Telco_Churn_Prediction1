# Telco Churn Prediction Dashboard using Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

data_raw = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
data_raw['TotalCharges'] = pd.to_numeric(data_raw['TotalCharges'], errors='coerce')
data_raw.dropna(inplace=True)
data_raw.drop(columns=['customerID'], inplace=True)
data_raw['Churn'] = data_raw['Churn'].map({'No': 0, 'Yes': 1})
data = pd.get_dummies(data_raw, drop_first=True)

X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

MODEL_PATH = "rf_model.pkl"
if not os.path.exists(MODEL_PATH):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_smote, y_train_smote)
    joblib.dump(model, MODEL_PATH)
else:
    model = joblib.load(MODEL_PATH)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")
st.title("📊 Telco Churn Prediction Dashboard")

st.header("1. Vue d'ensemble")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Clients", len(data))
with col2:
    churn_rate = data['Churn'].mean() * 100
    st.metric("% Churn", f"{churn_rate:.2f}%")
with col3:
    st.metric("Churners", data['Churn'].sum())

fig1, ax1 = plt.subplots()
data['Churn'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax1)
ax1.set_ylabel('')
ax1.set_title("Répartition Churn")
st.pyplot(fig1)

st.header("2. Profil des clients churners")
selected_feat = st.selectbox("Choisir une caractéristique:", ['Contract', 'PaymentMethod', 'InternetService', 'TechSupport'])
grouped = data_raw[[selected_feat, 'Churn']]
fig2, ax2 = plt.subplots()
sns.countplot(x=selected_feat, hue='Churn', data=grouped, ax=ax2)
plt.xticks(rotation=45)
st.pyplot(fig2)

st.header("3. Performance du Modèle")
cm = confusion_matrix(y_test, y_pred)
st.subheader("Matrice de confusion")
st.write(pd.DataFrame(cm, columns=['Prévu: Non', 'Prévu: Oui'], index=['Réel: Non', 'Réel: Oui']))
st.subheader("Rapport de classification")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
fig3, ax3 = plt.subplots()
ax3.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax3.plot([0, 1], [0, 1], 'k--')
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.legend()
st.pyplot(fig3)

st.header("4. Importance des variables")
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
st.bar_chart(importances.head(10))

st.header("5. Simulation client")
st.markdown("Saisir les données client pour prédire le risque de churn")
if st.button("Réinitialiser les champs"):
    st.experimental_rerun()

input_data = {}
for col in X.columns:
    if "_" in col and col.endswith("Yes"):
        input_data[col] = 1 if st.radio(f"{col}", ["Non", "Oui"]) == "Oui" else 0
    else:
        input_data[col] = st.number_input(f"{col}", value=float(data[col].mean()))

input_df = pd.DataFrame([input_data])
pred = model.predict(input_df)[0]
pred_prob = model.predict_proba(input_df)[0][1]

st.subheader("Résultat de prédiction")
st.write(f"Probabilité de churn : {pred_prob*100:.2f}%")
st.write(f"Prédiction : {'Churn' if pred == 1 else 'Pas de churn'}")

st.header("6. Exploration des clients churners")
with st.expander("Voir un échantillon de clients ayant churné"):
    churners = data_raw[data_raw['Churn'] == 1]
    st.dataframe(churners.sample(10))