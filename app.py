import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import joblib

# Load the dataset
data = pd.read_csv('heart.csv')
print(data.shape)
print(data.head())

X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

# Preprocess the data (handle categorical variables)
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = "NA"

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1": f1_score(y_test, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

lr_metrics = evaluate_model(lr, X_test, y_test)
print("Logistic Regression:", lr_metrics)

joblib.dump(lr, "model/logistic_regression.pkl")

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_metrics = evaluate_model(dt, X_test, y_test)
print("Decision Tree:", dt_metrics)

joblib.dump(dt, "model/decision_tree.pkl")

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

knn_metrics = evaluate_model(knn, X_test, y_test)
print("KNN:", knn_metrics)

joblib.dump(knn, "model/knn.pkl")

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

nb_metrics = evaluate_model(nb, X_test, y_test)
print("Naive Bayes:", nb_metrics)

joblib.dump(nb, "model/naive_bayes.pkl")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_metrics = evaluate_model(rf, X_test, y_test)
print("Random Forest:", rf_metrics)

joblib.dump(rf, "model/random_forest.pkl")

# XGBoost
xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train, y_train)

xgb_metrics = evaluate_model(xgb, X_test, y_test)
print("XGBoost:", xgb_metrics)

joblib.dump(xgb, "model/xgboost.pkl")

#Create Comparison Table and Save it for Readme
results = pd.DataFrame([
    ["Logistic Regression", *lr_metrics.values()],
    ["Decision Tree", *dt_metrics.values()],
    ["KNN", *knn_metrics.values()],
    ["Naive Bayes", *nb_metrics.values()],
    ["Random Forest", *rf_metrics.values()],
    ["XGBoost", *xgb_metrics.values()],
], columns=["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"])

print(results)

# Create Streamlit App
st.title("Heart Disease Prediction Models")
#st.title("Machine Learning Classification App")
st.write("Upload test data and select a model to see performance.")

models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl")
}

# Dropdown to select model
model_name = st.selectbox(
    "Select a Machine Learning Model",
    list(models.keys())
)

selected_model = models[model_name]

# Upload test data
uploaded_file = st.file_uploader(
    "Upload CSV file (Test Data)",
    type=["csv"]
)

# Read uploaded file
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())

# Separate X and Y
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

# Preprocess the data (handle categorical variables)
    X = pd.get_dummies(X, drop_first=True)

# Make Predictions
    y_pred = selected_model.predict(X)

# Calculate Metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="weighted")
    recall = recall_score(y, y_pred, average="weighted")
    f1 = f1_score(y, y_pred, average="weighted")
    mcc = matthews_corrcoef(y, y_pred)

# Display Metrics
    st.subheader("Model Performance Metrics")

    st.write("Accuracy:", accuracy)
    st.write("Precision:", precision)
    st.write("Recall:", recall)
    st.write("F1 Score:", f1)
    st.write("MCC Score:", mcc)

# Show Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    st.write(cm)

# Show Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))
   

    
    #st.text(report)

# Combine X_test and y_test
test_data = pd.DataFrame(X_test, columns=X.columns)
test_data["target"] = y_test.values

# Save as CSV
test_data.to_csv("test_data_for_streamlit.csv", index=False)

