import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import streamlit as st
import joblib
import warnings
warnings.filterwarnings('ignore')


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report

)

# Create Streamlit App

# Page Set Up
st.title("Heart Disease Prediction Models")
#st.title("Machine Learning Classification App")
st.write("Upload test data and select a model to see performance.")

#Load trained models
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
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))
   

    
    #st.text(report)

# # Combine X_test and y_test
# test_data = pd.DataFrame(X_test, columns=X.columns)
# test_data["target"] = y_test.values

# # Save as CSV
# test_data.to_csv("test_data_for_streamlit.csv", index=False)

