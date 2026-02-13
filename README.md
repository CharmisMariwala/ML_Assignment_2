# Machine Learning Classification Assignment – 2

## 1. Problem Statement
The objective of this project is to implement, evaluate, and compare multiple machine learning classification models on a given dataset. The trained models are deployed using a Streamlit web application to demonstrate real-world usage by allowing users to upload test data and view model performance.

---

## 2. Dataset Description
The dataset used for this project is a heart disease classification dataset obtained from a public repository.

- Number of instances: 918
- Number of features: 11 input features
- Target variable: HeartDisease
- Type of problem: Binary Classification

The dataset is preprocessed by encoding categorical variables, splitting into training and testing sets, and applying feature scaling where required.

---

## 3. Machine Learning Models Used
The following six classification models were implemented and evaluated on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes Classifier  
5. Random Forest Classifier (Ensemble)  
6. XGBoost Classifier (Ensemble)

---

## 4. Model Performance Comparison

| Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-----------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8859 | 0.9297 | 0.8872 | 0.8859 | 0.8852 | 0.7694 |
| Decision Tree | 0.7880 | 0.7813 | 0.7880 | 0.7880 | 0.7868 | 0.5691 |
| KNN | 0.8859 | 0.9360 | 0.8859 | 0.8859 | 0.8856 | 0.7686 |
| Naive Bayes | **0.9130** | **0.9451** | **0.9134** | **0.9130** | **0.9131** | **0.8246** |
| Random Forest | 0.8696 | 0.9314 | 0.8694 | 0.8696 | 0.8694 | 0.7356 |
| XGBoost | 0.8587 | 0.9219 | 0.8587 | 0.8587 | 0.8587 | 0.7140 |

---

## 5. Observations and Analysis

| Model Name | Observation |
|-----------|-------------|
| Logistic Regression | Shows strong and stable performance with balanced precision, recall, and a high AUC score, indicating good generalization ability. |
| Decision Tree | Performs relatively weaker compared to other models and may suffer from overfitting due to its simple tree-based structure. |
| KNN | Achieves performance comparable to Logistic Regression, but its effectiveness depends heavily on feature scaling and the choice of k value. |
| Naive Bayes | Achieves the best overall performance with the highest accuracy, AUC, F1 score, and MCC, indicating strong predictive capability despite its simplicity. |
| Random Forest | Improves stability over Decision Tree by using ensemble learning, but does not outperform Naive Bayes on this dataset. |
| XGBoost | Delivers competitive performance with good accuracy and AUC, but slightly underperforms compared to Naive Bayes for this dataset. |

---

## 6. Streamlit Application
A Streamlit-based web application is developed that allows users to upload test data in CSV format, select a trained model from a dropdown, and view evaluation metrics along with a confusion matrix and classification report.

---

## 7. Deployment
The Streamlit application was deployed using Streamlit Community Cloud. The deployed app provides an interactive interface to evaluate the trained machine learning models.

---

## 8. Conclusion
Among all the implemented models, the Naive Bayes classifier achieved the best overall performance in terms of accuracy, AUC, F1 score, and MCC. This project successfully demonstrates an end-to-end machine learning workflow including model training, evaluation, deployment, and real-time inference using a web-based interface.
