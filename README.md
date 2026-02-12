# 📌 Machine Learning Assignment 2  
**Course:** Machine Learning  
**Program:** M.Tech (AIML / DSE)  
**Student Name:** Ansarul Islam Laskar  

---

# 🧠 Multi-Class Classification Model Comparison & Deployment

## 🔹 a. Problem Statement

The objective of this project is to implement and compare multiple machine learning classification models on a selected multi-class dataset.  

This project demonstrates:
- Implementation of 6 classification algorithms
- Model evaluation using multiple performance metrics
- Comparative analysis of models
- Deployment of an interactive web application using Streamlit
- End-to-end ML workflow (training → evaluation → deployment)

---

## 🔹 b. Dataset Description

- **Dataset Type:** Multi-class Classification  
- **Source:** Kaggle (https://www.kaggle.com/datasets/ikjotsingh221/obesity-risk-prediction-cleaned)
- **Number of Instances:** 2086  
- **Number of Features:** 14  
- **Target Variable:** Class labels (0–6)  

### Dataset Characteristics:
- Data cleaned and preprocessed
- Train-test split applied
- Feature scaling performed where required
- Suitable for multi-class classification problems

---

## 🔹 c. Models Used

The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Method)  
6. XGBoost (Ensemble Method)  

All models were evaluated using the same dataset split to ensure fair comparison.

---

# 📊 Evaluation Metrics

Each model was evaluated using:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

# 📈 Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|------|-----------|--------|----------|------|
| Logistic Regression | XX | XX | XX | XX | XX | XX |
| Decision Tree | XX | XX | XX | XX | XX | XX |
| kNN | XX | XX | XX | XX | XX | XX |
| Naive Bayes | XX | XX | XX | XX | XX | XX |
| Random Forest (Ensemble) | XX | XX | XX | XX | XX | XX |
| XGBoost (Ensemble) | XX | XX | XX | XX | XX | XX |

> Replace "XX" with your actual evaluation results.

---

# 🔍 Observations on Model Performance

| ML Model Name | Observation |
|---------------|------------|
| Logistic Regression | Performs well for linearly separable data but may struggle with complex patterns. |
| Decision Tree | Easy to interpret but prone to overfitting if not tuned properly. |
| kNN | Sensitive to scaling and value of K. Performs well on structured data. |
| Naive Bayes | Fast and efficient but assumes feature independence. |
| Random Forest | Reduces overfitting and improves performance through ensemble learning. |
| XGBoost | Typically provides strong performance due to boosting and better generalization. |

---

# 🌐 Streamlit Web Application

The project includes a deployed Streamlit web application.

### 🔹 App Features:
- CSV Dataset Upload (Test Data)
- Model Selection Dropdown
- Display of Evaluation Metrics
- Confusion Matrix Visualization
- Classification Report Display

🔗 **Live App Link:**  
https://pmpoqfetzu4n3uqngbnmny.streamlit.app/

---

# 📂 Project Structure

```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
    │-- logistic_regression.pkl
    │-- decision_tree.pkl
    │-- knn.pkl
    │-- naive_bayes.pkl
    │-- random_forest.pkl
    │-- xgboost.pkl
```

---

# ⚙️ Requirements

```
streamlit
scikit-learn
numpy
pandas
matplotlib
seaborn
xgboost
joblib
```

---

# 🚀 Deployment

The application is deployed using:
- GitHub Repository
- Streamlit Community Cloud
- Main branch deployment
- app.py as the entry file

---

# 🎯 Conclusion

This project successfully demonstrates the implementation and comparison of six classification models on a multi-class dataset.  

Ensemble models such as Random Forest and XGBoost generally provide better performance due to their ability to reduce variance and capture complex patterns.

The deployed Streamlit application provides an interactive platform to test and compare models in real time.
