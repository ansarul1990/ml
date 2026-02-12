# 📌 Machine Learning Assignment 2  
**Course:** Machine Learning  
**Program:** M.Tech (AIML / DSE)  
**Student Name:** Ansarul Islam Laskar  
**BITS ID:** 2025AA05568

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

| ML Model Name | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
|----------|----------|--------|-----------|--------|----------|--------|
| Logistic Regression | 0.8684   | 0.9832 | 0.8697    | 0.8684 | 0.8661   | 0.8473 |
| Decision Tree | 0.9187   | 0.9515 | 0.9223    | 0.9187 | 0.9194   | 0.9054 |
| kNN | 0.8349   | 0.9713 | 0.8326    | 0.8349 | 0.828    | 0.8092 |
| Naive Bayes | 0.6196   | 0.9089 | 0.6387    | 0.6196 | 0.599    | 0.5672 |
| Random Forest | 0.9545   | 0.9947 | 0.9566    | 0.9545 | 0.955    | 0.9471 |
| XGBoost  | 0.9593   | 0.9963 | 0.9607    | 0.9593 | 0.9596   | 0.9526 |


---

# 🔍 Observations on Model Performance

| ML Model Name | Observation |
|---------------|------------|
| Logistic Regression | Performed reasonably well with 86.84% accuracy and strong AUC (0.9832). This indicates that the dataset has partial linear separability and logistic regression can capture general trends effectively. |
| Decision Tree | Achieved good performance (91.87% accuracy). While it performs better than traditional linear models, it is slightly less effective than ensemble methods due to possible overfitting or lack of boosting. |
| kNN | Showed moderate performance (83.49% accuracy). The model is sensitive to feature scaling and the choice of K value, which may impact overall performance. |
| Naive Bayes | Recorded the lowest performance (61.96% accuracy). The strong feature independence assumption of Naive Bayes likely does not hold for this dataset, resulting in weaker predictive ability. |
| Random Forest (Ensemble) | Delivered strong performance (95.45% accuracy, 0.9947 AUC). The ensemble approach reduces overfitting and improves generalization compared to a single decision tree. |
| XGBoost (Ensemble) | Achieved the best overall performance (95.93% accuracy, 0.9963 AUC, highest MCC of 0.9526). The boosting technique effectively captured complex patterns in the dataset and provided superior predictive power. |


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

---

# 🎯 Conclusion

This project successfully implemented and compared six classification algorithms on a multi-class dataset.

The results clearly demonstrate that ensemble methods outperform individual models. XGBoost achieved the highest overall performance across all evaluation metrics, followed closely by Random Forest.

Traditional models such as Logistic Regression and Decision Tree performed reasonably well, while Naive Bayes showed comparatively weaker performance due to its independence assumption.

Overall, ensemble learning techniques demonstrated superior generalization ability, robustness, and predictive accuracy for this dataset.
