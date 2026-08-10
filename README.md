# Telco Customer Churn Prediction - ML Assignment 2

## a. Problem Statement
The goal of this assignment is to implement, evaluate, and deploy multiple machine learning classification models to predict telecommunications customer churn (whether a customer will discontinue their subscription: 'Yes' or 'No'). Identifying at-risk customers enables proactively targeted retention offers.

## b. Dataset Description
- **Dataset Source:** Telco Customer Churn Dataset (Kaggle / IBM / UCI)
- **Number of Instances:** 7,043 customer records
- **Number of Features:** 19 input features (e.g., tenure, MonthlyCharges, TotalCharges, Contract type, PaymentMethod, InternetService, OnlineSecurity, TechSupport)
- **Target Variable:** `target` (Binary: 1 = Churned, 0 = Retained)

## c. GitHub Repository Link
[https://github.com/2025ac05627/AP-Machine-Learning-Assignment-2/tree/main](https://github.com/2025ac05627/AP-Machine-Learning-Assignment-2/tree/main)

## d. Models Used & Performance Evaluation

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.8070** | **0.8416** | 0.6584 | 0.5668 | **0.6092** | **0.4843** |
| **Decision Tree** | 0.7942 | 0.8284 | 0.6296 | 0.5455 | 0.5845 | 0.4507 |
| **kNN** | 0.7601 | 0.7835 | 0.5511 | 0.5187 | 0.5344 | 0.3734 |
| **Naive Bayes** | 0.6558 | 0.8096 | 0.4269 | **0.8663** | 0.5719 | 0.3951 |
| **Random Forest (Ensemble)** | 0.8041 | 0.8401 | **0.6713** | 0.5134 | 0.5818 | 0.4639 |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Among the all this delivers overall best accuracy (80.70%), AUC (0.8416), F1 (0.6092), and MCC (0.4843). |
| **Decision Tree** | This Model provides 79.42% accuracy and an AUC of 0.8284, providing a clean, interpretable decision hierarchy. |
| **kNN** | Trails behind the other classifiers with lower AUC (0.7835) and MCC (0.3734), struggling slightly with high-dimensional feature spaces. |
| **Naive Bayes** | Catches the highest percentage of actual churners with an impressive Recall of **86.63%**, though it incurs extra false alarms, leading to a lower Precision (42.69%) and Accuracy (65.58%). |
| **Random Forest (Ensemble)** | Achieves the highest Precision (**67.13%**) and matches linear performance closely with 80.41% accuracy and 0.8401 AUC, effectively minimizing false alarms. |
| **Overall Winner** | **Logistic Regression** is the overall top performer on this dataset due to its leading Accuracy, AUC, F1, and MCC scores. |

---

### Note on Performance Metrics & Class Imbalance

In Telco Customer Churn dataset we have a class imbalance of **~73.5% vs 26.5%**, there is an inherent precision-recall trade-off. While individual metrics like Accuracy (~80%+) and ROC-AUC (~0.84) are high across top models, other scores like Precision, Recall, F1 (~0.57–0.61), and MCC (~0.45–0.48) reflect the difficulty of predicting the minority positive class (churners) without over-generating false positives. 
