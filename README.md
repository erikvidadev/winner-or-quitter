# 🎓 Winner or Quitter: Student Churn Prediction

This project focuses on predicting student dropout (**churn**) by comparing various machine learning models. The goal is to identify risk factors and enable early detection of at-risk students based on academic performance data.

## 📋 Project Overview
The models were trained on a dataset containing over 63,000 records, including student performance metrics, demographic indicators, and academic progress.

**Target Variable Distribution:**
* **Active (0):** 50,034 records
* **Churned (1):** 13,199 records

---

## 🚀 Model Comparison (Metrics)

Model performance was measured using standard classification metrics on the test dataset (20% split):

| Metric | Logistic Regression | Random Forest | XGBoost |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 88.21% | **93.86%** | 93.80% |
| **Precision (for Churn)** | 0.68 | **0.92** | 0.88 |
| **Recall (for Churn)** | **0.83** | 0.77 | 0.81 |
| **F1-Score (for Churn)** | 0.75 | 0.84 | **0.85** |



### Key Findings:
* **Logistic Regression:** Boasts the highest **Recall** (0.83), meaning it effectively "notices" trouble, though it produces a relatively high number of false alarms (lower Precision).
* **Random Forest:** The most accurate model on test data, operating with extremely high precision.
* **XGBoost:** Offers the best balance (**F1-score: 0.85**). This model is most suitable for practical use as it effectively handles complex, non-linear relationships.

---

## 🔍 Decision Factors (Feature Importance)

We examined which variables most significantly influence the decision for each model:

1.  **XGBoost:** Primarily focuses on the **number of credits taken** (`credits_taken_term`) and the term success rate.
2.  **Random Forest:** Places high importance on the **number of active terms** (`total_active_terms`).
3.  **Logistic Regression:** Considers the **total number of missed credits** (`missed_credits_total`) as the most vital indicator.



---

## 🧪 Testing on Unique Profiles (Predict Proba)

The robustness of the models was tested on three realistic student profiles. The table shows the **probability of staying** (Probability of Active status %):

| Student Profile | Logistic Regression | Random Forest | XGBoost |
| :--- | :---: | :---: | :---: |
| **AT-RISK** (1st term, 10/30 credits) | 7.49% | 63.50% | **6.91%** |
| **AVERAGE** (3rd term, 26/30 credits) | 41.16% | 72.00% | **17.54%** |
| **EXCELLENT** (6th term, 4.8 GPA, 30/30 credits) | **87.48%** | 74.00% | 77.77% |

### Conclusions:
* **XGBoost is the most rigorous:** It predicts a 93%+ churn probability for the at-risk student, facilitating early intervention.
* **Random Forest is overly optimistic** toward struggling students (63% stay probability), which could be risky for a decision support system.
* **Logistic Regression** provides the widest range between the two extremes.

---

## 🛠️ Future Development Opportunities (Roadmap)

While the current version provides a stable foundation, several directions were identified to further improve prediction accuracy:

### 1. Data Quality and Sample Weighting
Since the outcome for **active** students is still uncertain (right-censored data), implementing **status-based weighting** is recommended:
* Assign higher weights to **Graduated** and **Dropped out** (closed) life cycles during training.
* Assign lower weights to **Active** students to reduce "noise" from incomplete data.

### 2. Time-series Approach
Instead of a static snapshot, the model could be extended to monitor **trends** in student performance (e.g., improving or declining GPA over consecutive terms).

### 3. Hyperparameter Tuning
Fine-tuning XGBoost and Random Forest models (using GridSearch or Optuna), specifically to optimize the balance between *Precision* and *Recall* for the "Churned" class.

### 4. Integration of External Factors
Incorporating socio-demographic data (e.g., distance from university, scholarship types, employment alongside studies), which can significantly influence churn intent but are currently missing from the technical dataset.

---
*Created as part of the Winner or Quitter project - 2026*