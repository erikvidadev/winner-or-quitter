import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from base_model import BaseModel
from data_handler import DataHandler
from evaluator import ModelEvaluator


class LogisticRegressionClassifier(BaseModel):
    """
    Logistic Regression model with built-in scaling and coefficient analysis.
    Follows the Single Responsibility Principle by encapsulating scaling and training.
    """

    def __init__(self, max_iter: int = 1000, random_state: int = 42):
        self.model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.model_name = "Logistic Regression"
        self.feature_names = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Standardizes features and trains the logistic regression model."""
        self.feature_names = X_train.columns
        # Scaling is essential for Logistic Regression convergence and interpretation
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        print(f"{self.model_name} trained successfully.")

    def predict(self, X: pd.DataFrame):
        """Predicts class labels for the provided features."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame):
        """Predicts class probabilities for the provided features."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self):
        """Displays and analyzes model coefficients (weights)."""
        # In Logistic Regression, the magnitude of coefficients indicates feature importance
        importances = self.model.coef_[0]
        indices = np.argsort(np.abs(importances))[::-1]

        print(f"\n--- {self.model_name} Feature Importance (Coefficients) ---")
        for i in range(min(10, len(self.feature_names))):
            feat = self.feature_names[indices[i]]
            coeff = importances[indices[i]]
            # Positive weight = increases dropout probability, Negative = decreases it
            print(f"{i + 1}. {feat}: {coeff:.4f}")


# ==========================================
# 1-3. Data Handling Pipeline
# ==========================================
data_handler = DataHandler()
df = data_handler.load_data("../data/student_data_final.csv")

X, y = data_handler.prepare_features_target(df, target_col="is_churned")
X_train, X_test, y_train, y_test = data_handler.split_data(X, y)

# ==========================================
# 4. Model Training and Evaluation
# ==========================================
logreg_model = LogisticRegressionClassifier()
logreg_model.train(X_train, y_train)

y_pred = logreg_model.predict(X_test)
evaluator = ModelEvaluator()
evaluator.evaluate_predictions(y_test, y_pred, logreg_model.model_name)

# Display ranked feature weights
logreg_model.get_feature_importance()

# ==========================================
# 5. Testing on THREE specific Student Personas
# ==========================================
# Get baseline means for feature padding
train_means = X_train.mean().to_dict()


def create_test_student_df(updates):
    """Helper function to create a single-row DataFrame for testing."""
    student_dict = {col: [train_means[col]] for col in X_train.columns}
    student_dict.update(updates)
    return pd.DataFrame(student_dict)


# Define test scenarios
test_cases = [
    ("AT-RISK (Freshman, struggling)", create_test_student_df({
        "total_active_terms": [1],
        "credits_done_term": [10.0],
        "gpa_term": [1.8],
        "term_success_rate": [0.33],
        "credit_loss_ratio": [0.67],
        "is_first_year": [1]
    })),
    ("AVERAGE (Sophomore, stable)", create_test_student_df({
        "total_active_terms": [3],
        "credits_done_term": [26.0],
        "gpa_term": [3.2],
        "term_success_rate": [0.86],
        "credit_loss_ratio": [0.14],
        "is_first_year": [0]
    })),
    ("EXCELLENT (Near graduation, exemplary)", create_test_student_df({
        "total_active_terms": [6],
        "credits_done_term": [30.0],
        "gpa_term": [4.8],
        "term_success_rate": [1.0],
        "credit_loss_ratio": [0.0],
        "is_first_year": [0]
    }))
]

print("\n" + "=" * 60)
print("FINE-TUNED LOGISTIC REGRESSION MODEL TESTING")
print("=" * 60)

for title, student_df in test_cases:
    prediction = logreg_model.predict(student_df)[0]
    probabilities = logreg_model.predict_proba(student_df)[0]

    print(f"\n>>> Profile: {title}")
    print(f"Prediction: {'Will Churn' if prediction == 1 else 'Will Stay'}")
    print(f"Retention Probability: {probabilities[0]:.2%}")
    print(f"Dropout Probability: {probabilities[1]:.2%}")