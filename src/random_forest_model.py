import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from base_model import BaseModel
from data_handler import DataHandler
from evaluator import ModelEvaluator


class RandomForestClassifierModel(BaseModel):
    """
    Random Forest model implementation following the BaseModel interface.
    Encapsulates training, prediction, and feature importance analysis.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = None, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced'
        )
        self.model_name = "Random Forest"
        self.feature_names = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Trains the model and stores feature names for analysis."""
        self.feature_names = X_train.columns
        self.model.fit(X_train, y_train)
        print(f"{self.model_name} trained successfully.")

    def predict(self, X: pd.DataFrame):
        """Returns binary class predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        """Returns class probabilities."""
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        """Calculates and prints the weight of decision factors (Gini importance)."""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print(f"\n--- {self.model_name} Top Feature Importances ---")
        for i in range(min(10, len(self.feature_names))):
            feature_name = self.feature_names[indices[i]]
            importance_score = importances[indices[i]]
            print(f"{i + 1}. {feature_name}: {importance_score:.4f}")


# ==========================================
# 1-3. Data Management Pipeline
# ==========================================
data_handler = DataHandler()
df = data_handler.load_data("../data/student_data_final.csv")

# Separating features and target
X, y = data_handler.prepare_features_target(df, target_col="is_churned")

# Train-test split
X_train, X_test, y_train, y_test = data_handler.split_data(X, y)

# ==========================================
# 4. Model Training and Evaluation
# ==========================================
rf_model = RandomForestClassifierModel(n_estimators=200)
rf_model.train(X_train, y_train)

# Generate and evaluate metrics
y_pred = rf_model.predict(X_test)
evaluator = ModelEvaluator()
evaluator.evaluate_predictions(y_test, y_pred, rf_model.model_name)

# Display feature ranking
rf_model.get_feature_importance()

# ==========================================
# 5. Testing on THREE Realistic Student Personas
# ==========================================

# Calculate training means to fill baseline values for test cases
train_means = X_train.mean().to_dict()


def create_test_student_df(updates: dict) -> pd.DataFrame:
    """Creates a single-row DataFrame based on training averages with specific overrides."""
    student_dict = {col: [train_means[col]] for col in X_train.columns}
    student_dict.update(updates)
    return pd.DataFrame(student_dict)


# Build profiles based on critical decision factors
test_cases = [
    ("AT-RISK (Freshman, struggling)", create_test_student_df({
        "total_active_terms": [1],
        "credits_done_term": [10.0],
        "gpa_term": [1.8],
        "term_success_rate": [0.33],
        "credit_loss_ratio": [0.67]
    })),
    ("AVERAGE (Sophomore, stable)", create_test_student_df({
        "total_active_terms": [3],
        "credits_done_term": [26.0],
        "gpa_term": [3.2],
        "term_success_rate": [0.86],
        "credit_loss_ratio": [0.14]
    })),
    ("EXCELLENT (Near graduation, exemplary)", create_test_student_df({
        "total_active_terms": [6],
        "credits_done_term": [30.0],
        "gpa_term": [4.8],
        "term_success_rate": [1.0],
        "credit_loss_ratio": [0.0]
    }))
]

print("\n" + "=" * 60)
print("FINE-TUNED RANDOM FOREST TESTING")
print("=" * 60)

for title, student_df in test_cases:
    prediction = rf_model.predict(student_df)[0]
    probabilities = rf_model.predict_proba(student_df)[0]

    print(f"\n>>> Profile: {title}")
    print(f"Prediction: {'Will Churn' if prediction == 1 else 'Will Stay'}")
    print(f"Retention Probability: {probabilities[0]:.2%}")
    print(f"Dropout Probability: {probabilities[1]:.2%}")