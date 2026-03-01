import pandas as pd
import numpy as np
import xgboost as xgb

from base_model import BaseModel
from data_handler import DataHandler
from evaluator import ModelEvaluator


class XGBoostClassifierModel(BaseModel):
    """
    XGBoost model implementation following the BaseModel interface.
    Utilizes gradient boosting for high-performance classification.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, random_state: int = 42):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric='logloss'
        )
        self.model_name = "XGBoost"
        self.feature_names = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Trains the XGBoost model and captures feature names for importance analysis."""
        self.feature_names = X_train.columns
        self.model.fit(X_train, y_train)
        print(f"{self.model_name} trained successfully.")

    def predict(self, X: pd.DataFrame):
        """Returns binary class predictions (0: Stay, 1: Churn)."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        """Returns class probabilities for both outcomes."""
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        """Calculates and displays the top 10 most influential features according to XGBoost."""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print(f"\n--- {self.model_name} Top Feature Importances ---")
        for i in range(min(10, len(self.feature_names))):
            feature_name = self.feature_names[indices[i]]
            score = importances[indices[i]]
            print(f"{i + 1}. {feature_name}: {score:.4f}")


# ==========================================
# 1-3. Data Management Pipeline
# ==========================================
data_handler = DataHandler()
df = data_handler.load_data("../data/student_data_final.csv")

# Feature/Target separation
X, y = data_handler.prepare_features_target(df, target_col="is_churned")

# Train-test split
X_train, X_test, y_train, y_test = data_handler.split_data(X, y)

# ==========================================
# 4. Model Training and Evaluation
# ==========================================
xgb_model = XGBoostClassifierModel(learning_rate=0.05)
xgb_model.train(X_train, y_train)

# Predictions and metric evaluation
y_pred = xgb_model.predict(X_test)
evaluator = ModelEvaluator()
evaluator.evaluate_predictions(y_test, y_pred, xgb_model.model_name)

# Retrieve decision factors
xgb_model.get_feature_importance()

# ==========================================
# 5. Testing on THREE Fine-tuned Student Personas
# ==========================================
# Baseline averages for missing feature completion
train_means = X_train.mean().to_dict()


def create_test_student_df(updates: dict) -> pd.DataFrame:
    """Helper to create a student profile based on global averages with overrides."""
    student_dict = {col: [train_means[col]] for col in X_train.columns}
    student_dict.update(updates)
    return pd.DataFrame(student_dict)


# Realistic profiles where 'total_active_terms' aligns with performance metrics
test_cases = [
    ("AT-RISK (Freshman, poor performance)", create_test_student_df({
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
    ("EXCELLENT (Senior, exemplary)", create_test_student_df({
        "total_active_terms": [6],
        "credits_done_term": [30.0],
        "gpa_term": [4.8],
        "term_success_rate": [1.0],
        "credit_loss_ratio": [0.0],
        "is_first_year": [0]
    }))
]

print("\n" + "=" * 60)
print("FINE-TUNED XGBOOST MODEL TESTING")
print("=" * 60)

for title, student_df in test_cases:
    prediction = xgb_model.predict(student_df)[0]
    probabilities = xgb_model.predict_proba(student_df)[0]

    status_label = "Will Churn" if prediction == 1 else "Will Stay"

    print(f"\n>>> Profile: {title}")
    print(f"Prediction: {status_label}")
    print(f"Retention Probability: {probabilities[0]:.2%}")
    print(f"Dropout Probability: {probabilities[1]:.2%}")