import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class ModelEvaluator:
    """
    Responsible for model performance visualization and metric reporting.
    Follows the Single Responsibility Principle by isolating evaluation logic.
    """

    def __init__(self, class_names=["Active", "Churned"]):
        self.class_names = class_names

    def evaluate_predictions(self, y_test, y_pred, model_name: str):
        """Prints high-level accuracy and a detailed classification report."""
        accuracy = accuracy_score(y_test, y_pred)
        print(f"--- {model_name} Results ---")
        print(f"Accuracy Score: {accuracy:.2%}\n")
        # target_names uses the English labels defined in __init__
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        return accuracy

    def plot_confusion_matrix(self, y_test, y_pred, model_name: str):
        """Visualizes the True Positives, False Positives, etc., using a heatmap."""
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)

        plt.xlabel("Predicted Label (Model guess)")
        plt.ylabel("True Label (Actual data)")
        plt.title(f"{model_name} - Confusion Matrix")
        plt.tight_layout()
        plt.show()