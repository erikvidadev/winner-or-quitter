from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract Base Class (ABC) for all machine learning models.
    Enforces the 'Interface Segregation' and 'Liskov Substitution' principles
    by ensuring all derived models implement these core methods.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Fits the model to the training data.

        Args:
            X_train: Feature matrix (DataFrame or ndarray)
            y_train: Target labels (Series or ndarray)
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predicts class labels for the given features.

        Args:
            X: Feature matrix to predict on
        Returns:
            Array of binary predictions (0 or 1)
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Predicts class probabilities for the given features.

        Args:
            X: Feature matrix to predict on
        Returns:
            Array of probability pairs [prob_stay, prob_churn]
        """
        pass