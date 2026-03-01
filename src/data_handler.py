import pandas as pd
from sklearn.model_selection import train_test_split


class DataHandler:
    """
    Responsible for data ingestion, feature separation, and dataset partitioning.
    Encapsulates all data manipulation logic to ensure a clean pipeline.
    """

    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """Loads the dataset from a CSV file."""
        return pd.read_csv(file_path)

    @staticmethod
    def prepare_features_target(df: pd.DataFrame, target_col: str):
        """
        Separates the feature matrix (X) from the target vector (y).

        Args:
            df: The complete DataFrame.
            target_col: The name of the column to predict (e.g., 'is_churned').
        """
        df = df.copy()
        X = df.drop(columns=[target_col])
        y = df[target_col]

        print(f"Target variable distribution ({target_col}):")
        print(y.value_counts(normalize=True).map('{:.2%}'.format))
        return X, y

    @staticmethod
    def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
        """
        Splits the data into training and testing sets using stratified sampling.

        Args:
            test_size: Proportion of the dataset to include in the test split.
            stratify: Ensures the churn ratio is preserved in both sets.
        """
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )