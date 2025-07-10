import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from typing import Optional, Union, Tuple, Dict
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

class SklearnModel:
    def __init__(self, model_name: str, random_seed: int):
        """
        Initializes the scikit-learn model with a given name and random seed.
        """
        self.model_name = model_name
        self.model = None
        random.seed(random_seed)
        np.random.seed(random_seed)

    def build(self, config: Dict[str, Union[int, str]], input_dim: int) -> RandomForestClassifier:
        """
        Builds a RandomForestClassifier model based on the provided configuration.
        """
        # Helper function to get configuration values with defaults
        def get(key, default=None):
            return config.get(key, default)

        # Create and return the RandomForestClassifier model
        model = RandomForestClassifier(
            n_estimators=get("n_estimators", 100),
            max_depth=get("max_depth", None),
            min_samples_split=get("min_samples_split", 2),
            min_samples_leaf=get("min_samples_leaf", 1),
            bootstrap=get("bootstrap", True),
            random_state=np.random.seed(),
        )

        # Store the model in the instance variable
        self.model = model

        return model

    def train(
        self,
        X_train: pd.DataFrame,
        Y_train: pd.Series,
        X_val: pd.DataFrame,
        Y_val: pd.Series,
        **kwargs
    ) -> None:
        """
        Trains the scikit-learn model with the provided training data, validation data, and configuration parameters.
        """
        self.model.fit(X_train, Y_train)

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> pd.Series:
        """
        Predicts the class labels for the provided dataset.
        If a threshold is provided, it returns binary predictions based on that threshold.
        """
        probabilities = self.model.predict_proba(X)[:, 1]  # For binary classification, take class 1 probabilities
        if threshold is not None:
            return pd.Series((probabilities >= threshold).astype(int))
        return pd.Series(probabilities)

    def evaluate(
        self, X: pd.DataFrame, Y: pd.Series, threshold: Optional[float] = 0.5
    ) -> Tuple[Dict[str, Dict[str, float]], pd.Series]:
        """
        Evaluates the model's performance on the provided dataset and returns a tuple containing a dictionary of
        evaluation metrics and the predictions.
        """
        predictions = self.predict(X, threshold)
        report = classification_report(Y, predictions, output_dict=True)
        return report, predictions

    def save(self, path: str) -> None:
        """
        Saves the scikit-learn model to the specified path.
        """
        import joblib
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        """
        Loads a scikit-learn model from the specified path.
        """
        import joblib
        if not os.path.exists(path):
            raise FileNotFoundError(f"🛑 Model file not found at '{path}'.")
        self.model = joblib.load(path)
