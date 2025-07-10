import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


class SklearnModel:
    def __init__(self, model_name: str, random_seed: int):
        """
        Initializes the Sklearn model with a given name and random seed.
        """
        super().__init__(model_name, random_seed)
        self.model = None

    def build(self, config: Dict[str, Union[int, str]]) -> RandomForestClassifier:
        """
        Builds a Sklearn model based on the provided configuration.
        """

        # Helper function to get configuration values with defaults
        def get(key, default=None):
            return config.get(key, default)

        # Create a RandomForestClassifier model with provided hyperparameters
        self.model = RandomForestClassifier(
            n_estimators=get("n_estimators", 100),
            max_depth=get("max_depth", None),
            min_samples_split=get("min_samples_split", 2),
            min_samples_leaf=get("min_samples_leaf", 1),
            bootstrap=get("bootstrap", True),
            random_state=np.random.seed(),
        )

        return self.model

    def train(
        self, X_train: pd.DataFrame, Y_train: pd.Series, class_weight: Dict[int, float] = {0: 1.0, 1: 1.0}
    ) -> None:
        """
        Trains the Sklearn model with the provided training data and configuration parameters.
        """
        self.model.fit(X_train, Y_train, sample_weight=class_weight)

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> pd.Series:
        """
        Predicts the probabilities of the positive class for the provided dataset.
        If a threshold is provided, it returns binary predictions based on that threshold.
        """
        if threshold is not None:
            return pd.Series((self.model.predict_proba(X)[:, 1] >= threshold).astype(int))
        return pd.Series(self.model.predict_proba(X)[:, 1])

    def evaluate(
        self, X: pd.DataFrame, Y: pd.Series, threshold: float
    ) -> Tuple[Dict[str, Dict[str, float]], pd.Series]:
        """
        Evaluates the model's performance on the provided dataset and returns
        a tuple containing a dictionary of evaluation metrics and the predictions.
        """
        predictions = self.predict(X, threshold)
        return classification_report(Y, predictions, output_dict=True), predictions

    def save(self, path: str) -> None:
        """
        Saves the Sklearn model to the specified path.
        """
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        """
        Loads a Sklearn model from the specified path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"🛑 Model file not found at '{path}'.")
        self.model = joblib.load(path)
