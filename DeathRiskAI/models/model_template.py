import random
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Union

import tensorflow as tf


class ModelTemplate(ABC):
    """
    Base class for all models (TensorFlow, scikit-learn, etc.).
    This class provides a template for model training, prediction, evaluation, and saving/loading models.
    """

    def __init__(self, model_name: str, random_seed: int = 42):
        """
        Initializes the model with a given name and random seed.

        :param model_name: Name of the model.
        """
        self.model_name = model_name
        self.model: Optional[object] = None  # Placeholder for the actual model instance
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

    @abstractmethod
    def build(self, config: Union[dict, object], input_dim: int) -> object:
        """
        Builds the model based on the provided configuration.

        :param config: Hyperparameter configuration or similar parameters.
        :param input_dim: Number of features in the input data.
        :return: A built model (e.g., RandomForestClassifier, Keras model, etc.).
        """
        pass

    @abstractmethod
    def train(
        self, X_train: pd.DataFrame, Y_train: pd.Series, X_val: pd.DataFrame, Y_val: pd.Series, **kwargs
    ) -> None:
        """
        Trains the model using the provided data.

        :param X_train: Training features.
        :param Y_train: Training target variable.
        :param X_val: Validation features.
        :param Y_val: Validation target variable.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> pd.Series:
        """
        Makes predictions using the trained model.

        :param X: Features to make predictions on.
        :param threshold: Optional threshold to convert probabilities to binary predictions.
        :return: Predictions (either probabilities or binary predictions).
        """
        pass

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, Y: pd.Series, threshold: float) -> Union[dict, pd.Series]:
        """
        Evaluates the model's performance on the provided dataset.

        :param X: Features for evaluation.
        :param Y: Actual target variable.
        :param threshold: Threshold for binary classification.
        :return: Evaluation metrics and predictions.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Saves the trained model to the specified path.

        :param path: Path to save the model.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Loads the model from the specified path.

        :param path: Path to load the model.
        """
        pass
