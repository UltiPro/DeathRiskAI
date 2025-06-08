from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class Model(ABC):
    """
    Base class for all models in the DeathRiskAI project.
    This class provides a template for model training, prediction, evaluation and saving/loading models.
    """

    def __init__(self, model_name: str):
        """
        Initializes the model with a given name.

        :param model_name: Name of the model.
        """
        self.model_name = model_name
        self.model: Optional[object] = None  # Placeholder for the actual model instance

    @abstractmethod
    def train(self, X: pd.DataFrame, Y: pd.Series) -> object:
        """
        Trains the model using the provided features and target.

        :param X: Features for training.
        :param Y: Target variable for training.
        :return: Trained model instance.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the trained model.

        :param X: Features to make predictions on.
        :return: Predictions as a pandas Series.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def evaluate(self, data: pd.DataFrame) -> dict[str, float]:
        """
        Evaluates the model on the provided data and returns metrics such as accuracy, precision, recall, etc.

        :param data: DataFrame with features and target.
        :return: Dictionary with evaluation metrics.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Saves the model to the specified path.

        :param path: Path to save the model.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Loads the model from the specified path.

        :param path: Path to load the model from.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
