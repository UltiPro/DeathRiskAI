import random
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Dict
import tensorflow as tf

from feature_transformation import RANDOM_SEED


class ModelTemplate(ABC):
    """
    Base class for all models (TensorFlow, scikit-learn, etc.).
    This class provides a template for model building, training, prediction, evaluation, and saving/loading models.
    """

    def __init__(self, model_name: str, random_seed: int = RANDOM_SEED):
        """
        Initializes a model using the provided name and random seed.
        """
        self.model_name: str = model_name
        self.model: Optional[object] = None  # Placeholder for a model object
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

    @abstractmethod
    def build(self, config: Union[dict, object], **kwargs) -> object:
        """
        Builds a model based on the provided configuration.
        """
        raise NotImplementedError("Build method is not implemented for this model.")

    @abstractmethod
    def train(self, X_train: pd.DataFrame, Y_train: pd.Series, **kwargs) -> None:
        """
        Trains a model using the provided dataset.
        """
        raise NotImplementedError("Train method is not implemented for this model.")

    @abstractmethod
    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> pd.Series:
        """
        Makes predictions using a model on the provided dataset.
        """
        raise NotImplementedError("Predict method is not implemented for this model.")

    @abstractmethod
    def evaluate(
        self, X: pd.DataFrame, Y: pd.Series, threshold: float
    ) -> Tuple[Dict[str, Dict[str, float]], pd.Series]:
        """
        Evaluates a model's performance on the provided dataset.
        """
        raise NotImplementedError("Evaluate method is not implemented for this model.")

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Saves a model to the specified path.
        """
        raise NotImplementedError("Save method is not implemented for this model.")

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Loads a model from the specified path.
        """
        raise NotImplementedError("Load method is not implemented for this model.")
