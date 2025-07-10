import os
import sys
import pandas as pd
from typing import Optional, Union, Tuple, List, Dict
from sklearn.metrics import classification_report
import tensorflow as tf
from keras_tuner.engine.hyperparameters import HyperParameters

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_template import ModelTemplate


class TensorflowModel(ModelTemplate):
    def __init__(self, model_name: str, random_seed: int):
        """
        Initializes the TensorFlow model with a given name and random seed.
        """
        super().__init__(model_name, random_seed)
        self.model = None

    def build(
        self, config: Union[Dict[str, float | int | str], HyperParameters], input_dim: int
    ) -> tf.keras.Model:
        """
        Builds a TensorFlow model based on the provided configuration.
        """
        # Define available optimizers
        optimizers = {
            "adam": tf.keras.optimizers.Adam,
            "rmsprop": tf.keras.optimizers.RMSprop,
            "sgd": tf.keras.optimizers.SGD,
            "adamax": tf.keras.optimizers.Adamax,
            "nadam": tf.keras.optimizers.Nadam,
        }

        # Helper functions to get configuration values with defaults
        def get(key, default=None):
            try:
                return config[key]
            except (KeyError, TypeError):
                return default

        # Helper function to convert configuration values to int
        def get_int(key, default=None):
            return int(get(key, default))

        # Helper function to convert configuration values to float
        def get_float(key, default=None):
            return float(get(key, default))

        # Create a sequential model
        model = tf.keras.Sequential()

        # Add an input layer with the specified input dimension
        model.add(tf.keras.layers.Input(shape=(input_dim,)))

        # Add hidden layers based on the configuration
        activation = get("activation", "relu")
        for i in range(get_int("num_layers", 1)):
            model.add(tf.keras.layers.Dense(units=get_int(f"units_{i}", 64), activation=activation))
            model.add(tf.keras.layers.Dropout(rate=get_float(f"dropout_{i}", 0.0)))

        # Add the output layer
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        # Compile the model with the specified optimizer, loss function and metrics
        model.compile(
            optimizer=optimizers.get(get("optimizer", "adam"))(learning_rate=get_float("lr", 0.001)),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.AUC(),
            ],
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
        epochs: int,
        batch_size: int = 64,
        callbacks: List[tf.keras.callbacks.Callback] = [],
        class_weight: Dict[int, float] = {0: 1.0, 1: 1.0},
    ) -> tf.keras.callbacks.History:
        """
        Trains the TensorFlow model with the provided training data, validation data and configuration parameters.
        Returns the training history.
        """
        return self.model.fit(
            X_train.values,
            Y_train.values,
            validation_data=(X_val.values, Y_val.values),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            shuffle=True,
            verbose=2,
        )

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> pd.Series:
        """
        Predicts the probabilities of the positive class for the provided dataset.
        If a threshold is provided, it returns binary predictions based on that threshold.
        """
        if threshold is not None:
            return pd.Series((self.model.predict(X).flatten() >= threshold).astype(int))
        return pd.Series(self.model.predict(X).flatten())

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
        Saves the TensorFlow model to the specified path.
        """
        if not os.path.exists(os.path.dirname(path)):
            raise FileNotFoundError(f"🛑 Directory does not exist for saving model at '{path}'.")
        self.model.save(path)

    def load(self, path: str) -> None:
        """
        Loads a TensorFlow model from the specified path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"🛑 Model file not found at '{path}'.")
        self.model = tf.keras.models.load_model(path)
