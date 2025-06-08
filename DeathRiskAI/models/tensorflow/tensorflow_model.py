import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_template import ModelTemplate

import tensorflow as tf
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TensorflowModel(ModelTemplate):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = None

    def build_model(self, input_dim: int) -> tf.keras.Model:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu", input_shape=(input_dim,)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(4, activation="relu"),
                tf.keras.layers.Dense(2, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    def train(
        self,
        X: pd.DataFrame,
        Y: pd.Series,
        X_val: pd.DataFrame,
        Y_val: pd.Series,
        epochs: int = 20,
        batch_size: int = 32,
    ) -> object:

        self.model = self.build_model(input_dim=X.shape[1])

        # Ensure model directory exists
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{self.model_name}.h5"

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ModelCheckpoint(
                model_path, monitor="val_loss", save_best_only=True, verbose=1
            ),
        ]

        self.model.fit(
            X.values,
            Y.values,
            validation_data=(X_val.values, Y_val.values),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2,
        )

        return self.model

    def predict(self, X) -> pd.Series:
        preds = self.model.predict(X)
        return pd.Series(preds.flatten()).apply(lambda x: 1 if x >= 0.5 else 0)

    def evaluate(self, X, Y) -> dict:
        preds = self.predict(X)
        return {
            "accuracy": accuracy_score(Y, preds),
            "precision": precision_score(Y, preds),
            "recall": recall_score(Y, preds),
            "f1_score": f1_score(Y, preds),
        }

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = tf.keras.models.load_model(path)
