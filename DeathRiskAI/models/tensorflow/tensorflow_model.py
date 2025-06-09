import sys
import numpy as np
import random
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_template import ModelTemplate

import tensorflow as tf
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.utils import plot_model


class TensorflowModel(ModelTemplate):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = None
        tf.random.set_seed(42)
        np.random.seed(42)
        random.seed(42)

    def build_model(self, input_dim: int) -> tf.keras.Model:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    def build_model_with_hp(self, hp, input_dim: int) -> tf.keras.Model:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_dim,)))

        for i in range(hp.Int("num_layers", 2, 5)):
            model.add(
                tf.keras.layers.Dense(
                    units=hp.Int(f"units_{i}", min_value=32, max_value=256, step=32),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                )
            )
            model.add(
                tf.keras.layers.Dropout(
                    rate=hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.1)
                )
            )

        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Float("lr", 1e-4, 1e-2, sampling="log")
            ),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def build_model_from_config(
        self, hp_config: dict, input_dim: int
    ) -> tf.keras.Model:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_dim,)))

        for i in range(hp_config["num_layers"]):
            model.add(
                tf.keras.layers.Dense(
                    units=hp_config[f"units_{i}"],
                    activation=hp_config["activation"],
                )
            )
            model.add(tf.keras.layers.Dropout(rate=hp_config[f"dropout_{i}"]))

        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.AUC(),
            ],
        )
        return model

    def train(self, X, Y, X_val, Y_val, epochs=100, batch_size=32) -> object:
        self.model = self.build_model(input_dim=X.shape[1])

        # Zapisz strukturę modelu do pliku graficznego
        os.makedirs("visualizations", exist_ok=True)
        plot_model(
            self.model,
            to_file=f"visualizations/{self.model_name}_structure.png",
            show_shapes=True,
            show_layer_names=True,
        )

        os.makedirs("models", exist_ok=True)
        model_path = f"models/{self.model_name}.keras"

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
            class_weight={0: 1.0, 1: 10.0},
        )
        return self.model

    def predict_proba(self, X) -> pd.Series:
        preds = self.model.predict(X)
        return pd.Series(preds.flatten())

    def predict(self, X, threshold=0.3) -> pd.Series:
        return self.predict_proba(X).apply(lambda x: 1 if x >= threshold else 0)

    def evaluate(self, X, Y) -> dict:
        preds = self.predict(X)
        return {
            "accuracy": accuracy_score(Y, preds),
            "precision": precision_score(Y, preds, zero_division=0),
            "recall": recall_score(Y, preds, zero_division=0),
            "f1_score": f1_score(Y, preds, zero_division=0),
        }

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = tf.keras.models.load_model(path)
