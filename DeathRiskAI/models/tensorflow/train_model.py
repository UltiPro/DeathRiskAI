import pandas as pd
import tensorflow as tf
from tensorflow_model import TensorflowModel
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from plot_utils import plot_training_history
import json
import os

from imblearn.over_sampling import SMOTE


def load_trainval_data():
    X_trainval = pd.read_parquet("./../trainval_test_data/X_trainval.parquet")
    Y_trainval = pd.read_parquet("./../trainval_test_data/Y_trainval.parquet").squeeze()
    return X_trainval, Y_trainval


def load_best_hp(path="results/best_hp.json"):
    with open(path, "r") as f:
        best_hp = json.load(f)
    return best_hp


if __name__ == "__main__":
    # Load data
    X_trainval, Y_trainval = load_trainval_data()
    input_dim = X_trainval.shape[1]

    # Oversampling SMOTE tylko na trainval
    print("🔄 Applying SMOTE oversampling to address class imbalance...")
    smote = SMOTE(random_state=42)
    X_resampled, Y_resampled = smote.fit_resample(X_trainval.values, Y_trainval.values)

    # Zmień na DataFrame (niekonieczne, ale wygodne do debugowania i kompatybilności)
    X_resampled = pd.DataFrame(X_resampled, columns=X_trainval.columns)
    Y_resampled = pd.Series(Y_resampled, name="death")

    # Load best hyperparameters
    best_hp = load_best_hp()

    # Initialize model
    model_wrapper = TensorflowModel(model_name="final_model")
    model = model_wrapper.build_model_from_config(best_hp, input_dim=input_dim)
    model_wrapper.model = (
        model  # 🔁 Ustawienie we wrapperze do użycia w visualize_model
    )

    # ✅ Wizualizacja struktury modelu
    model_wrapper.visualize_model()

    # Prepare callbacks
    os.makedirs("models", exist_ok=True)
    stop_early = EarlyStopping(
        monitor="val_loss", patience=30, restore_best_weights=True
    )
    checkpoint = ModelCheckpoint(
        "models/final_model.keras", monitor="val_loss", save_best_only=True, verbose=1
    )

    # Train final model on resampled data
    history = model.fit(
        X_resampled,
        Y_resampled,
        validation_split=0.1,  # 10% of trainval for validation
        epochs=100,
        batch_size=32,
        callbacks=[stop_early, checkpoint],
        verbose=2,
    )

    # Plot training history
    plot_training_history(history, name="final_model")

    print("\n✅ Final model trained and saved to models/final_model.keras")
