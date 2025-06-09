import os
import pandas as pd
import keras_tuner as kt
import tensorflow as tf
from tensorflow_model import TensorflowModel
from tensorflow.keras.callbacks import EarlyStopping
from plot_utils import plot_training_history, plot_tuner_results
import json

if __name__ == "__main__":
    # Load the entire trainval data for tuning
    X_train = pd.read_parquet("./../train_test_data//X_trainval.parquet")
    Y_train = pd.read_parquet("./../train_test_data//Y_trainval.parquet").squeeze()

    input_dim = X_train.shape[1]

    def model_builder(hp):
        model_wrapper = TensorflowModel(model_name="tuned_model")
        return model_wrapper.build_model_with_hp(hp, input_dim=input_dim)

    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=30,
        factor=3,
        directory="kerastuner",
        project_name="death_risk_tuning",
        overwrite=True,
    )

    stop_early = EarlyStopping(monitor="val_loss", patience=5)

    # We split trainval (90%) again just for tuning val data (10% z tego)
    X_tune_train, X_tune_val = (
        X_train.iloc[: -int(0.1 * len(X_train))],
        X_train.iloc[-int(0.1 * len(X_train)) :],
    )
    Y_tune_train, Y_tune_val = (
        Y_train.iloc[: -int(0.1 * len(Y_train))],
        Y_train.iloc[-int(0.1 * len(Y_train)) :],
    )

    tuner.search(
        X_tune_train.values,
        Y_tune_train.values,
        validation_data=(X_tune_val.values, Y_tune_val.values),
        callbacks=[stop_early],
        batch_size=32,
        verbose=2,
        epochs=30,
    )

    plot_tuner_results(tuner)

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nNajlepsze hiperparametry:")
    for param, value in best_hp.values.items():
        print(f"{param}: {value}")

    # Save best hyperparameters for later use
    with open("results/best_hp.json", "w") as f:
        json.dump(best_hp.values, f, indent=4)

    # Build the best model and train it fully
    model = tuner.hypermodel.build(best_hp)
    history = model.fit(
        X_train.values,
        Y_train.values,
        validation_data=(X_tune_val.values, Y_tune_val.values),
        epochs=30,
        batch_size=32,
        callbacks=[stop_early],
        verbose=2,
    )

    plot_training_history(history, name="tuned_model")

    # Save best model
    os.makedirs("models", exist_ok=True)
    model.save("models/tuned_model.keras")
