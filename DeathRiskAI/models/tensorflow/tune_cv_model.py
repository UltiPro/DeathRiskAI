import pandas as pd
import keras_tuner as kt
from tensorflow_model import TensorflowModel
from tensorflow.keras.callbacks import EarlyStopping
from plot_utils import plot_training_history, plot_tuner_results
import os
import json

def load_data_for_tuning():
    X_train = pd.read_parquet("./../train_test_data/fold1_X_train.parquet")
    Y_train = pd.read_parquet("./../train_test_data/fold1_Y_train.parquet").squeeze()
    X_val = pd.read_parquet("./../train_test_data/fold1_X_val.parquet")
    Y_val = pd.read_parquet("./../train_test_data/fold1_Y_val.parquet").squeeze()
    return X_train, Y_train, X_val, Y_val

def model_builder(hp):
    model_wrapper = TensorflowModel(model_name="tuned_model")
    return model_wrapper.build_model_with_hp(hp, input_dim=input_dim)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    X_train, Y_train, X_val, Y_val = load_data_for_tuning()
    input_dim = X_train.shape[1]

    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=20,
        factor=3,
        directory="kerastuner",
        project_name="death_risk_tuning",
        overwrite=True,
    )

    stop_early = EarlyStopping(monitor="val_loss", patience=5)

    tuner.search(
        X_train.values, Y_train.values,
        validation_data=(X_val.values, Y_val.values),
        callbacks=[stop_early],
        batch_size=32,
        verbose=2,
    )

    plot_tuner_results(tuner)

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nNajlepsze hiperparametry:")
    for param in best_hp.values.keys():
        print(f"{param}: {best_hp.get(param)}")

    with open("results/best_hp.json", "w") as f:
        json.dump(best_hp.values, f, indent=4)

    model = tuner.hypermodel.build(best_hp)
    history = model.fit(
        X_train.values, Y_train.values,
        validation_data=(X_val.values, Y_val.values),
        epochs=20, batch_size=32,
        callbacks=[stop_early],
        verbose=2,
    )

    plot_training_history(history, name="tuned_model")
    model.save("models/tuned_model.keras")
