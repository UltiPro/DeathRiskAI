import os
import json
import pandas as pd
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE

from tensorflow_model import TensorflowModel
from utils import RANDOM_SEED, save_tuner_results


class MyTuner(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        kwargs["batch_size"] = trial.hyperparameters.values.get("batch_size", 64)
        return super().run_trial(trial, *args, **kwargs)


if __name__ == "__main__":
    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)

    # Load trainval data for hyperparameter tuning
    print("🔄 Loading trainval data for hyperparameter tuning...")
    X_trainval = pd.read_parquet("./../trainval_test_data/X_trainval.parquet")
    Y_trainval = pd.read_parquet("./../trainval_test_data/Y_trainval.parquet").squeeze()

    # Define the model builder function for Keras Tuner
    def model_builder(hp):
        # Define the hyperparameters for the model
        config = {
            "num_layers": hp.Int("num_layers", min_value=1, max_value=10, step=1),
            "activation": hp.Choice("activation", values=["relu", "elu", "tanh", "swish", "selu"]),
            "optimizer": hp.Choice("optimizer", values=["adam", "rmsprop", "sgd", "adamax", "nadam"]),
            "lr": hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log"),
            "batch_size": hp.Choice("batch_size", values=[32, 64, 128, 256, 512]),
        }

        # Define the hyperparameters (units and dropout) for each layer
        for i in range(10):
            config[f"units_{i}"] = hp.Int(f"units_{i}", min_value=32, max_value=512, step=32)
            config[f"dropout_{i}"] = hp.Float(f"dropout_{i}", min_value=0.0, max_value=0.5, step=0.1)

        # Create and return the TensorFlow model using provided configurations
        return TensorflowModel(model_name="tuned_model", random_seed=RANDOM_SEED).build(
            config=config,
            input_dim=X_trainval.shape[1],
        )

    # Initialize the Keras Tuner with Hyperband
    tuner = MyTuner(
        model_builder,
        objective="val_binary_accuracy",
        max_epochs=30,
        factor=3,
        directory="kerastuner",
        project_name="death_risk_tuning",
        overwrite=True,
    )

    # Early stopping to prevent overfitting
    stop_early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    # Split trainval data into training and validation sets
    # Split 90% for training and 10% for validation
    print("🔄 Splitting trainval data into training and validation sets...")
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval, test_size=0.1, stratify=Y_trainval, random_state=RANDOM_SEED
    )

    # Apply SMOTE oversampling to the training set only
    print("🔄 Applying SMOTE oversampling to the training set only...")
    smote = SMOTE(random_state=RANDOM_SEED)
    X_tune_train, Y_tune_train = smote.fit_resample(X_train, Y_train)

    # Start hyperparameter tuning
    print("🔄 Starting hyperparameter tuning...")
    tuner.search(
        X_tune_train.values,
        Y_tune_train.values,
        validation_data=(X_val.values, Y_val.values),
        callbacks=[stop_early],
        verbose=2,
        epochs=30,
    )

    print("✅ Hyperparameter tuning completed successfully!")

    # Get the best hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("✅ Best hyperparameters:")
    for param, value in best_hp.values.items():
        print(f"{param}: {value}")

    # Save tuner results
    print("💾 Saving tuner results...")
    save_tuner_results(tuner)

    # Save the best hyperparameters
    print("💾 Saving the best hyperparameters...")
    with open("results/best_hp.json", "w") as f:
        json.dump(best_hp.values, f, indent=4)

    print("✅ Hyperparameter tuning completed successfully!")
