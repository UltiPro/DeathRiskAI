import os
import json
import pandas as pd
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE

from tensorflow_model import TensorflowModel
from utils import RANDOM_SEED, save_tuner_results


if __name__ == "__main__":
    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)

    # Define the model builder function for Keras Tuner
    def model_builder(hp):
        model_wrapper = TensorflowModel(model_name="tuned_model")
        return model_wrapper.build_model_with_hp(hp, input_dim=X_trainval.shape[1])

    # Initialize the Keras Tuner with Hyperband
    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=30,
        factor=3,
        directory="kerastuner",
        project_name="death_risk_tuning",
        overwrite=True,
    )

    # Early stopping to prevent overfitting
    stop_early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    # Load trainval data for hyperparameter tuning
    print("🔄 Loading trainval data for hyperparameter tuning...")
    X_trainval = pd.read_parquet("./../trainval_test_data/X_trainval.parquet")
    Y_trainval = pd.read_parquet("./../trainval_test_data/Y_trainval.parquet").squeeze()

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
        X_tune_train,
        Y_tune_train,
        validation_data=(X_val.values, Y_val.values),
        callbacks=[stop_early],
        batch_size=64,
        verbose=2,
        epochs=30,
    )

    # Save tuner results
    print("🔄 Saving tuner results...")
    save_tuner_results(tuner)

    # Get the best hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Save the best hyperparameters
    print("🔄 Saving the best hyperparameters to results/best_hp.json...")
    with open("results/best_hp.json", "w") as f:
        json.dump(best_hp.values, f, indent=4)

    print("✅ Best hyperparameters:")
    for param, value in best_hp.values.items():
        print(f"{param}: {value}")
