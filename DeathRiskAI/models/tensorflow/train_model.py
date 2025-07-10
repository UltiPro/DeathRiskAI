import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from imblearn.over_sampling import SMOTE

from tensorflow_model import TensorflowModel
from utils import RANDOM_SEED, save_model_visualization, save_training_history

if __name__ == "__main__":
    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Initialize the TensorFlow model
    tensorflowModel = TensorflowModel(model_name="final_model", random_seed=RANDOM_SEED)

    # Early stopping to prevent overfitting
    stop_early = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    # Model checkpoint to save the best model
    checkpoint = ModelCheckpoint(
        "models/final_model.keras", monitor="val_loss", save_best_only=True, verbose=1
    )

    # Load trainval data for training the model
    print("🔄 Loading trainval data for training the model...")
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

    # Load the best hyperparameters
    print("🔄 Loading the best hyperparameters from results/best_hp.json...")
    with open("results/best_hp.json", "r") as f:
        best_hp = json.load(f)

    # Build the model with the best hyperparameters
    print("🔄 Building the model with the best hyperparameters...")
    model = tensorflowModel.build(best_hp, input_dim=X_tune_train.shape[1])

    # Train the final model
    print("🔄 Training the final model...")
    history = tensorflowModel.train(
        X_tune_train,
        Y_tune_train,
        X_val,
        Y_val,
        epochs=100,
        batch_size=best_hp["batch_size"],
        callbacks=[stop_early, checkpoint],
        # Smote has already balanced the classes
        # Change class weights only if necessary
        # class_weight={0: 1.0, 1: 1.0},
    )

    print("✅ Final model trained successfully!")

    # Visualize and save the model architecture
    print("📈 Visualizing and 💾 saving the model architecture...")
    save_model_visualization(model, name="final_model")

    # Save the training history
    print("💾 Saving the training history...")
    save_training_history(history, name="final_model")

    # Save the final model
    print("💾 Saving the final model...")
    tensorflowModel.save("models/final_model.keras")

    print("✅ Model training pipeline completed successfully!")
