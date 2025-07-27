import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from scikit_model import SklearnModel
from utils import RANDOM_SEED, save_feature_importance


if __name__ == "__main__":
    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Initialize the Sklearn model
    sklearnModel = SklearnModel(model_name="final_model", random_seed=RANDOM_SEED)

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
    model = sklearnModel.build(best_hp)

    # Train the final model
    print("🔄 Training the final model...")
    sklearnModel.train(X_tune_train, Y_tune_train, class_weight={0: 1.0, 1: 1.0})

    print("✅ Final model trained successfully!")

    # Visualize and save the feature importance
    print("📈 Visualizing and 💾 saving the feature importance...")
    save_feature_importance(model, X_train.columns, name="final_model", top_n=32)

    # Save the final model
    print("💾 Saving the final model...")
    sklearnModel.save("models/final_model.pkl")

    print("✅ Model training pipeline completed successfully!")
