import os
import json
import pandas as pd
from sklearn.model_selection import GridSearchCV
from utils import save_grid_search_results
from scikit_model import SklearnModel

if __name__ == "__main__":
    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)

    # Initialize the Sklearn model
    sklearnModel = SklearnModel(model_name="random_forest_model", random_seed=42)

    # Configure GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=sklearnModel.build(config={}),
        param_grid={
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
        },
        cv=3,
        n_jobs=-1,
        verbose=2,
        scoring="f1",
    )

    # Load trainval data for hyperparameter tuning
    print("🔄 Loading trainval data for hyperparameter tuning...")
    X_trainval = pd.read_parquet("./../trainval_test_data/X_trainval.parquet")
    Y_trainval = pd.read_parquet("./../trainval_test_data/Y_trainval.parquet").squeeze()

    # Start hyperparameter tuning
    print("🔄 Starting hyperparameter tuning...")
    grid_search.fit(X_trainval, Y_trainval)

    print("✅ Hyperparameter tuning completed successfully!")

    # Get the best hyperparameters
    best_hp = grid_search.best_params_

    print("✅ Best hyperparameters:")
    for param, value in best_hp.items():
        print(f"{param}: {value}")

    # Save grid search results
    print("💾 Saving grid search results...")
    save_grid_search_results(grid_search)

    # Save the best hyperparameters
    print("💾 Saving the best hyperparameters...")
    with open("results/best_hp.json", "w") as f:
        json.dump(best_hp, f, indent=4)

    print("✅ Hyperparameter tuning pipeline completed successfully!")
