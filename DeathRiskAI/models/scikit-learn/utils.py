import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

from feature_transformation import RANDOM_SEED as RND_SEED

RANDOM_SEED = RND_SEED

# Upewnij się, że foldery do zapisów istnieją
os.makedirs("models", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)


def save_grid_search_results(
    grid_search: GridSearchCV, path: str = "results/grid_search_results.json"
) -> None:
    """
    Saves the results of GridSearchCV, including the best parameters, best score, and all results, to a JSON file.

    :param grid_search: The fitted GridSearchCV object.
    :param path: The file path to save the results.
    """
    # Create the results directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Extract the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Extract the full results (including all parameter combinations and their scores)
    results = grid_search.cv_results_

    # Prepare the data to save (best parameters, best score, and detailed results)
    data_to_save = {"best_params": best_params, "best_score": best_score, "results": results}

    # Save the results to a JSON file
    with open(path, "w") as f:
        json.dump(data_to_save, f, indent=4)

    print(f"✅ GridSearchCV results saved to {path}")


def save_feature_importance(model, feature_names: list, name="model"):
    """
    Plots and saves the feature importance of a trained model.
    For RandomForest or other tree-based models.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]

        plt.figure(figsize=(8, 6))
        plt.title(f"Feature Importance - {name}")
        plt.barh(range(len(feature_names)), importances[indices], align="center")
        plt.yticks(range(len(feature_names)), [feature_names[i] for i in indices])
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(f"visualizations/{name}_feature_importance.png")
        plt.close()
    else:
        print(f"🛑 Model does not have feature importances attribute.")
