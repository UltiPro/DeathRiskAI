import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

from feature_transformation import RANDOM_SEED as RND_SEED

RANDOM_SEED = RND_SEED


def save_grid_search_results(grid_search: GridSearchCV, name: str = "grid_search") -> None:
    """
    Saves the results of GridSearchCV to a JSON file.
    """
    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)

    # Extract the full results
    results = grid_search.cv_results_

    # Convert all ndarrays to lists for JSON serialization
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results[key] = value.tolist()

    # Save the results to a JSON file
    with open(f"results/{name}_summary.txt", "w") as f:
        json.dump(
            {
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
                "results": results,
            },
            f,
            indent=4,
        )


def save_feature_importance(model, feature_names: list, name="model", top_n=32):
    """
    P
    """
    if hasattr(model, "feature_importances_"):
        # Ensure the visualizations directory exists
        os.makedirs("visualizations", exist_ok=True)

        # Sort the feature importances and get the top N
        top_indices = model.feature_importances_.argsort()[::-1][:top_n]

        # Plot the top N feature importances
        plt.figure(figsize=(8, 6))
        plt.barh(range(top_n), model.feature_importances_[top_indices], align="center")
        plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.title(f"Feature Importance for {name}")
        plt.tight_layout()
        plt.savefig(f"visualizations/{name}_feature_importance.png")
        plt.close()
    else:
        print(f"🛑 Model does not have feature importances attribute.")
