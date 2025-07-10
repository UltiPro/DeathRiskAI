import os
import joblib
import matplotlib.pyplot as plt

from feature_transformation import RANDOM_SEED as RND_SEED

RANDOM_SEED = RND_SEED




# Upewnij się, że foldery do zapisów istnieją
os.makedirs("models", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)


def save_model(model, path: str) -> None:
    """
    Saves the scikit-learn model to the specified path.
    """
    joblib.dump(model, path)
    print(f"💾 Model saved to {path}")


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


def save_training_metrics(training_metrics, name="model"):
    """
    Saves the training metrics to a CSV file.
    """
    training_metrics.to_csv(f"results/{name}_training_metrics.csv", index=False)
    print(f"💾 Training metrics saved to results/{name}_training_metrics.csv")
