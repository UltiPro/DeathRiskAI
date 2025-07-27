import os
import json
import pandas as pd

from tensorflow_model import TensorflowModel
from global_utils import find_best_threshold, save_metrics_table, save_metrics_plot
from utils import RANDOM_SEED


if __name__ == "__main__":
    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)

    # Initialize the TensorFlow model
    tensorflowModel = TensorflowModel(model_name="final_model", random_seed=RANDOM_SEED)

    # Load the TensorFlow model
    print("🔄 Loading the TensorFlow model...")
    tensorflowModel.load("models/final_model.keras")

    # Load test data for evaluation the model
    print("🔄 Loading test data for evaluation the model...")
    X_test = pd.read_parquet("./../trainval_test_data/X_test.parquet")
    Y_test = pd.read_parquet("./../trainval_test_data/Y_test.parquet").squeeze()

    # Predict probabilities on the test set
    print("🔄 Predicting probabilities on the test set...")
    probabilities = tensorflowModel.predict(X_test)

    # Find the best threshold for F1-score
    print("🔍 Finding the best threshold for F1-score...")
    best_threshold, best_f1_score = find_best_threshold(Y_test, probabilities)

    print(f"✅ Best threshold found: {best_threshold} with F1-score: {best_f1_score}")

    # Evaluate the model on the test set with the best threshold
    print("📊 Evaluating the model on the test set with the best threshold...")
    raport, predictions = tensorflowModel.evaluate(X_test, Y_test, threshold=best_threshold)

    print("📊 Evaluation metrics:")
    for metric, value in raport.items():
        print(f"{metric}: {value}")

    # Save the best threshold and F1-score
    print("💾 Saving the best threshold and F1-score...")
    with open("results/threshold.json", "w") as f:
        json.dump({"threshold": best_threshold, "f1_score": best_f1_score}, f, indent=4)

    # Save true values, predictions and probabilities
    print("💾 Saving true values, predictions and probabilities...")
    pd.DataFrame({"true_value": Y_test, "prediction": predictions, "probability": probabilities}).to_csv(
        "results/predictions.csv", index=False
    )

    # Save evaluation metrics
    print("💾 Saving evaluation metrics...")
    with open("results/metrics.json", "w") as f:
        json.dump(raport, f, indent=4)

    # Save evaluation metrics table
    print("💾 Saving evaluation metrics table...")
    save_metrics_table(raport)

    # Plot and save evaluation metrics graph
    print("📈 Plotting and 💾 saving evaluation metrics graph...")
    save_metrics_plot(raport)

    print("✅ Model evaluation pipeline completed successfully!")
