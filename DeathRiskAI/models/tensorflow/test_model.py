import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
from tensorflow_model import TensorflowModel
import json


def find_best_threshold(y_true, y_proba):
    thresholds = np.arange(0.0, 1.01, 0.01)
    f1_scores = [f1_score(y_true, y_proba >= t) for t in thresholds]
    best_threshold = float(thresholds[np.argmax(f1_scores)])
    best_f1 = float(max(f1_scores))
    return best_threshold, best_f1


if __name__ == "__main__":
    # Load test data
    X_test = pd.read_parquet("./../trainval_test_data/X_test.parquet")
    Y_test = pd.read_parquet("./../trainval_test_data/Y_test.parquet").squeeze()

    # Load final model
    model = TensorflowModel(model_name="final_model")
    model.load("models/final_model.keras")

    # Predict probabilities
    proba = model.predict_proba(X_test)

    # Find best threshold based on F1-score
    best_thresh, best_f1 = find_best_threshold(Y_test, proba)
    print(f"\n🔍 Best threshold found: {best_thresh:.2f} (F1-score: {best_f1:.4f})")

    # Predict using best threshold
    predictions = (proba >= best_thresh).astype(int)

    # Save threshold
    with open("results/best_threshold.json", "w") as f:
        json.dump({"best_threshold": best_thresh, "f1_score": best_f1}, f)

    # Save predictions
    results = pd.DataFrame({
        "true": Y_test,
        "pred": predictions,
        "proba": proba
    })
    results.to_csv("results/test_predictions.csv", index=False)

    # Classification report
    report = classification_report(Y_test, predictions, output_dict=True)
    print("\n📊 Classification Report on Test Data:")
    print(classification_report(Y_test, predictions))

    pd.DataFrame(report).transpose().to_csv("results/test_metrics.csv", index=True)

    print("\n✅ Test predictions, metrics and threshold saved to results/")
