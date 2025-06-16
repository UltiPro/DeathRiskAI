import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
from tensorflow_model import TensorflowModel


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

    # Find best threshold
    best_thresh, best_f1 = find_best_threshold(Y_test, proba)
    print(f"\n🔍 Best threshold found: {best_thresh:.2f} (F1-score: {best_f1:.4f})")

    # Predict using best threshold
    predictions = (proba >= best_thresh).astype(int)

    # Ensure output folder
    os.makedirs("results", exist_ok=True)

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

    # 📈 Plot classification metrics
    metrics_df = pd.DataFrame(report).transpose()
    metrics_to_plot = ["precision", "recall", "f1-score"]
    classes_to_plot = ["0", "1"]
    metrics_df = metrics_df.loc[classes_to_plot, metrics_to_plot]

    ax = metrics_df.plot(kind="bar", figsize=(8, 6))
    plt.title("Precision, Recall, F1-score per Class")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("results/classification_metrics.png")
    plt.close()

    print("📈 Saved bar chart: results/classification_metrics.png")

    # 📄 Save pretty table of metrics to TXT
    with open("results/test_metrics_table.txt", "w") as f:
        f.write("📊 Classification Report on Test Data\n\n")
        f.write("{:<15} {:<10} {:<10} {:<10} {:<10}\n".format("Class", "Precision", "Recall", "F1-score", "Support"))
        f.write("-" * 60 + "\n")
        for label in ["0", "1", "accuracy", "macro avg", "weighted avg"]:
            if label == "accuracy":
                support_sum = int(report["0"]["support"]) + int(report["1"]["support"])
                f.write("{:<15} {:<10} {:<10} {:<10.2f} {:<10}\n".format(
                    label, "", "", report["accuracy"], support_sum
                ))
            else:
                row = report[label]
                f.write("{:<15} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}\n".format(
                    label,
                    row.get("precision", 0.0),
                    row.get("recall", 0.0),
                    row.get("f1-score", 0.0),
                    int(row.get("support", 0))
                ))

    print("📝 Saved metrics as text table to: results/test_metrics_table.txt")
