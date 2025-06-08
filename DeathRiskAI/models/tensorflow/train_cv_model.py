import pandas as pd
from collections import defaultdict
from tensorflow_model import TensorflowModel
import os

def load_fold_data(fold: int):
    X_train = pd.read_parquet(f"./../train_test_data/fold{fold}_X_train.parquet")
    Y_train = pd.read_parquet(f"./../train_test_data/fold{fold}_Y_train.parquet").squeeze()
    X_val = pd.read_parquet(f"./../train_test_data/fold{fold}_X_val.parquet")
    Y_val = pd.read_parquet(f"./../train_test_data/fold{fold}_Y_val.parquet").squeeze()
    return X_train, Y_train, X_val, Y_val

def train_on_all_folds(num_folds=5, epochs=20, batch_size=32):
    metrics_summary = defaultdict(list)
    os.makedirs("results", exist_ok=True)

    for fold in range(1, num_folds + 1):
        print(f"\n=== Fold {fold} ===")
        X_train, Y_train, X_val, Y_val = load_fold_data(fold)
        model = TensorflowModel(model_name=f"tf_model_fold{fold}")
        model.train(X_train, Y_train, X_val, Y_val, epochs=epochs, batch_size=batch_size)
        model.save(f"models/tf_model_fold{fold}.keras")
        metrics = model.evaluate(X_val, Y_val)
        for k, v in metrics.items():
            metrics_summary[k].append(v)
        print(f"Fold {fold} metrics: {metrics}")

    print("\n=== Average Metrics ===")
    avg_metrics = {metric: sum(values) / len(values) for metric, values in metrics_summary.items()}
    for metric, avg in avg_metrics.items():
        print(f"{metric}: {avg:.4f}")

    pd.DataFrame(metrics_summary).to_csv("results/cv_metrics.csv", index=False)

if __name__ == "__main__":
    train_on_all_folds()
