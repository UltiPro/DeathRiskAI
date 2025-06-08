import pandas as pd
from collections import defaultdict

from tensorflow_model import TensorflowModel


def load_fold_data(fold: int):
    X_train = pd.read_parquet(f"./../train_test_data/fold{fold}_X_train.parquet")
    Y_train = pd.read_parquet(
        f"./../train_test_data/fold{fold}_Y_train.parquet"
    ).squeeze()
    X_val = pd.read_parquet(f"./../train_test_data/fold{fold}_X_val.parquet")
    Y_val = pd.read_parquet(f"./../train_test_data/fold{fold}_Y_val.parquet").squeeze()
    return X_train, Y_train, X_val, Y_val


def train_on_all_folds(num_folds=5, epochs=10, batch_size=32):
    metrics_summary = defaultdict(list)

    for fold in range(1, num_folds + 1):
        print(f"\n=== Fold {fold} ===")

        # Load data
        X_train, Y_train, X_val, Y_val = load_fold_data(fold)

        # Train model
        model = TensorflowModel(model_name=f"tf_model_fold{fold}")
        model.train(
            X_train, Y_train, X_val, Y_val, epochs=epochs, batch_size=batch_size
        )
        model.save(f"instances/{model.model_name}.h5")

        # Evaluate
        metrics = model.evaluate(X_val, Y_val)
        for k, v in metrics.items():
            metrics_summary[k].append(v)
        print(f"Fold {fold} metrics: {metrics}")

    # Average metrics across folds
    print("\n=== Average Metrics ===")
    for metric, values in metrics_summary.items():
        avg = sum(values) / len(values)
        print(f"{metric}: {avg:.4f}")


if __name__ == "__main__":
    train_on_all_folds(num_folds=5, epochs=10, batch_size=32)
