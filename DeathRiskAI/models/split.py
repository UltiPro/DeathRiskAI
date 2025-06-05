import argparse
from colorama import init, Fore, Style
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

init()


def load_gold_data() -> pd.DataFrame:
    """
    Load the gold dataset and return it as a pandas DataFrame.
    """
    print("Loading gold dataset...")
    return pd.read_csv("./../data/2_gold/gold.csv")


def create_folds(
    df: pd.DataFrame, test_size: float = 0.1, n_splits: int = 5, random_state: int = 42
) -> None:
    # Checking for 'death' column in the dataset
    print("Checking for 'death' column in the dataset...")
    if "death" not in df.columns:
        raise ValueError("Column 'death' not found in the dataset.")

    # Splitting dataset into features and target variable
    print("Splitting dataset into features and target variable...")
    X = df.drop(columns=["death"])
    Y = df["death"]

    # Train-test split
    print(f"Train-test splitting with test_size={test_size}...")
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X, Y, test_size=test_size, stratify=Y, random_state=random_state
    )

    # Create Stratified K-Folds
    print(f"Creating Stratified K-Folds with n_splits={n_splits}...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Saving the train-validation folds
    print("Saving train-validation folds...")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_trainval, Y_trainval)):
        X_train, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
        Y_train, Y_val = Y_trainval.iloc[train_idx], Y_trainval.iloc[val_idx]

        fold_idx += 1

        # Saving the folds to parquet files
        print(f"Saving fold {fold_idx} to parquet files...")
        X_train.to_parquet(
            f"./train_test_data/fold{fold_idx}_X_train.parquet", index=False
        )
        X_val.to_parquet(f"./train_test_data/fold{fold_idx}_X_val.parquet", index=False)
        Y_train.to_frame(name="death").to_parquet(
            f"./train_test_data/fold{fold_idx}_Y_train.parquet", index=False
        )
        Y_val.to_frame(name="death").to_parquet(
            f"./train_test_data/fold{fold_idx}_Y_val.parquet", index=False
        )

    # Saving the test set
    print("Saving test set...")
    X_test.to_parquet("./train_test_data/X_test.parquet", index=False)
    Y_test.to_frame(name="death").to_parquet(
        "./train_test_data/Y_test.parquet", index=False
    )

    print(Fore.GREEN + "Done." + Style.RESET_ALL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create stratified folds from gold dataset."
    )
    parser.add_argument(
        "--test_size", type=float, default=0.1, help="Test set size (default: 0.1)"
    )
    parser.add_argument(
        "--n_splits", type=int, default=5, help="Number of folds (default: 5)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    create_folds(
        load_gold_data(),
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.random_state,
    )
