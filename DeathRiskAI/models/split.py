import sys
from colorama import init, Fore, Style
import pickle
import pandas as pd
from sklearn.model_selection import StratifiedKFold

init()


def load_gold_data() -> pd.DataFrame:
    """
    Load the gold dataset and return it as a pandas DataFrame.
    """
    print("Loading gold dataset...")
    return pd.read_csv("./../data/2_gold/gold.csv")


def create_folds(df: pd.DataFrame, n_splits: int = 5):
    # Check if 'death' column exists
    print("Checking for 'death' column in the dataset...")
    if "death" not in df.columns:
        raise ValueError("Column 'death' not found in the dataset.")

    # Input data - split into features and target variable
    print("Splitting dataset into features and target variable...")
    X = df.drop(columns=["death"])
    y = df["death"]

    # Create Stratified K-Folds
    print(f"Creating Stratified K-Folds with n_splits={n_splits}...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        folds.append(
            {"X_train": X_train, "X_val": X_val, "y_train": y_train, "y_val": y_val}
        )

    # Save folds to a pickle file
    print(f"Saving folds to 'folds_{n_splits}_split.pkl'...")
    with open(f"folds_{n_splits}_split.pkl", "wb") as f:
        pickle.dump(folds, f)

    print(Fore.GREEN + "Done." + Style.RESET_ALL)


if __name__ == "__main__":
    n_splits = 5  # default n_splits
    try:
        n_splits = int(sys.argv[1])
    except (ValueError, IndexError):
        print(
            Fore.RED
            + "Invalid input for n_splits. Using default value of 5."
            + Style.RESET_ALL
        )
    create_folds(load_gold_data(), n_splits)
