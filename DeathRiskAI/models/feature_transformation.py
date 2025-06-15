import argparse
from colorama import init, Fore, Style
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

init()


def load_gold_data() -> pd.DataFrame:
    """
    Load the gold dataset and return it as a pandas DataFrame.
    """
    print("Loading gold dataset...")
    return pd.read_csv("./../data/2_gold/gold.csv")


def feature_transformation(
    df: pd.DataFrame, test_size: float = 0.1, random_state: int = 42
) -> None:
    # Checking for 'death' column in the dataset
    print("Checking for 'death' column in the dataset...")
    if "death" not in df.columns:
        raise ValueError("Column 'death' not found in the dataset.")

    # Splitting dataset into features and target variable
    print("Splitting dataset into features and target variable...")
    X = df.drop(columns=["death"])
    Y = df["death"]

    # Feature Transformation – Standardization
    print("Feature Transformation – Standardizing features...")

    # Separating classes
    X_0 = X[Y == 0]
    X_1 = X[Y == 1]

    # Calculating z-scores for class 0
    # class 1 is not standardized
    z_scores_0 = np.abs((X_0 - X_0.mean()) / X_0.std())
    is_outlier_0 = (z_scores_0 > 2).any(axis=1)

    # Removing outliers from class 0
    X_0_clean = X_0[~is_outlier_0]
    Y_0_clean = pd.Series(0, index=X_0_clean.index, name="death")

    # Not removing outliers from class 1
    X_1_clean = X_1
    Y_1_clean = pd.Series(1, index=X_1_clean.index, name="death")

    # Concatenating both classes
    X = pd.concat([X_0_clean, X_1_clean])
    Y = pd.concat([Y_0_clean, Y_1_clean])

    # Feature Transformation - Normalization
    print("Feature Transformation – Normalizing features...")
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Train-test split
    print(f"Train-test splitting with test_size={test_size}...")
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X_scaled, Y, test_size=test_size, stratify=Y, random_state=random_state
    )

    # Save the training set
    print("Saving training set...")
    X_trainval.to_parquet("./trainval_test_data/X_trainval.parquet", index=False)
    Y_trainval.to_frame(name="death").to_parquet(
        "./trainval_test_data/Y_trainval.parquet", index=False
    )

    # Save the testing set
    print("Saving testing set...")
    X_test.to_parquet("./trainval_test_data/X_test.parquet", index=False)
    Y_test.to_frame(name="death").to_parquet(
        "./trainval_test_data/Y_test.parquet", index=False
    )

    print(Fore.GREEN + "Done." + Style.RESET_ALL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature transformation script for 'DeathRiskAI' project."
    )
    parser.add_argument(
        "--test_size", type=float, default=0.1, help="Testing set size (default: 0.1)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    feature_transformation(
        load_gold_data(),
        test_size=args.test_size,
        random_state=args.random_state,
    )
