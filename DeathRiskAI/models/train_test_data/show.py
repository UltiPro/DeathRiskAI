import sys
import os
import pandas as pd


def show_parquet(file_path):
    if not os.path.isfile(file_path):
        print(f"File '{file_path}' does not exist.")
        return
    try:
        df = pd.read_parquet(file_path)
        print(df)
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python show.py <parquet_file>")
    else:
        show_parquet(sys.argv[1])
