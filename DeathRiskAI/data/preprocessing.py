import pandas as pd


def bronze_to_silver(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bronze to Silver DataFrame transformation.
    This function takes a DataFrame and performs the following transformations:

    - Drop unnecessary columns and "rasist" columns

    - Rename columns for consistency ## to do
    - Handle missing values ## to do

    - Return the transformed DataFrame
    Parameters:
    df (pd.DataFrame): The input DataFrame to be transformed.
    Returns:
    pd.DataFrame: The transformed DataFrame.
    """

    # Drop unnecessary columns and "rasist" columns
    df = df.drop(
        columns=[
            "encounter_id",
            "hospital_id",
            "patient_id",
            "ethnicity",
            "icu_id",
            "apache_2_diagnosis",
            "apache_3j_diagnosis",
            "apache_4a_hospital_death_prob",
            "apache_4a_icu_death_prob",
            "apache_3j_bodysystem",
            "apache_2_bodysystem",
        ]
    )

    # Rename columns for consistency
    df = df.rename(
        columns={
            "hospital_death": "death",
            "arf_apache": "arf",
            "bilirubin_apache": "bilirubin",
            "bun_apache": "bun",
            "creatinine_apache": "creatinine",
            "fio2_apache": "fio2",
            "gcs_eyes_apache": "gcs_eyes",
            "gcs_motor_apache": "gcs_motor",


            "gcs_verbal_apache": "gcs_verbal",
            "heart_rate_apache": "heart_rate",
            "intubated_apache": "intubated",
            "map_apache": "map",
        }
    )

    # Handle missing values
    # df = df.fillna(0)  # Example: Fill missing values with 0

    return df


def silver_to_gold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Silver to Gold DataFrame transformation.
    This function takes a DataFrame and performs the following transformations:
    - to do

    """
    pass


with open("./0_bronze/bronze.csv", "r") as file:
    pass
