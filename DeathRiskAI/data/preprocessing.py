import pandas as pd
from colorama import init, Fore, Style

init()


def bronze_to_silver(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bronze to Silver DataFrame transformation.
    This function takes a DataFrame and performs the following transformations:

    - Calculate missing values depending on other columns
    - Drop unnecessary columns
    - Drop rows with missing values in key columns
    - Rename columns for consistency
    - Reorder columns to match the desired output

    Parameters:
    df (pd.DataFrame): The input DataFrame to be transformed.
    Returns:
    pd.DataFrame: The transformed DataFrame.
    """

    input_columns_len = len(df.columns)
    input_rows_len = len(df)

    # Calculate missing values depending on other columns
    print(
        "\nTransforming DataFrame - Calculate missing values depending on other columns..."
    )

    # If bmi is missing, calculate it using the formula: weight / (height / 100) ** 2
    print("Calculate BMI if missing...")
    df["bmi"] = df.apply(
        lambda row: (
            row["weight"] / ((row["height"] / 100) ** 2)
            if pd.isnull(row["bmi"])
            and not pd.isnull(row["weight"])
            and not pd.isnull(row["height"])
            else row["bmi"]
        ),
        axis=1,
    )

    # Drop unnecessary columns
    print("Transforming DataFrame - Drop unnecessary columns...")
    print(
        "Number of columns before dropping: "
        + Fore.RED
        + str(input_columns_len)
        + Style.RESET_ALL
    )

    # "???" means that the column is not used in the model but may be useful for future analysis
    df = df.drop(
        columns=[
            "encounter_id",
            "hospital_id",
            "patient_id",
            "ethnicity",  # removed due to racial classification
            "height",  # invloved in BMI value
            "hospital_admit_source",
            "icu_admit_source",  # ???
            "icu_id",
            "icu_stay_type",  # ???
            "icu_type",  # ???
            "pre_icu_los_days",  # ???
            "readmission_status",  # ???
            "weight",  # invloved in BMI value
            "apache_2_diagnosis",  # disqualification, result indicates probability of death
            "apache_3j_diagnosis",  # disqualification, result indicates probability of death
            "gcs_eyes_apache",  # ???
            "gcs_motor_apache",  # ???
            "gcs_unable_apache",  # ???
            "gcs_verbal_apache",  # ???
            "intubated_apache",  # ???
            "resprate_apache",  # ???
            "ventilated_apache",  # ???
            "d1_diasbp_invasive_max",
            "d1_diasbp_invasive_min",
            "d1_diasbp_noninvasive_max",
            "d1_diasbp_noninvasive_min",
            "d1_mbp_invasive_max",
            "d1_mbp_invasive_min",
            "d1_mbp_noninvasive_max",
            "d1_mbp_noninvasive_min",
            "d1_resprate_max",  # ???
            "d1_resprate_min",  # ???
            "d1_sysbp_invasive_max",
            "d1_sysbp_invasive_min",
            "d1_sysbp_noninvasive_min",
            "d1_sysbp_noninvasive_max",
            "h1_diasbp_invasive_max",
            "h1_diasbp_invasive_min",
            "h1_diasbp_noninvasive_max",
            "h1_diasbp_noninvasive_min",
            "h1_mbp_invasive_max",
            "h1_mbp_invasive_min",
            "h1_mbp_noninvasive_max",
            "h1_mbp_noninvasive_min",
            "h1_resprate_max",  # ???
            "h1_resprate_min",  # ???
            "h1_sysbp_invasive_max",
            "h1_sysbp_invasive_min",
            "h1_sysbp_noninvasive_min",
            "h1_sysbp_noninvasive_max",
            "apache_4a_hospital_death_prob",  # ???
            "apache_4a_icu_death_prob",  # ???
            "apache_3j_bodysystem",  # disqualification, result indicates probability of death
            "apache_2_bodysystem",  # disqualification, result indicates probability of death
        ]
    )

    print(
        "Number of columns after dropping: "
        + Fore.RED
        + str(len(df.columns))
        + Style.RESET_ALL
    )

    # Drop rows with missing values in key columns
    print("Transforming DataFrame - Drop rows with missing values in key columns...")
    print(
        "Number of rows before dropping: "
        + Fore.RED
        + str(input_rows_len)
        + Style.RESET_ALL
    )

    df = df.dropna(subset=["gender", "age", "bmi"]).copy()

    print("Number of rows after dropping: " + Fore.RED + str(len(df)) + Style.RESET_ALL)

    # Rename columns for consistency
    print("Transforming DataFrame - Rename columns for consistency...")

    columns_to_rename = {
        "hospital_death": "death",
        "albumin_apache": "albumin",
        "arf_apache": "arf",
        "bilirubin_apache": "bilirubin",
        "bun_apache": "bun",
        "creatinine_apache": "creatinine",
        "fio2_apache": "fio2",
        "glucose_apache": "glucose",
        "heart_rate_apache": "heart_rate",
        "hematocrit_apache": "hematocrit",
        "map_apache": "map",
        "paco2_apache": "paco2",
        "paco2_for_ph_apache": "paco2_for_ph",
        "pao2_apache": "pao2",
        "ph_apache": "ph",
        "sodium_apache": "sodium",
        "temp_apache": "temp",
        "urineoutput_apache": "urineoutput",
        "ventilated_apache": "ventilated",
        "wbc_apache": "wbc",
    }
    df = df.rename(columns=columns_to_rename)

    print(
        "Number of renamed columns: "
        + Fore.RED
        + str(len(columns_to_rename))
        + Style.RESET_ALL
    )

    # Reorder columns to match the desired output
    print("Transforming DataFrame - Reorder columns to match the desired output...")

    df = df[
        [
            # Outcome (y)
            "death",
            # Basic information of the patient (x)
            "gender",
            "age",
            "bmi",
            # Comorbidities (x)
            "aids",
            "cirrhosis",
            "diabetes_mellitus",
            "hepatic_failure",
            "immunosuppression",
            "leukemia",
            "lymphoma",
            "solid_tumor_with_metastasis",
            # Other information of the patient (x)
            "arf",
            "elective_surgery",
            # Vital signs (x)
            "albumin",
            "h1_albumin_min",
            "h1_albumin_max",
            "d1_albumin_min",
            "d1_albumin_max",
            "bilirubin",
            "h1_bilirubin_min",
            "h1_bilirubin_max",
            "d1_bilirubin_min",
            "d1_bilirubin_max",
            "bun",
            "h1_bun_min",
            "h1_bun_max",
            "d1_bun_min",
            "d1_bun_max",
            "h1_calcium_min",
            "h1_calcium_max",
            "d1_calcium_min",
            "d1_calcium_max",
            "creatinine",
            "h1_creatinine_min",
            "h1_creatinine_max",
            "d1_creatinine_min",
            "d1_creatinine_max",
            "glucose",
            "h1_glucose_min",
            "h1_glucose_max",
            "d1_glucose_min",
            "d1_glucose_max",
            "h1_hco3_min",
            "h1_hco3_max",
            "d1_hco3_min",
            "d1_hco3_max",
            "heart_rate",
            "h1_heartrate_min",
            "h1_heartrate_max",
            "d1_heartrate_min",
            "d1_heartrate_max",
            "h1_hemaglobin_min",
            "h1_hemaglobin_max",
            "d1_hemaglobin_min",
            "d1_hemaglobin_max",
            "hematocrit",
            "h1_hematocrit_min",
            "h1_hematocrit_max",
            "d1_hematocrit_min",
            "d1_hematocrit_max",
            "h1_inr_min",
            "h1_inr_max",
            "d1_inr_min",
            "d1_inr_max",
            "h1_lactate_min",
            "h1_lactate_max",
            "d1_lactate_min",
            "d1_lactate_max",
            "map",
            "h1_mbp_min",
            "h1_mbp_max",
            "d1_mbp_min",
            "d1_mbp_max",
            "ph",
            "h1_arterial_ph_min",
            "h1_arterial_ph_max",
            "d1_arterial_ph_min",
            "d1_arterial_ph_max",
            "h1_platelets_min",
            "h1_platelets_max",
            "d1_platelets_min",
            "d1_platelets_max",
            "h1_potassium_min",
            "h1_potassium_max",
            "d1_potassium_min",
            "d1_potassium_max",
            "sodium",
            "h1_sodium_min",
            "h1_sodium_max",
            "d1_sodium_min",
            "d1_sodium_max",
            "h1_spo2_min",
            "h1_spo2_max",
            "d1_spo2_min",
            "d1_spo2_max",
            "temp",
            "h1_temp_min",
            "h1_temp_max",
            "d1_temp_min",
            "d1_temp_max",
            "wbc",
            "h1_wbc_min",
            "h1_wbc_max",
            "d1_wbc_min",
            "d1_wbc_max",
            # Other signs (x)
            "h1_arterial_pco2_min",
            "h1_arterial_pco2_max",
            "d1_arterial_pco2_min",
            "d1_arterial_pco2_max",
            "h1_arterial_po2_min",
            "h1_arterial_po2_max",
            "d1_arterial_po2_min",
            "d1_arterial_po2_max",
            "h1_diasbp_min",
            "h1_diasbp_max",
            "d1_diasbp_min",
            "d1_diasbp_max",
            "fio2",
            "pao2",
            "h1_pao2fio2ratio_min",
            "h1_pao2fio2ratio_max",
            "d1_pao2fio2ratio_min",
            "d1_pao2fio2ratio_max",
            "h1_sysbp_min",
            "h1_sysbp_max",
            "d1_sysbp_min",
            "d1_sysbp_max",
            "urineoutput",
        ]
    ]

    return df


def silver_to_gold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Silver to Gold DataFrame transformation.
    This function takes a DataFrame and performs the following transformations:

    - Fill missing values in specific columns
    - Normalize floating-point numerical columns
    - Encode categorical columns

    Parameters:
    df (pd.DataFrame): The input DataFrame to be transformed.
    Returns:
    pd.DataFrame: The transformed DataFrame.
    """

    # Fill missing values in specific columns
    print("Transforming DataFrame - Fill missing values in specific columns...")

    # With 0
    print("Filling missing values with 0...")
    columns_to_fill = [
        "aids",
        "cirrhosis",
        "diabetes_mellitus",
        "hepatic_failure",
        "immunosuppression",
        "leukemia",
        "lymphoma",
        "solid_tumor_with_metastasis",
        "arf",
        "elective_surgery",
    ]
    df[columns_to_fill] = df[columns_to_fill].fillna(0)

    # With mean
    print("Filling missing values with mean of top 10 values...")
    columns_to_fill_mean = [
        "albumin",
        "h1_albumin_min",
        "h1_albumin_max",
        "d1_albumin_min",
        "d1_albumin_max",
        "bilirubin",
        "h1_bilirubin_min",
        "h1_bilirubin_max",
        "d1_bilirubin_min",
        "d1_bilirubin_max",
        "bun",
        "h1_bun_min",
        "h1_bun_max",
        "d1_bun_min",
        "d1_bun_max",
        "h1_calcium_min",
        "h1_calcium_max",
        "d1_calcium_min",
        "d1_calcium_max",
        "creatinine",
        "h1_creatinine_min",
        "h1_creatinine_max",
        "d1_creatinine_min",
        "d1_creatinine_max",
        "glucose",
        "h1_glucose_min",
        "h1_glucose_max",
        "d1_glucose_min",
        "d1_glucose_max",
        "h1_hco3_min",
        "h1_hco3_max",
        "d1_hco3_min",
        "d1_hco3_max",
        "heart_rate",
        "h1_heartrate_min",
        "h1_heartrate_max",
        "d1_heartrate_min",
        "d1_heartrate_max",
        "h1_hemaglobin_min",
        "h1_hemaglobin_max",
        "d1_hemaglobin_min",
        "d1_hemaglobin_max",
        "hematocrit",
        "h1_hematocrit_min",
        "h1_hematocrit_max",
        "d1_hematocrit_min",
        "d1_hematocrit_max",
        "h1_inr_min",
        "h1_inr_max",
        "d1_inr_min",
        "d1_inr_max",
        "h1_lactate_min",
        "h1_lactate_max",
        "d1_lactate_min",
        "d1_lactate_max",
        "map",
        "h1_mbp_min",
        "h1_mbp_max",
        "d1_mbp_min",
        "d1_mbp_max",
        "ph",
        "h1_arterial_ph_min",
        "h1_arterial_ph_max",
        "d1_arterial_ph_min",
        "d1_arterial_ph_max",
        "h1_platelets_min",
        "h1_platelets_max",
        "d1_platelets_min",
        "d1_platelets_max",
        "h1_potassium_min",
        "h1_potassium_max",
        "d1_potassium_min",
        "d1_potassium_max",
        "sodium",
        "h1_sodium_min",
        "h1_sodium_max",
        "d1_sodium_min",
        "d1_sodium_max",
        "h1_spo2_min",
        "h1_spo2_max",
        "d1_spo2_min",
        "d1_spo2_max",
        "temp",
        "h1_temp_min",
        "h1_temp_max",
        "d1_temp_min",
        "d1_temp_max",
        "wbc",
        "h1_wbc_min",
        "h1_wbc_max",
        "d1_wbc_min",
        "d1_wbc_max",
        "h1_arterial_pco2_min",
        "h1_arterial_pco2_max",
        "d1_arterial_pco2_min",
        "d1_arterial_pco2_max",
        "h1_arterial_po2_min",
        "h1_arterial_po2_max",
        "d1_arterial_po2_min",
        "d1_arterial_po2_max",
        "h1_diasbp_min",
        "h1_diasbp_max",
        "d1_diasbp_min",
        "d1_diasbp_max",
        "fio2",
        "pao2",
        "h1_pao2fio2ratio_min",
        "h1_pao2fio2ratio_max",
        "d1_pao2fio2ratio_min",
        "d1_pao2fio2ratio_max",
        "h1_sysbp_min",
        "h1_sysbp_max",
        "d1_sysbp_min",
        "d1_sysbp_max",
        "urineoutput",
    ]
    for column in columns_to_fill_mean:
        if column in df.columns:
            top_10 = df[column].value_counts().head(10).index
            top_10_values = df[df[column].isin(top_10)][column]
            mean_top_10 = round(top_10_values.mean(), 2)
            df[column] = df[column].fillna(mean_top_10)

    # Normalize floating-point numerical columns
    # Round floating-point numerical columns to 2 decimal places
    print("Transforming DataFrame - Normalize floating-point numerical columns...")
    float_columns = df.select_dtypes(include=["float64"]).columns
    for column in float_columns:
        df[column] = df[column].round(2)

    # Encode categorical columns
    print("Transforming DataFrame - Encode categorical columns...")
    categorical_columns = df.select_dtypes(include=["object"]).columns
    for column in categorical_columns:
        if column == "gender":
            print("Encoding gender column...")
            df[column] = df[column].map({"M": 0, "F": 1})
        else:
            print(f"Encoding {str(column)} column...")
            df[column] = df[column].astype("category").cat.codes

    return df


if __name__ == "__main__":
    print("Choose a transformation:")
    print("1. Bronze to Silver")
    print("2. Silver to Gold")
    choice = input("Enter your choice (1 or 2): ")
    if choice == "1":
        with open("./0_bronze/bronze.csv", "r") as file:
            df = pd.read_csv(file)
            df = bronze_to_silver(df)
            df.to_csv("./1_silver/silver.csv", index=False)
        print(
            Fore.GREEN
            + "Transformation complete. Silver DataFrame saved."
            + Style.RESET_ALL
        )
    elif choice == "2":
        with open("./1_silver/silver.csv", "r") as file:
            df = pd.read_csv(file)
            df = silver_to_gold(df)
            df.to_csv("./2_gold/gold.csv", index=False)
        print(
            Fore.GREEN
            + "Transformation complete. Gold DataFrame saved."
            + Style.RESET_ALL
        )
    else:
        print(Fore.RED + "Invalid choice. Exiting." + Style.RESET_ALL)
