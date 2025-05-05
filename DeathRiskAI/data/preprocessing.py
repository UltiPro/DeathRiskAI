import pandas as pd
from colorama import Fore, Style, init

init()

def bronze_to_silver(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bronze to Silver DataFrame transformation.
    This function takes a DataFrame and performs the following transformations:

    - Drop unnecessary columns
    - Rename columns for consistency
    # to do
    - Reorder columns to match the desired output
    # to do

    Parameters:
    df (pd.DataFrame): The input DataFrame to be transformed.
    Returns:
    pd.DataFrame: The transformed DataFrame.
    """

    input_columns_len = len(df.columns)

    # Drop unnecessary columns
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

    print(f"Number of columns before dropping: {input_columns_len}")
    print(f"Number of columns after dropping: {len(df.columns)}")

    # Rename columns for consistency
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

    print(f"Number of renamed columns: {len(columns_to_rename)}")

    # Reorder columns to match the desired output
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
            "elective_surgery",
            # Vital signs (x)
            "albumin",
            "h1_albumin_min",
            "h1_albumin_max",
            "d1_albumin_min",
            "d1_albumin_max",
            "arf",
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

            "fio2",
            "map",
            "paco2",
            "paco2_for_ph",
            "pao2",
            "ph",
            "urineoutput",

            "h1_diasbp_min",
            "h1_diasbp_max",
            "d1_diasbp_min",
            "d1_diasbp_max",
            "heart_rate",
            "h1_heartrate_min",
            "h1_heartrate_max",
            "d1_heartrate_min",
            "d1_heartrate_max",
            "h1_mbp_min",
            "h1_mbp_max",
            "d1_mbp_min",
            "d1_mbp_max",
            "h1_spo2_min",
            "h1_spo2_max",
            "d1_spo2_min",
            "d1_spo2_max",
            "h1_sysbp_min",
            "h1_sysbp_max",
            "d1_sysbp_min",
            "d1_sysbp_max",
            "h1_arterial_pco2_min",
            "h1_arterial_pco2_max",
            "d1_arterial_pco2_min",
            "d1_arterial_pco2_max",
            "h1_arterial_ph_min",
            "h1_arterial_ph_max",
            "d1_arterial_ph_min",
            "d1_arterial_ph_max",
            "h1_arterial_po2_min",
            "h1_arterial_po2_max",
            "d1_arterial_po2_min",
            "d1_arterial_po2_max",
            "h1_pao2fio2ratio_min",
            "h1_pao2fio2ratio_max",
            "d1_pao2fio2ratio_min",
            "d1_pao2fio2ratio_max",
        ]
    ]

    print("Done transforming DataFrame")

    return df


def silver_to_gold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Silver to Gold DataFrame transformation.
    This function takes a DataFrame and performs the following transformations:
    - to do

    """
    pass


with open("./0_bronze/bronze.csv", "r") as file:
    df = pd.read_csv(file)
    df = bronze_to_silver(df)
    # save the transformed DataFrame to a new CSV file
    df.to_csv("./1_silver/silver.csv", index=False)
