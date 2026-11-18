"""Data Processing Module.

Handles sample dataset generation, validation, cleaning, missing-value imputation,
duplicate removal, outlier detection, and social media usage categorization.

Course: Probability and Statistics Using Python (26B16CA212)
Institution: Jaypee Institute of Information Technology, Noida
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Student_ID",
    "Age",
    "Gender",
    "Daily_Social_Media_Hours",
    "Most_Used_Platform",
    "Study_Hours_Per_Day",
    "Sleep_Hours",
    "Academic_Percentage",
    "Concentration_Level",
    "Stress_Level",
]

NUMERICAL_COLUMNS = [
    "Age",
    "Daily_Social_Media_Hours",
    "Study_Hours_Per_Day",
    "Sleep_Hours",
    "Academic_Percentage",
    "Concentration_Level",
    "Stress_Level",
]

VALID_PLATFORMS = [
    "Instagram",
    "YouTube",
    "WhatsApp",
    "Snapchat",
    "Reddit",
    "Twitter/X",
    "LinkedIn",
]

VALID_GENDERS = ["Male", "Female", "Other"]


def categorize_social_media(hours: float) -> str:
    """Categorize daily social media hours into Low, Moderate, or High.

    - Low: Less than 2 hours/day (< 2.0)
    - Moderate: 2 to 4 hours/day (2.0 <= hours <= 4.0)
    - High: Greater than 4 hours/day (> 4.0)
    """
    if pd.isna(hours):
        return "Unknown"
    val = float(hours)
    if val < 2.0:
        return "Low"
    elif val <= 4.0:
        return "Moderate"
    else:
        return "High"


def generate_sample_dataset(n_samples: int = 300, random_state: int = 42) -> pd.DataFrame:
    """Generate a statistically plausible synthetic dataset of student records.

    Note: This is SYNTHETIC sample data generated using reproducible statistical
    distributions for academic demonstration. It does NOT represent real student survey data.

    Parameters:
        n_samples (int): Number of student records to generate (default 300).
        random_state (int): Seed for reproducible pseudo-random generation.

    Returns:
        pd.DataFrame: Generated dataset with realistic inter-variable associations.
    """
    rng = np.random.default_rng(random_state)

    student_ids = [f"STU_{1000 + i + 1}" for i in range(n_samples)]
    ages = rng.choice([18, 19, 20, 21, 22, 23], size=n_samples, p=[0.20, 0.35, 0.25, 0.12, 0.05, 0.03])
    genders = rng.choice(VALID_GENDERS, size=n_samples, p=[0.52, 0.45, 0.03])

    platform_weights = [0.38, 0.26, 0.15, 0.10, 0.05, 0.04, 0.02]
    platforms = rng.choice(VALID_PLATFORMS, size=n_samples, p=platform_weights)

    # Base social media hours: Log-normal skewed distribution typical of screen time
    base_sm = rng.gamma(shape=3.5, scale=0.9, size=n_samples)
    sm_hours = np.clip(np.round(base_sm, 1), 0.5, 9.0)

    # Study hours: moderately negatively correlated with social media, positive base
    # Study = 7.0 - 0.45 * SM + noise
    study_noise = rng.normal(0, 0.9, size=n_samples)
    study_hours = 6.2 - (0.42 * sm_hours) + study_noise
    study_hours = np.clip(np.round(study_hours, 1), 1.0, 8.5)

    # Sleep hours: mildly negatively correlated with social media
    # Sleep = 8.5 - 0.30 * SM + noise
    sleep_noise = rng.normal(0, 0.75, size=n_samples)
    sleep_hours = 8.2 - (0.28 * sm_hours) + sleep_noise
    sleep_hours = np.clip(np.round(sleep_hours, 1), 4.5, 9.5)

    # Stress level (1 to 10): positively correlated with social media, negatively with sleep
    stress_raw = 3.2 + (0.55 * sm_hours) - (0.35 * (sleep_hours - 6.5)) + rng.normal(0, 1.1, size=n_samples)
    stress_level = np.clip(np.round(stress_raw).astype(int), 1, 10)

    # Concentration level (1 to 10): negatively correlated with social media & stress
    conc_raw = 8.5 - (0.45 * sm_hours) - (0.25 * stress_level) + rng.normal(0, 1.0, size=n_samples)
    concentration_level = np.clip(np.round(conc_raw).astype(int), 1, 10)

    # Academic percentage: driven by study hours (+), concentration (+), sleep (+), SM hours (-)
    # Realistic collegiate range ~ 48% to 96%
    academic_raw = (
        48.0
        + (3.8 * study_hours)
        - (1.9 * sm_hours)
        + (1.2 * sleep_hours)
        + (1.4 * concentration_level)
        - (0.8 * stress_level)
        + rng.normal(0, 3.5, size=n_samples)
    )
    academic_pct = np.clip(np.round(academic_raw, 2), 45.0, 98.0)

    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Age": ages,
        "Gender": genders,
        "Daily_Social_Media_Hours": sm_hours,
        "Most_Used_Platform": platforms,
        "Study_Hours_Per_Day": study_hours,
        "Sleep_Hours": sleep_hours,
        "Academic_Percentage": academic_pct,
        "Concentration_Level": concentration_level,
        "Stress_Level": stress_level,
    })

    df["Usage_Category"] = df["Daily_Social_Media_Hours"].apply(categorize_social_media)
    return df


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    """Validate user-uploaded or active dataset against academic schema.

    Returns:
        Tuple[bool, List[str], List[str]]: (is_valid, error_list, warning_list)
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or df.empty:
        return False, ["The dataset is completely empty or null."], []

    # Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")

    # Range and type checks on available columns
    if "Daily_Social_Media_Hours" in df.columns:
        invalid_sm = df[df["Daily_Social_Media_Hours"] < 0]
        if not invalid_sm.empty:
            errors.append(f"Found {len(invalid_sm)} records with negative Daily_Social_Media_Hours.")
        high_sm = df[df["Daily_Social_Media_Hours"] > 24]
        if not high_sm.empty:
            warnings.append(f"Found {len(high_sm)} records with Daily_Social_Media_Hours > 24.")

    if "Academic_Percentage" in df.columns:
        invalid_pct = df[(df["Academic_Percentage"] < 0) | (df["Academic_Percentage"] > 100)]
        if not invalid_pct.empty:
            errors.append(f"Found {len(invalid_pct)} records with Academic_Percentage outside [0, 100].")

    if "Study_Hours_Per_Day" in df.columns:
        invalid_study = df[df["Study_Hours_Per_Day"] < 0]
        if not invalid_study.empty:
            errors.append(f"Found {len(invalid_study)} records with negative Study_Hours_Per_Day.")

    if "Sleep_Hours" in df.columns:
        invalid_sleep = df[df["Sleep_Hours"] < 0]
        if not invalid_sleep.empty:
            errors.append(f"Found {len(invalid_sleep)} records with negative Sleep_Hours.")

    if "Stress_Level" in df.columns:
        out_of_range_stress = df[(df["Stress_Level"] < 1) | (df["Stress_Level"] > 10)]
        if not out_of_range_stress.empty:
            warnings.append(f"Found {len(out_of_range_stress)} records with Stress_Level outside 1-10.")

    if "Concentration_Level" in df.columns:
        out_of_range_conc = df[(df["Concentration_Level"] < 1) | (df["Concentration_Level"] > 10)]
        if not out_of_range_conc.empty:
            warnings.append(f"Found {len(out_of_range_conc)} records with Concentration_Level outside 1-10.")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def clean_dataset(
    df: pd.DataFrame,
    missing_strategy: str = "mean",
    remove_duplicates: bool = True,
    drop_out_of_range: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean and preprocess the dataset transparently without silently discarding data.

    Parameters:
        df: Input DataFrame
        missing_strategy: Imputation method for numerical columns: 'mean', 'median', 'mode', or 'drop'
        remove_duplicates: Whether to drop duplicate rows
        drop_out_of_range: Whether to drop mathematically impossible records (e.g. negative hours)

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and detailed audit report.
    """
    clean_df = df.copy()
    audit: Dict[str, Any] = {
        "initial_rows": len(df),
        "duplicates_removed": 0,
        "missing_values_before": int(df.isnull().sum().sum()),
        "missing_by_column": df.isnull().sum().to_dict(),
        "imputation_strategy": missing_strategy,
        "imputed_columns": [],
        "invalid_rows_removed": 0,
    }

    # 1. Handle Duplicates
    if remove_duplicates:
        dupes_count = int(clean_df.duplicated().sum())
        if dupes_count > 0:
            clean_df = clean_df.drop_duplicates()
            audit["duplicates_removed"] = dupes_count

    # 2. Handle Invalid Values if requested
    if drop_out_of_range:
        mask = pd.Series(True, index=clean_df.index)
        if "Daily_Social_Media_Hours" in clean_df.columns:
            mask = mask & (clean_df["Daily_Social_Media_Hours"] >= 0) & (clean_df["Daily_Social_Media_Hours"] <= 24)
        if "Academic_Percentage" in clean_df.columns:
            mask = mask & (clean_df["Academic_Percentage"] >= 0) & (clean_df["Academic_Percentage"] <= 100)
        if "Study_Hours_Per_Day" in clean_df.columns:
            mask = mask & (clean_df["Study_Hours_Per_Day"] >= 0) & (clean_df["Study_Hours_Per_Day"] <= 24)
        if "Sleep_Hours" in clean_df.columns:
            mask = mask & (clean_df["Sleep_Hours"] >= 0) & (clean_df["Sleep_Hours"] <= 24)

        invalid_count = int((~mask).sum())
        clean_df = clean_df[mask]
        audit["invalid_rows_removed"] = invalid_count

    # 3. Handle Missing Values
    if missing_strategy == "drop":
        clean_df = clean_df.dropna()
    else:
        for col in clean_df.columns:
            if clean_df[col].isnull().any():
                audit["imputed_columns"].append(col)
                if pd.api.types.is_numeric_dtype(clean_df[col]):
                    if missing_strategy == "mean":
                        fill_val = clean_df[col].mean()
                    elif missing_strategy == "median":
                        fill_val = clean_df[col].median()
                    elif missing_strategy == "mode":
                        fill_val = clean_df[col].mode().iloc[0] if not clean_df[col].mode().empty else 0
                    else:
                        fill_val = clean_df[col].mean()
                    clean_df[col] = clean_df[col].fillna(fill_val)
                else:
                    # Categorical mode fill
                    mode_val = clean_df[col].mode().iloc[0] if not clean_df[col].mode().empty else "Unknown"
                    clean_df[col] = clean_df[col].fillna(mode_val)

    # 4. Guarantee Usage Category is present
    if "Daily_Social_Media_Hours" in clean_df.columns:
        clean_df["Usage_Category"] = clean_df["Daily_Social_Media_Hours"].apply(categorize_social_media)

    audit["final_rows"] = len(clean_df)
    audit["missing_values_after"] = int(clean_df.isnull().sum().sum())
    return clean_df, audit


def detect_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: float = 1.5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Detect statistical outliers in a numerical column using IQR or Z-score method.

    Parameters:
        df: DataFrame
        column: Column to inspect
        method: 'iqr' (default) or 'zscore'
        threshold: 1.5 for IQR rule, or 3.0 for standard Z-score rule

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Outlier rows and calculation metadata.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return pd.DataFrame(), {"count": 0}

    if method.lower() == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (threshold * iqr)
        upper_bound = q3 + (threshold * iqr)

        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        outliers = df[outlier_mask]

        info = {
            "method": "IQR",
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": len(outliers),
            "outlier_percentage": round((len(outliers) / len(df)) * 100, 2),
        }
        return outliers, info

    elif method.lower() == "zscore":
        mean = series.mean()
        std = series.std(ddof=1)
        if std == 0 or np.isnan(std):
            return pd.DataFrame(), {"method": "Z-Score", "outlier_count": 0}

        z_scores = np.abs((df[column] - mean) / std)
        outliers = df[z_scores > threshold]

        info = {
            "method": "Z-Score",
            "mean": float(mean),
            "std": float(std),
            "threshold": float(threshold),
            "outlier_count": len(outliers),
            "outlier_percentage": round((len(outliers) / len(df)) * 100, 2),
        }
        return outliers, info

    else:
        raise ValueError("Method must be 'iqr' or 'zscore'.")


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate high-level dataset health metrics."""
    return {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numerical_columns": [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
        "categorical_columns": [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])],
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
    }
