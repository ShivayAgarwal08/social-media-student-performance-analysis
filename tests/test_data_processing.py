"""Tests for Data Processing Module.

Course: Probability and Statistics Using Python (26B16CA212)
Institution: Jaypee Institute of Information Technology, Noida
"""

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    REQUIRED_COLUMNS,
    categorize_social_media,
    clean_dataset,
    detect_outliers,
    generate_sample_dataset,
    get_dataset_summary,
    validate_dataset,
)


def test_categorize_social_media():
    assert categorize_social_media(0.5) == "Low"
    assert categorize_social_media(1.9) == "Low"
    assert categorize_social_media(2.0) == "Moderate"
    assert categorize_social_media(3.5) == "Moderate"
    assert categorize_social_media(4.0) == "Moderate"
    assert categorize_social_media(4.1) == "High"
    assert categorize_social_media(8.0) == "High"
    assert categorize_social_media(np.nan) == "Unknown"


def test_generate_sample_dataset():
    df = generate_sample_dataset(n_samples=100, random_state=42)
    assert len(df) == 100
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    assert "Usage_Category" in df.columns

    # Verify realistic bounds
    assert (df["Daily_Social_Media_Hours"] >= 0).all()
    assert (df["Academic_Percentage"] >= 0).all() and (df["Academic_Percentage"] <= 100).all()
    assert (df["Study_Hours_Per_Day"] >= 0).all()
    assert (df["Sleep_Hours"] >= 0).all()
    assert (df["Stress_Level"] >= 1).all() and (df["Stress_Level"] <= 10).all()
    assert (df["Concentration_Level"] >= 1).all() and (df["Concentration_Level"] <= 10).all()


def test_validate_dataset_valid():
    df = generate_sample_dataset(n_samples=50, random_state=123)
    is_valid, errors, warnings = validate_dataset(df)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_dataset_missing_columns():
    bad_df = pd.DataFrame({"Student_ID": ["S1"], "Age": [20]})
    is_valid, errors, warnings = validate_dataset(bad_df)
    assert is_valid is False
    assert any("Missing required columns" in e for e in errors)


def test_validate_dataset_invalid_values():
    bad_df = generate_sample_dataset(n_samples=20, random_state=99)
    bad_df.loc[0, "Daily_Social_Media_Hours"] = -5.0
    bad_df.loc[1, "Academic_Percentage"] = 150.0

    is_valid, errors, warnings = validate_dataset(bad_df)
    assert is_valid is False
    assert any("negative Daily_Social_Media_Hours" in e for e in errors)
    assert any("Academic_Percentage outside [0, 100]" in e for e in errors)


def test_clean_dataset_imputation_and_duplicates():
    df = generate_sample_dataset(n_samples=40, random_state=1)
    # Add a duplicate
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert len(df) == 41

    # Add missing values
    df.loc[2, "Study_Hours_Per_Day"] = np.nan
    df.loc[3, "Most_Used_Platform"] = np.nan

    cleaned, audit = clean_dataset(df, missing_strategy="mean", remove_duplicates=True)
    assert len(cleaned) == 40
    assert audit["duplicates_removed"] == 1
    assert cleaned["Study_Hours_Per_Day"].isnull().sum() == 0
    assert cleaned["Most_Used_Platform"].isnull().sum() == 0


def test_clean_dataset_drop_strategy():
    df = generate_sample_dataset(n_samples=30, random_state=2)
    df.loc[0, "Sleep_Hours"] = np.nan
    cleaned, audit = clean_dataset(df, missing_strategy="drop", remove_duplicates=False)
    assert len(cleaned) == 29
    assert audit["missing_values_after"] == 0


def test_detect_outliers_iqr():
    df = pd.DataFrame({"val": [10, 12, 11, 10, 12, 11, 10, 100, -50]})
    outliers, info = detect_outliers(df, "val", method="iqr")
    assert info["outlier_count"] == 2
    assert 100 in outliers["val"].values
    assert -50 in outliers["val"].values


def test_detect_outliers_zscore():
    data = [10.0] * 50 + [100.0]
    df = pd.DataFrame({"val": data})
    outliers, info = detect_outliers(df, "val", method="zscore", threshold=3.0)
    assert info["outlier_count"] >= 1
    assert 100.0 in outliers["val"].values


def test_get_dataset_summary():
    df = generate_sample_dataset(n_samples=25, random_state=5)
    summary = get_dataset_summary(df)
    assert summary["total_records"] == 25
    assert summary["total_columns"] >= 10
    assert summary["duplicate_rows"] == 0
