"""Tests for Descriptive Statistics Module.

Course: Probability and Statistics Using Python (26B16CA212)
Institution: Jaypee Institute of Information Technology, Noida
"""

import numpy as np
import pandas as pd
import pytest

from src.statistics_analysis import (
    calculate_descriptive_stats,
    compute_summary_table,
    interpret_distribution,
)


def test_calculate_descriptive_stats_known_data():
    data = pd.Series([2, 4, 4, 4, 5, 5, 7, 9])
    res = calculate_descriptive_stats(data)

    assert res["Count"] == 8
    assert res["Mean"] == 5.0
    assert res["Median"] == 4.5
    assert res["Mode"] == 4.0
    assert res["Minimum"] == 2.0
    assert res["Maximum"] == 9.0
    assert res["Range"] == 7.0
    # Sample variance with Bessel correction (N-1 = 7)
    # Sum of squared diffs: (2-5)^2 + 3*(4-5)^2 + 2*(5-5)^2 + (7-5)^2 + (9-5)^2 = 9 + 3 + 0 + 4 + 16 = 32
    # 32 / 7 = 4.5714
    assert np.isclose(res["Variance"], 32 / 7, atol=1e-3)
    assert np.isclose(res["Standard_Deviation"], np.sqrt(32 / 7), atol=1e-3)
    assert res["IQR"] == 1.5


def test_calculate_descriptive_stats_empty():
    res = calculate_descriptive_stats(pd.Series([]))
    assert res["Count"] == 0
    assert np.isnan(res["Mean"])


def test_compute_summary_table():
    df = pd.DataFrame({
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        "B": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    summary = compute_summary_table(df)
    assert "A" in summary.columns
    assert "B" in summary.columns
    assert "Mean" in summary.index
    assert summary.loc["Mean", "A"] == 3.0
    assert summary.loc["Mean", "B"] == 30.0


def test_interpret_distribution():
    stats_dict = {
        "Skewness": 0.12,
        "Kurtosis": -0.05,
        "Mean": 50.0,
        "Median": 49.5,
        "Standard_Deviation": 5.0,
    }
    interp = interpret_distribution(stats_dict)
    assert "Approximately symmetric" in interp["Skewness_Interpretation"]
    assert "Mesokurtic" in interp["Kurtosis_Interpretation"]
