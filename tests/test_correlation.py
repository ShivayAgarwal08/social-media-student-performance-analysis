"""Tests for Correlation Analysis Module.

Course: Probability and Statistics Using Python (26B16CA212)
Institution: Jaypee Institute of Information Technology, Noida
"""

import numpy as np
import pandas as pd
import pytest

from src.correlation_analysis import (
    analyze_core_relationships,
    compute_correlation_matrix,
    compute_pairwise_correlation,
)


def test_perfect_positive_correlation():
    x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    y = 2 * x + 5
    res = compute_pairwise_correlation(x, y, "X", "Y")

    assert np.isclose(res["pearson_r"], 1.0, atol=1e-4)
    assert np.isclose(res["spearman_rho"], 1.0, atol=1e-4)
    assert res["direction"] == "positive"
    assert res["strength"] == "very strong"
    assert res["is_significant"] is True


def test_perfect_negative_correlation():
    x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    y = -3 * x + 10
    res = compute_pairwise_correlation(x, y, "X", "Y")

    assert np.isclose(res["pearson_r"], -1.0, atol=1e-4)
    assert np.isclose(res["spearman_rho"], -1.0, atol=1e-4)
    assert res["direction"] == "negative"
    assert res["strength"] == "very strong"


def test_compute_correlation_matrix():
    df = pd.DataFrame({
        "v1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "v2": [2.0, 4.0, 6.0, 8.0, 10.0],
        "v3": [5.0, 4.0, 3.0, 2.0, 1.0],
    })
    corr, pvals = compute_correlation_matrix(df, ["v1", "v2", "v3"], method="pearson")

    assert corr.shape == (3, 3)
    assert pvals.shape == (3, 3)
    assert np.isclose(corr.loc["v1", "v2"], 1.0)
    assert np.isclose(corr.loc["v1", "v3"], -1.0)
    assert corr.loc["v1", "v1"] == 1.0


def test_insufficient_sample_size():
    x = pd.Series([1, 2])
    y = pd.Series([3, 4])
    res = compute_pairwise_correlation(x, y)
    assert "error" in res
