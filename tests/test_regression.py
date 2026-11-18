"""Tests for Regression Analysis Module.

Course: Probability and Statistics Using Python (26B16CA212)
Institution: Jaypee Institute of Information Technology, Noida
"""

import numpy as np
import pandas as pd
import pytest

from src.regression_analysis import (
    build_multiple_linear_regression,
    build_simple_linear_regression,
    predict_academic_performance,
)


def test_simple_linear_regression_fit():
    np.random.seed(42)
    x = np.linspace(1, 10, 100)
    # y = 50 + 3*x + noise
    y = 50 + 3 * x + np.random.normal(0, 0.5, 100)
    df = pd.DataFrame({"Feature": x, "Target": y})

    res = build_simple_linear_regression(
        df, feature_col="Feature", target_col="Target", test_size=0.2, random_state=42
    )

    assert res["r2_test"] > 0.90
    assert np.isclose(res["slope"], 3.0, atol=0.5)
    assert np.isclose(res["intercept"], 50.0, atol=2.0)
    assert len(res["y_test"]) == 20
    assert len(res["residuals_test"]) == 20


def test_multiple_linear_regression_fit():
    np.random.seed(42)
    x1 = np.linspace(1, 10, 100)
    x2 = np.linspace(5, 15, 100)
    y = 30 + (2.5 * x1) + (1.5 * x2) + np.random.normal(0, 0.5, 100)
    df = pd.DataFrame({"X1": x1, "X2": x2, "Target": y})

    res = build_multiple_linear_regression(
        df, feature_cols=["X1", "X2"], target_col="Target", test_size=0.25, random_state=42
    )

    assert res["r2_test"] > 0.90
    assert res["test_size"] == 25
    assert "X1" in res["coefficients"]
    assert "X2" in res["coefficients"]


def test_predict_academic_performance():
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    # Fit simple 1D
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([50.0, 60.0, 70.0, 80.0])
    model.fit(X, y)

    pred_res = predict_academic_performance(model, ["Feat"], {"Feat": 5.0})
    assert pred_res["predicted_percentage"] == 90.0

    # Test bounding at 100%
    pred_high = predict_academic_performance(model, ["Feat"], {"Feat": 10.0})
    assert pred_high["predicted_percentage"] == 100.0
