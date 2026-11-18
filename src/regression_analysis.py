"""Regression Analysis Module.

Implements Ordinary Least Squares (OLS) Simple and Multiple Linear Regression models
with rigorous train-test splitting to prevent data leakage, model diagnostics
(R², Adjusted R², MAE, MSE, RMSE), residual analysis, and interactive inference.

Course: Probability and Statistics Using Python (26B16CA212)
Institution: Jaypee Institute of Information Technology, Noida
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def build_simple_linear_regression(
    df: pd.DataFrame,
    feature_col: str = "Daily_Social_Media_Hours",
    target_col: str = "Academic_Percentage",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate a Simple Linear Regression model.

    y = beta_0 + beta_1 * x + epsilon
    """
    clean_df = df[[feature_col, target_col]].dropna()
    X = clean_df[[feature_col]]
    y = clean_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics on held-out test set
    r2_test = float(r2_score(y_test, y_test_pred))
    mae_test = float(mean_absolute_error(y_test, y_test_pred))
    mse_test = float(mean_squared_error(y_test, y_test_pred))
    rmse_test = float(np.sqrt(mse_test))

    n_test = len(y_test)
    k = 1  # 1 feature
    adj_r2 = 1.0 - (((1.0 - r2_test) * (n_test - 1)) / (n_test - k - 1)) if n_test > (k + 1) else r2_test

    intercept = float(model.intercept_)
    slope = float(model.coef_[0])

    # Residuals
    residuals_test = y_test - y_test_pred

    # Pedagogical interpretation
    slope_dir = "decreases" if slope < 0 else "increases"
    interpretation = (
        f"For every 1 additional unit increase in '{feature_col}', "
        f"'{target_col}' on average {slope_dir} by approximately {abs(slope):.2f}% (β₁ = {slope:+.3f}). "
        f"The baseline intercept (β₀) is {intercept:.2f}%. "
        f"On held-out testing data, the model explains {r2_test * 100:.1f}% of variance (R² = {r2_test:.3f}) "
        f"with an average prediction error (RMSE) of {rmse_test:.2f}%."
    )

    return {
        "model_type": "Simple Linear Regression",
        "feature_col": feature_col,
        "target_col": target_col,
        "intercept": round(intercept, 4),
        "slope": round(slope, 4),
        "equation": f"{target_col} = {intercept:.2f} + ({slope:+.3f} × {feature_col})",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "r2_test": round(r2_test, 4),
        "adj_r2_test": round(adj_r2, 4),
        "mae_test": round(mae_test, 4),
        "mse_test": round(mse_test, 4),
        "rmse_test": round(rmse_test, 4),
        "y_test": y_test.values,
        "y_test_pred": y_test_pred,
        "residuals_test": residuals_test.values,
        "interpretation": interpretation,
        "raw_model": model,
    }


def build_multiple_linear_regression(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "Academic_Percentage",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate a Multiple Linear Regression model with test-set validation.

    y = beta_0 + beta_1 * x_1 + beta_2 * x_2 + ... + beta_k * x_k + epsilon
    """
    if feature_cols is None:
        feature_cols = ["Daily_Social_Media_Hours", "Study_Hours_Per_Day", "Sleep_Hours"]

    clean_df = df[feature_cols + [target_col]].dropna()
    X = clean_df[feature_cols]
    y = clean_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    r2_test = float(r2_score(y_test, y_test_pred))
    mae_test = float(mean_absolute_error(y_test, y_test_pred))
    mse_test = float(mean_squared_error(y_test, y_test_pred))
    rmse_test = float(np.sqrt(mse_test))

    n_test = len(y_test)
    k = len(feature_cols)
    adj_r2 = 1.0 - (((1.0 - r2_test) * (n_test - 1)) / (n_test - k - 1)) if n_test > (k + 1) else r2_test

    intercept = float(model.intercept_)
    coef_dict = {col: round(float(coef), 4) for col, coef in zip(feature_cols, model.coef_)}

    residuals_test = y_test - y_test_pred

    # Build algebraic equation string
    eq_parts = [f"{intercept:.2f}"]
    for col, coef in coef_dict.items():
        eq_parts.append(f"({coef:+.3f} × {col})")
    equation = f"{target_col} = " + " + ".join(eq_parts)

    interpretation = (
        f"Multiple Linear Regression evaluated on {n_test} held-out test students: "
        f"R² = {r2_test:.3f} (Adjusted R² = {adj_r2:.3f}), MAE = {mae_test:.2f}%, RMSE = {rmse_test:.2f}%. "
        f"Holding other predictors constant: "
        + "; ".join([f"{col}: {coef:+.3f}% per unit" for col, coef in coef_dict.items()])
        + "."
    )

    return {
        "model_type": "Multiple Linear Regression",
        "feature_cols": feature_cols,
        "target_col": target_col,
        "intercept": round(intercept, 4),
        "coefficients": coef_dict,
        "equation": equation,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "r2_test": round(r2_test, 4),
        "adj_r2_test": round(adj_r2, 4),
        "mae_test": round(mae_test, 4),
        "mse_test": round(mse_test, 4),
        "rmse_test": round(rmse_test, 4),
        "y_test": y_test.values,
        "y_test_pred": y_test_pred,
        "residuals_test": residuals_test.values,
        "interpretation": interpretation,
        "raw_model": model,
    }


def predict_academic_performance(
    model: LinearRegression,
    feature_cols: List[str],
    input_values: Dict[str, float],
) -> Dict[str, Any]:
    """Predict academic percentage for custom user inputs with statistical guardrails."""
    input_array = np.array([[input_values[col] for col in feature_cols]])
    raw_pred = float(model.predict(input_array)[0])
    bounded_pred = max(0.0, min(100.0, raw_pred))

    return {
        "predicted_percentage": round(bounded_pred, 2),
        "raw_predicted": round(raw_pred, 2),
        "input_values": input_values,
        "disclaimer": (
            "Cautionary Academic Note: This is an empirical OLS statistical point estimate "
            "based on sample patterns. It is NOT a deterministic guarantee of individual "
            "student capability, as learning outcomes involve intrinsic motivation, teaching "
            "quality, prior mastery, and health factors outside linear models."
        ),
    }
