"""Descriptive Statistics Module.

Computes academic descriptive statistics for univariate and multivariate student data,
including measures of central tendency, dispersion, position, shape, and academic formulas.

Course: Probability and Statistics Using Python (26B16CA212)
Institution: Jaypee Institute of Information Technology, Noida
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


def calculate_descriptive_stats(series: pd.Series) -> Dict[str, Any]:
    """Calculate comprehensive descriptive statistics for a numerical series.

    Computes:
    - Central Tendency: Mean, Median, Mode
    - Dispersion: Variance, Standard Deviation, Range, IQR
    - Positional: Minimum, Maximum, Q1 (25th percentile), Q2 (50th percentile), Q3 (75th percentile)
    - Shape: Skewness, Kurtosis
    - Sample Count: Non-null count
    """
    clean_series = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean_series)

    if n == 0:
        return {
            "Count": 0,
            "Mean": np.nan,
            "Median": np.nan,
            "Mode": np.nan,
            "Minimum": np.nan,
            "Maximum": np.nan,
            "Range": np.nan,
            "Variance": np.nan,
            "Standard_Deviation": np.nan,
            "Q1_25th": np.nan,
            "Q2_Median": np.nan,
            "Q3_75th": np.nan,
            "IQR": np.nan,
            "Skewness": np.nan,
            "Kurtosis": np.nan,
        }

    # Central Tendency
    mean_val = float(clean_series.mean())
    median_val = float(clean_series.median())
    mode_series = clean_series.mode()
    mode_val = float(mode_series.iloc[0]) if not mode_series.empty else mean_val

    # Positional
    min_val = float(clean_series.min())
    max_val = float(clean_series.max())
    range_val = max_val - min_val

    q1 = float(clean_series.quantile(0.25))
    q2 = float(clean_series.quantile(0.50))
    q3 = float(clean_series.quantile(0.75))
    iqr_val = q3 - q1

    # Dispersion (ddof=1 for sample statistics with Bessel's correction)
    variance_val = float(clean_series.var(ddof=1)) if n > 1 else 0.0
    std_val = float(clean_series.std(ddof=1)) if n > 1 else 0.0

    # Shape
    skew_val = float(stats.skew(clean_series, bias=False)) if n > 2 else 0.0
    kurt_val = float(stats.kurtosis(clean_series, bias=False)) if n > 3 else 0.0

    return {
        "Count": int(n),
        "Mean": round(mean_val, 4),
        "Median": round(median_val, 4),
        "Mode": round(mode_val, 4),
        "Minimum": round(min_val, 4),
        "Maximum": round(max_val, 4),
        "Range": round(range_val, 4),
        "Variance": round(variance_val, 4),
        "Standard_Deviation": round(std_val, 4),
        "Q1_25th": round(q1, 4),
        "Q2_Median": round(q2, 4),
        "Q3_75th": round(q3, 4),
        "IQR": round(iqr_val, 4),
        "Skewness": round(skew_val, 4),
        "Kurtosis": round(kurt_val, 4),
    }


def compute_summary_table(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Generate a formatted descriptive statistics summary table for numerical columns.

    Parameters:
        df: Input DataFrame
        columns: Specific numerical columns to summarize. If None, selects all numerical columns.

    Returns:
        pd.DataFrame: Table with statistics as rows and variables as columns.
    """
    if columns is None:
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "Student_ID"]

    records = {}
    for col in columns:
        if col in df.columns:
            records[col] = calculate_descriptive_stats(df[col])

    summary_df = pd.DataFrame(records)
    return summary_df


def interpret_distribution(stats_dict: Dict[str, Any]) -> Dict[str, str]:
    """Generate plain-language academic interpretation of distribution characteristics."""
    skew = stats_dict.get("Skewness", 0.0)
    kurt = stats_dict.get("Kurtosis", 0.0)
    std = stats_dict.get("Standard_Deviation", 0.0)
    mean = stats_dict.get("Mean", 0.0)
    median = stats_dict.get("Median", 0.0)

    # Skewness interpretation
    if abs(skew) < 0.5:
        skew_interp = f"Approximately symmetric distribution (Skewness = {skew:+.2f})."
    elif skew >= 0.5:
        skew_interp = f"Positively skewed (right-tailed, Skewness = {skew:+.2f}); mean ({mean:.2f}) > median ({median:.2f})."
    else:
        skew_interp = f"Negatively skewed (left-tailed, Skewness = {skew:+.2f}); mean ({mean:.2f}) < median ({median:.2f})."

    # Kurtosis interpretation (excess kurtosis relative to normal distribution = 0)
    if abs(kurt) < 0.5:
        kurt_interp = f"Mesokurtic distribution (Kurtosis = {kurt:+.2f}), exhibiting normal tail behavior."
    elif kurt >= 0.5:
        kurt_interp = f"Leptokurtic distribution (Kurtosis = {kurt:+.2f}), indicating heavier tails and sharper peak."
    else:
        kurt_interp = f"Platykurtic distribution (Kurtosis = {kurt:+.2f}), indicating lighter tails and flatter peak."

    # Spread
    cv = (std / mean * 100) if mean != 0 else 0
    spread_interp = f"Coefficient of Variation (CV) = {cv:.1f}%, indicating {'high' if cv > 30 else 'moderate' if cv > 15 else 'low'} relative variability."

    return {
        "Skewness_Interpretation": skew_interp,
        "Kurtosis_Interpretation": kurt_interp,
        "Spread_Interpretation": spread_interp,
    }


def get_formula_guide() -> Dict[str, Dict[str, str]]:
    """Return academic formulas and definitions for pedagogical clarity."""
    return {
        "Mean": {
            "formula": r"\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i",
            "description": "Arithmetic average representing the central balance point of the data.",
        },
        "Median": {
            "formula": r"\text{Median} = \text{Value at position } \frac{n+1}{2} \text{ in sorted data}",
            "description": "50th percentile dividing upper and lower halves; robust to extreme outliers.",
        },
        "Sample Variance": {
            "formula": r"s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2",
            "description": "Average squared deviation from the mean, using Bessel's correction (n - 1) for unbiased sample estimation.",
        },
        "Standard Deviation": {
            "formula": r"s = \sqrt{s^2} = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}",
            "description": "Measure of dispersion expressed in the same units as original measurements.",
        },
        "Interquartile Range": {
            "formula": r"\text{IQR} = Q_3 - Q_1",
            "description": "Middle 50% range of the distribution; foundation of John Tukey's boxplot outlier detection.",
        },
        "Sample Skewness": {
            "formula": r"g_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^3",
            "description": "Measure of distributional asymmetry. 0 indicates perfect symmetry; >0 right-tail; <0 left-tail.",
        },
        "Sample Kurtosis": {
            "formula": r"g_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}",
            "description": "Measure of tailedness relative to a Gaussian distribution (excess kurtosis).",
        },
    }
