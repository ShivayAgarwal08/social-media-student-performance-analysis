"""Correlation Analysis Module.

Calculates Pearson product-moment and Spearman rank-order correlations,
computes exact two-tailed p-values, generates significance matrices, and provides
scholarly interpretations distinguishing correlation from causation.

Course: Probability and Statistics Using Python (26B16CA212)
Institution: Jaypee Institute of Information Technology, Noida
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def compute_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute correlation coefficient matrix and corresponding p-value matrix.

    Parameters:
        df: Input DataFrame
        columns: List of numerical columns (default: all numerical columns)
        method: 'pearson' or 'spearman'

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (Correlation Coefficients, P-Values)
    """
    if columns is None:
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "Student_ID"]

    clean_df = df[columns].dropna()
    p_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    corr_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)

    for col1 in columns:
        for col2 in columns:
            if col1 == col2:
                corr_matrix.loc[col1, col2] = 1.0
                p_matrix.loc[col1, col2] = 0.0
            else:
                s1 = clean_df[col1]
                s2 = clean_df[col2]
                if method.lower() == "spearman":
                    res = stats.spearmanr(s1, s2)
                else:
                    res = stats.pearsonr(s1, s2)
                corr_matrix.loc[col1, col2] = float(res.statistic)
                p_matrix.loc[col1, col2] = float(res.pvalue)

    return corr_matrix.astype(float), p_matrix.astype(float)


def compute_pairwise_correlation(
    series1: pd.Series,
    series2: pd.Series,
    name1: str = "Variable 1",
    name2: str = "Variable 2",
) -> Dict[str, Any]:
    """Compute both Pearson and Spearman correlations with complete academic interpretations.

    Parameters:
        series1: First numerical variable
        series2: Second numerical variable
        name1: Label for first variable
        name2: Label for second variable

    Returns:
        Dict[str, Any]: Metrics, p-values, strength, direction, and academic narratives.
    """
    df_pair = pd.DataFrame({"v1": series1, "v2": series2}).dropna()
    n = len(df_pair)

    if n < 3:
        return {
            "n": n,
            "error": "Insufficient sample size (minimum 3 observations required).",
        }

    # Pearson
    pearson_res = stats.pearsonr(df_pair["v1"], df_pair["v2"])
    r_val = float(pearson_res.statistic)
    p_val_pearson = float(pearson_res.pvalue)

    # Spearman
    spearman_res = stats.spearmanr(df_pair["v1"], df_pair["v2"])
    rho_val = float(spearman_res.statistic)
    p_val_spearman = float(spearman_res.pvalue)

    # Direction
    direction = "positive" if r_val > 0 else "negative" if r_val < 0 else "neutral"

    # Strength benchmark
    abs_r = abs(r_val)
    if abs_r < 0.20:
        strength = "very weak"
    elif abs_r < 0.40:
        strength = "weak"
    elif abs_r < 0.60:
        strength = "moderate"
    elif abs_r < 0.80:
        strength = "strong"
    else:
        strength = "very strong"

    # Statistical significance at alpha = 0.05
    is_significant = p_val_pearson < 0.05
    if p_val_pearson < 0.001:
        sig_text = "statistically highly significant (p < 0.001)"
    elif p_val_pearson < 0.01:
        sig_text = "statistically significant (p < 0.01)"
    elif p_val_pearson < 0.05:
        sig_text = "statistically significant (p < 0.05)"
    else:
        sig_text = "not statistically significant (p >= 0.05)"

    # Academic explanation
    explanation = (
        f"A {strength} {direction} linear correlation (r = {r_val:+.3f}) was observed between "
        f"{name1} and {name2}. This association is {sig_text}. "
        f"The coefficient of determination (r² = {r_val**2:.3f}) indicates that approximately "
        f"{r_val**2 * 100:.1f}% of the variance in {name2} can be linearly shared with {name1}."
    )

    causation_disclaimer = (
        "Academic Reminder: Correlation does not imply causation. A statistically significant "
        "association between social media hours and student metrics indicates a co-varying "
        "relationship, but does not prove direct causality without controlled longitudinal trials."
    )

    return {
        "n": n,
        "name1": name1,
        "name2": name2,
        "pearson_r": round(r_val, 4),
        "pearson_p": round(p_val_pearson, 6),
        "spearman_rho": round(rho_val, 4),
        "spearman_p": round(p_val_spearman, 6),
        "r_squared": round(r_val ** 2, 4),
        "direction": direction,
        "strength": strength,
        "is_significant": is_significant,
        "explanation": explanation,
        "causation_disclaimer": causation_disclaimer,
    }


def analyze_core_relationships(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Analyze the syllabus core pairs specified in the BCA curriculum."""
    core_pairs = [
        ("Daily_Social_Media_Hours", "Academic_Percentage", "Daily Social Media Hours", "Academic Percentage"),
        ("Daily_Social_Media_Hours", "Study_Hours_Per_Day", "Daily Social Media Hours", "Daily Study Hours"),
        ("Daily_Social_Media_Hours", "Sleep_Hours", "Daily Social Media Hours", "Sleep Duration"),
        ("Daily_Social_Media_Hours", "Stress_Level", "Daily Social Media Hours", "Stress Level"),
        ("Daily_Social_Media_Hours", "Concentration_Level", "Daily Social Media Hours", "Concentration Level"),
        ("Study_Hours_Per_Day", "Academic_Percentage", "Daily Study Hours", "Academic Percentage"),
    ]

    results = []
    for col1, col2, name1, name2 in core_pairs:
        if col1 in df.columns and col2 in df.columns:
            res = compute_pairwise_correlation(df[col1], df[col2], name1, name2)
            results.append(res)

    return results
