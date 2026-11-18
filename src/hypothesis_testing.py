"""Hypothesis Testing Module.

Implements academic inferential tests:
1. Two-Sample Independent Student's / Welch's t-Test
2. One-Way Analysis of Variance (ANOVA) with Tukey HSD Post-Hoc Test
3. Chi-Square Test of Independence with Cramér's V Effect Size

Course: Probability and Statistics Using Python (26B16CA212)
Institution: Jaypee Institute of Information Technology, Noida
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def run_independent_t_test(
    group1: pd.Series,
    group2: pd.Series,
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    metric_name: str = "Metric",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Conduct a Two-Sample Independent t-Test comparing two groups on a continuous metric.

    Checks equality of variance via Levene's test. If Levene's p < 0.05, Welch's t-test
    (equal_var=False) is automatically selected to prevent type I error inflation.
    """
    g1 = pd.to_numeric(group1, errors="coerce").dropna()
    g2 = pd.to_numeric(group2, errors="coerce").dropna()

    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return {"error": "Each group must have at least 2 non-null observations."}

    m1, m2 = float(g1.mean()), float(g2.mean())
    s1, s2 = float(g1.std(ddof=1)), float(g2.std(ddof=1))

    # Levene's Test for Homoscedasticity (equal variance)
    levene_stat, levene_p = stats.levene(g1, g2)
    equal_var = bool(levene_p >= 0.05)

    # Perform t-test (Welch if variances differ)
    t_res = stats.ttest_ind(g1, g2, equal_var=equal_var)
    t_stat = float(t_res.statistic)
    p_val = float(t_res.pvalue)

    # Degrees of freedom calculation
    if equal_var:
        df = n1 + n2 - 2
    else:
        # Welch-Satterthwaite approximation
        v1 = (s1 ** 2) / n1
        v2 = (s2 ** 2) / n2
        df = ((v1 + v2) ** 2) / (((v1 ** 2) / (n1 - 1)) + ((v2 ** 2) / (n2 - 1)))

    # Cohen's d effect size (pooled standard deviation)
    pooled_s = np.sqrt((((n1 - 1) * (s1 ** 2)) + ((n2 - 1) * (s2 ** 2))) / (n1 + n2 - 2))
    cohens_d = (m1 - m2) / pooled_s if pooled_s > 0 else 0.0

    reject_null = bool(p_val < alpha)

    h0 = f"H₀: There is no significant difference in mean {metric_name} between {group1_name} (μ₁) and {group2_name} (μ₂), i.e., μ₁ - μ₂ = 0."
    h1 = f"H₁: There is a statistically significant difference in mean {metric_name} between {group1_name} and {group2_name}, i.e., μ₁ - μ₂ ≠ 0."

    if reject_null:
        decision = (
            f"Reject the Null Hypothesis (H₀) at α = {alpha}. "
            f"There is sufficient statistical evidence (t({df:.1f}) = {t_stat:+.3f}, p = {p_val:.4e}) "
            f"to conclude that mean {metric_name} differs significantly between {group1_name} (Mean = {m1:.2f}) "
            f"and {group2_name} (Mean = {m2:.2f}). Effect size: Cohen's d = {cohens_d:+.2f}."
        )
    else:
        decision = (
            f"Fail to Reject the Null Hypothesis (H₀) at α = {alpha}. "
            f"There is insufficient statistical evidence (t({df:.1f}) = {t_stat:+.3f}, p = {p_val:.4f}) "
            f"to conclude that mean {metric_name} differs between {group1_name} (Mean = {m1:.2f}) "
            f"and {group2_name} (Mean = {m2:.2f})."
        )

    return {
        "test_name": "Two-Sample Independent " + ("Student's" if equal_var else "Welch's") + " t-Test",
        "group1_name": group1_name,
        "group2_name": group2_name,
        "n1": n1,
        "n2": n2,
        "mean1": round(m1, 4),
        "mean2": round(m2, 4),
        "std1": round(s1, 4),
        "std2": round(s2, 4),
        "levene_stat": round(float(levene_stat), 4),
        "levene_p": round(float(levene_p), 6),
        "equal_var_assumed": equal_var,
        "t_statistic": round(t_stat, 4),
        "degrees_of_freedom": round(float(df), 2),
        "p_value": p_val,
        "cohens_d": round(float(cohens_d), 4),
        "alpha": alpha,
        "reject_null": reject_null,
        "null_hypothesis": h0,
        "alt_hypothesis": h1,
        "decision": decision,
    }


def run_one_way_anova(
    df: pd.DataFrame,
    group_col: str = "Usage_Category",
    metric_col: str = "Academic_Percentage",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Execute One-Way ANOVA to compare continuous metric across 3+ categorical groups.

    Includes Eta-Squared (η²) effect size and Tukey HSD post-hoc test.
    """
    clean_df = df[[group_col, metric_col]].dropna()
    groups = clean_df[group_col].unique()

    if len(groups) < 2:
        return {"error": "One-Way ANOVA requires at least 2 distinct groups."}

    group_data = [clean_df[clean_df[group_col] == g][metric_col] for g in groups]

    # ANOVA calculation
    f_res = stats.f_oneway(*group_data)
    f_stat = float(f_res.statistic)
    p_val = float(f_res.pvalue)

    k = len(groups)
    n = len(clean_df)
    df_between = k - 1
    df_within = n - k

    # Group statistics
    group_stats = clean_df.groupby(group_col)[metric_col].agg(["count", "mean", "std"]).to_dict(orient="index")

    # Eta-Squared (η²): Between-Group SS / Total SS
    overall_mean = clean_df[metric_col].mean()
    ss_between = sum(len(gd) * ((gd.mean() - overall_mean) ** 2) for gd in group_data)
    ss_total = sum((clean_df[metric_col] - overall_mean) ** 2)
    eta_squared = float(ss_between / ss_total) if ss_total > 0 else 0.0

    reject_null = bool(p_val < alpha)

    h0 = f"H₀: All population group means of {metric_col} across '{group_col}' are equal (μ₁ = μ₂ = ... = μₖ)."
    h1 = f"H₁: At least one group mean of {metric_col} across '{group_col}' differs significantly."

    # Post-hoc Tukey HSD if ANOVA is significant
    tukey_records = []
    if reject_null and len(groups) >= 3:
        try:
            tukey = pairwise_tukeyhsd(endog=clean_df[metric_col], groups=clean_df[group_col], alpha=alpha)
            for row in tukey.summary().data[1:]:
                tukey_records.append({
                    "Group 1": str(row[0]),
                    "Group 2": str(row[1]),
                    "Mean Diff": round(float(row[2]), 3),
                    "P-Adj": round(float(row[3]), 4),
                    "CI Lower": round(float(row[4]), 3),
                    "CI Upper": round(float(row[5]), 3),
                    "Significant": bool(row[6]),
                })
        except Exception:
            tukey_records = []

    if reject_null:
        decision = (
            f"Reject the Null Hypothesis (H₀) at α = {alpha}. "
            f"Statistically significant variance exists across {group_col} groups (F({df_between}, {df_within}) = {f_stat:.3f}, p = {p_val:.4e}). "
            f"Group differences account for approximately {eta_squared * 100:.1f}% of total variance in {metric_col} (η² = {eta_squared:.3f})."
        )
    else:
        decision = (
            f"Fail to Reject the Null Hypothesis (H₀) at α = {alpha}. "
            f"No statistically significant difference in mean {metric_col} was detected across {group_col} groups (F({df_between}, {df_within}) = {f_stat:.3f}, p = {p_val:.4f})."
        )

    return {
        "test_name": "One-Way Analysis of Variance (ANOVA)",
        "group_col": group_col,
        "metric_col": metric_col,
        "groups": list(groups),
        "group_stats": group_stats,
        "f_statistic": round(f_stat, 4),
        "df_between": df_between,
        "df_within": df_within,
        "p_value": p_val,
        "eta_squared": round(eta_squared, 4),
        "alpha": alpha,
        "reject_null": reject_null,
        "null_hypothesis": h0,
        "alt_hypothesis": h1,
        "decision": decision,
        "tukey_posthoc": tukey_records,
    }


def run_chi_square_test(
    df: pd.DataFrame,
    cat_col1: str = "Usage_Category",
    cat_col2: str = "Most_Used_Platform",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Execute Pearson's Chi-Square (χ²) Test of Independence between two categorical variables."""
    clean_df = df[[cat_col1, cat_col2]].dropna()
    contingency_table = pd.crosstab(clean_df[cat_col1], clean_df[cat_col2])

    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return {"error": "Contingency table must be at least 2x2."}

    chi2_stat, p_val, dof, expected_arr = stats.chi2_contingency(contingency_table)
    expected_df = pd.DataFrame(expected_arr, index=contingency_table.index, columns=contingency_table.columns)

    # Check Cochran's guideline (expected counts < 5)
    cells_below_5 = int((expected_df < 5).sum().sum())
    total_cells = contingency_table.size
    pct_below_5 = round((cells_below_5 / total_cells) * 100, 1)

    # Cramér's V effect size: sqrt(chi2 / (n * min(r-1, c-1)))
    n = len(clean_df)
    min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
    cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0

    reject_null = bool(p_val < alpha)

    h0 = f"H₀: {cat_col1} and {cat_col2} are mutually independent (no categorical association)."
    h1 = f"H₁: {cat_col1} and {cat_col2} are dependent (significant categorical association)."

    if reject_null:
        decision = (
            f"Reject the Null Hypothesis (H₀) at α = {alpha}. "
            f"There is a statistically significant association between {cat_col1} and {cat_col2} "
            f"(χ²({dof}) = {chi2_stat:.3f}, p = {p_val:.4e}, Cramér's V = {cramers_v:.3f})."
        )
    else:
        decision = (
            f"Fail to Reject the Null Hypothesis (H₀) at α = {alpha}. "
            f"No statistically significant association was found between {cat_col1} and {cat_col2} "
            f"(χ²({dof}) = {chi2_stat:.3f}, p = {p_val:.4f})."
        )

    return {
        "test_name": "Pearson's Chi-Square Test of Independence",
        "cat_col1": cat_col1,
        "cat_col2": cat_col2,
        "observed_table": contingency_table,
        "expected_table": expected_df.round(2),
        "chi2_statistic": round(float(chi2_stat), 4),
        "degrees_of_freedom": int(dof),
        "p_value": float(p_val),
        "cramers_v": round(float(cramers_v), 4),
        "cells_below_5_pct": pct_below_5,
        "alpha": alpha,
        "reject_null": reject_null,
        "null_hypothesis": h0,
        "alt_hypothesis": h1,
        "decision": decision,
    }
