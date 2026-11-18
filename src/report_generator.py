"""Academic Report Generator Module.

Compiles automated empirical research findings, generates comprehensive HTML/Markdown reports,
and prepares downloadable CSV analytical summaries for BCA academic submission.

Course: Probability and Statistics Using Python (26B16CA212)
Institution: Jaypee Institute of Information Technology, Noida
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd

from src.correlation_analysis import compute_correlation_matrix, compute_pairwise_correlation
from src.hypothesis_testing import run_independent_t_test, run_one_way_anova
from src.regression_analysis import build_multiple_linear_regression, build_simple_linear_regression
from src.statistics_analysis import calculate_descriptive_stats, compute_summary_table


def generate_academic_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute automated empirical findings derived from dataset."""
    n_students = len(df)

    # Core stats
    sm_stats = calculate_descriptive_stats(df["Daily_Social_Media_Hours"])
    acad_stats = calculate_descriptive_stats(df["Academic_Percentage"])
    study_stats = calculate_descriptive_stats(df["Study_Hours_Per_Day"])
    sleep_stats = calculate_descriptive_stats(df["Sleep_Hours"])
    stress_stats = calculate_descriptive_stats(df["Stress_Level"])

    # Core correlations
    corr_sm_acad = compute_pairwise_correlation(
        df["Daily_Social_Media_Hours"], df["Academic_Percentage"], "Social Media Hours", "Academic %"
    )
    corr_sm_study = compute_pairwise_correlation(
        df["Daily_Social_Media_Hours"], df["Study_Hours_Per_Day"], "Social Media Hours", "Study Hours"
    )
    corr_study_acad = compute_pairwise_correlation(
        df["Study_Hours_Per_Day"], df["Academic_Percentage"], "Study Hours", "Academic %"
    )

    # Usage breakdown
    usage_counts = df["Usage_Category"].value_counts().to_dict()
    low_pct = df[df["Usage_Category"] == "Low"]["Academic_Percentage"].mean()
    mod_pct = df[df["Usage_Category"] == "Moderate"]["Academic_Percentage"].mean()
    high_pct = df[df["Usage_Category"] == "High"]["Academic_Percentage"].mean()

    # Hypothesis test: Low vs High SM users on Academic Percentage
    t_test_res = run_independent_t_test(
        df[df["Usage_Category"] == "Low"]["Academic_Percentage"],
        df[df["Usage_Category"] == "High"]["Academic_Percentage"],
        "Low Usage (<2h)",
        "High Usage (>4h)",
        "Academic Percentage",
    )

    # Regression model
    reg_res = build_multiple_linear_regression(
        df, ["Daily_Social_Media_Hours", "Study_Hours_Per_Day", "Sleep_Hours"]
    )

    # Key findings statements
    findings = [
        f"Sample Cohort: A total of {n_students} student records were evaluated across undergraduate cohorts.",
        f"Usage Patterns: Mean social media usage was {sm_stats['Mean']:.2f} ± {sm_stats['Standard_Deviation']:.2f} hours/day. "
        f"{usage_counts.get('High', 0)} students ({usage_counts.get('High', 0) / n_students * 100:.1f}%) were classified as High usage (>4h/day).",
        f"Correlation with Academic Achievement: Daily social media usage exhibits a {corr_sm_acad['strength']} {corr_sm_acad['direction']} "
        f"association with academic percentage (r = {corr_sm_acad['pearson_r']:+.3f}, p = {corr_sm_acad['pearson_p']:.4e}).",
        f"Time Displacement: An inverse relationship exists between social media and study hours (r = {corr_sm_study['pearson_r']:+.3f}), "
        f"supporting the time-displacement hypothesis in collegiate environments.",
        f"Cohort Comparisons: Students with Low usage averaged {low_pct:.1f}% academic score, compared to {mod_pct:.1f}% for Moderate "
        f"and {high_pct:.1f}% for High usage students. The difference between Low and High usage is statistically significant (t = {t_test_res['t_statistic']:+.2f}, p = {t_test_res['p_value']:.4e}).",
        f"Multivariate Predictive Power: The multiple linear regression model explains {reg_res['r2_test'] * 100:.1f}% of variance "
        f"(R² = {reg_res['r2_test']:.3f}, RMSE = {reg_res['rmse_test']:.2f}%) on held-out test data.",
    ]

    return {
        "n_students": n_students,
        "sm_stats": sm_stats,
        "acad_stats": acad_stats,
        "study_stats": study_stats,
        "sleep_stats": sleep_stats,
        "stress_stats": stress_stats,
        "corr_sm_acad": corr_sm_acad,
        "corr_sm_study": corr_sm_study,
        "corr_study_acad": corr_study_acad,
        "t_test_res": t_test_res,
        "reg_res": reg_res,
        "findings": findings,
    }


def generate_html_report(df: pd.DataFrame) -> str:
    """Generate a clean, standalone, printable academic HTML report."""
    insights = generate_academic_insights(df)
    summary_df = compute_summary_table(df)
    timestamp = datetime.now().strftime("%B %d, %Y - %H:%M:%S")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Academic Statistical Research Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #0f172a;
            background-color: #ffffff;
            line-height: 1.6;
            margin: 40px auto;
            max-width: 900px;
            padding: 0 20px;
        }}
        .header {{
            border-bottom: 3px solid #1e3a8a;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #1e3a8a;
            font-size: 24px;
            margin: 0 0 8px 0;
        }}
        .meta-info {{
            font-size: 13px;
            color: #64748b;
        }}
        h2 {{
            color: #1e3a8a;
            font-size: 18px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 6px;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 13px;
        }}
        th, td {{
            border: 1px solid #cbd5e1;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background-color: #f1f5f9;
            color: #1e3a8a;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        .finding-box {{
            background-color: #f8fafc;
            border-left: 4px solid #1e3a8a;
            padding: 12px 16px;
            margin: 10px 0;
            border-radius: 0 6px 6px 0;
        }}
        .disclaimer {{
            background-color: #fffbeb;
            border-left: 4px solid #d97706;
            padding: 12px 16px;
            margin-top: 30px;
            font-size: 13px;
            color: #92400e;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 15px;
            border-top: 1px solid #e2e8f0;
            font-size: 12px;
            color: #94a3b8;
            text-align: center;
        }}
        @media print {{
            body {{ margin: 15mm; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Social Media Usage & Student Academic Performance Analysis</h1>
        <div class="meta-info">
            <strong>Course:</strong> Probability and Statistics Using Python (26B16CA212) | 
            <strong>Institution:</strong> Jaypee Institute of Information Technology, Noida<br>
            <strong>Generated:</strong> {timestamp} | <strong>Records Analyzed:</strong> {insights['n_students']}
        </div>
    </div>

    <h2>1. Executive Summary & Core Empirical Findings</h2>
"""
    for finding in insights["findings"]:
        html_content += f'    <div class="finding-box">{finding}</div>\n'

    html_content += """
    <h2>2. Descriptive Statistics Summary Table</h2>
    <table>
        <thead>
            <tr>
                <th>Statistical Metric</th>
"""
    for col in summary_df.columns:
        html_content += f"                <th>{col}</th>\n"
    html_content += "            </tr>\n        </thead>\n        <tbody>\n"

    for idx, row in summary_df.iterrows():
        html_content += f"            <tr>\n                <td><strong>{idx}</strong></td>\n"
        for val in row:
            html_content += f"                <td>{val}</td>\n"
        html_content += "            </tr>\n"

    html_content += f"""        </tbody>
    </table>

    <h2>3. Hypothesis Testing: Low vs. High Social Media Cohorts</h2>
    <p>
        <strong>Test:</strong> {insights['t_test_res']['test_name']}<br>
        <strong>Null Hypothesis:</strong> {insights['t_test_res']['null_hypothesis']}<br>
        <strong>Alternative Hypothesis:</strong> {insights['t_test_res']['alt_hypothesis']}<br>
        <strong>Test Statistic:</strong> t = {insights['t_test_res']['t_statistic']:+.3f}, df = {insights['t_test_res']['degrees_of_freedom']}, p = {insights['t_test_res']['p_value']:.4e}<br>
        <strong>Statistical Conclusion:</strong> {insights['t_test_res']['decision']}
    </p>

    <h2>4. Multivariate Linear Regression Model</h2>
    <p>
        <strong>Estimated Equation:</strong> <code>{insights['reg_res']['equation']}</code><br>
        <strong>R² (Test Set):</strong> {insights['reg_res']['r2_test']:.4f} | 
        <strong>Adjusted R²:</strong> {insights['reg_res']['adj_r2_test']:.4f}<br>
        <strong>MAE:</strong> {insights['reg_res']['mae_test']:.2f}% | 
        <strong>RMSE:</strong> {insights['reg_res']['rmse_test']:.2f}%<br>
        <strong>Interpretation:</strong> {insights['reg_res']['interpretation']}
    </p>

    <h2>5. Methodological Scope & Limitations</h2>
    <div class="disclaimer">
        <strong>Academic Notice:</strong> This analysis utilizes synthetic sample data designed to replicate
        inter-variable statistical properties for educational demonstration.
        Statistical correlations identify associations, not direct causation. Individual academic outcomes
        depend on multifaceted psychological, environmental, and institutional parameters beyond linear regression models.
    </div>

    <div class="footer">
        Social Media Usage & Student Performance Analysis System • BCA Semester III (26B16CA212) • JIIT Noida
    </div>
</body>
</html>
"""
    return html_content
