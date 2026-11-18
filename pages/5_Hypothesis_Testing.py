"""Page 5: Hypothesis Testing.

Inferential statistical testing:
1. Two-Sample Independent Student's / Welch's t-Test (with Levene's variance check)
2. One-Way Analysis of Variance (ANOVA) with Tukey HSD Post-Hoc Test
3. Pearson's Chi-Square Test of Independence (with Cramér's V and Cochran's rule check)

Course: Probability and Statistics Using Python (26B16CA212)
Institution: Jaypee Institute of Information Technology, Noida
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_processing import clean_dataset, generate_sample_dataset
from src.hypothesis_testing import (
    run_chi_square_test,
    run_independent_t_test,
    run_one_way_anova,
)

st.set_page_config(page_title="Hypothesis Testing | JIIT Noida", page_icon="🧪", layout="wide")

st.markdown("""
<div class="academic-header">
    <h1>🧪 Hypothesis Testing & Inferential Statistics</h1>
    <p>Parametric and non-parametric hypothesis tests, p-value evaluation, effect sizes, and decision rules.</p>
</div>
""", unsafe_allow_html=True)

# Ensure session state has clean_df
if "clean_df" not in st.session_state:
    sample_csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_data.csv")
    if os.path.exists(sample_csv_path):
        df_init = pd.read_csv(sample_csv_path)
    else:
        df_init = generate_sample_dataset(300, random_state=42)
    cleaned, _ = clean_dataset(df_init)
    st.session_state.raw_df = df_init
    st.session_state.clean_df = cleaned

df = st.session_state.clean_df
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "Student_ID"]
cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != "Student_ID"]

test_tab = st.selectbox(
    "Select Hypothesis Test to Perform:",
    [
        "1. Two-Sample Independent t-Test (Comparing 2 Groups)",
        "2. One-Way Analysis of Variance (ANOVA - Comparing 3+ Groups)",
        "3. Chi-Square Test of Independence (Categorical Association)",
    ],
)

alpha_val = st.slider("Significance Level (α):", min_value=0.01, max_value=0.10, value=0.05, step=0.01)

st.markdown("---")

# ==================== TEST 1: TWO-SAMPLE T-TEST ====================
if "t-Test" in test_tab:
    st.subheader("🔬 Two-Sample Independent t-Test")
    st.markdown("""
    Compares the mean of a continuous outcome between two independent student groups. 
    Homoscedasticity is tested via **Levene's Test**; if unequal variances are detected ($p < 0.05$), 
    **Welch's t-test** is automatically applied.
    """)

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        cat_group_col = st.selectbox("Categorical Grouping Variable:", cat_cols, index=cat_cols.index("Usage_Category") if "Usage_Category" in cat_cols else 0)
    with col_t2:
        available_groups = sorted(df[cat_group_col].dropna().unique())
        if len(available_groups) < 2:
            st.error("Grouping variable must have at least 2 categories.")
            st.stop()
        g1_name = st.selectbox("Select Group 1:", available_groups, index=0)
        g2_name = st.selectbox("Select Group 2:", available_groups, index=min(2, len(available_groups) - 1))
    with col_t3:
        target_metric = st.selectbox("Continuous Outcome Metric:", num_cols, index=num_cols.index("Academic_Percentage") if "Academic_Percentage" in num_cols else 0)

    if g1_name == g2_name:
        st.warning("Please select two distinct groups.")
    else:
        series1 = df[df[cat_group_col] == g1_name][target_metric]
        series2 = df[df[cat_group_col] == g2_name][target_metric]

        res = run_independent_t_test(series1, series2, g1_name, g2_name, target_metric, alpha=alpha_val)

        # Hypothesis Display Cards
        st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 14px; margin-bottom: 16px;">
            <p style="margin: 0 0 6px 0;"><strong>{res['null_hypothesis']}</strong></p>
            <p style="margin: 0;"><strong>{res['alt_hypothesis']}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("t-Statistic", f"{res['t_statistic']:+.3f}")
        k2.metric("Degrees of Freedom", f"{res['degrees_of_freedom']:.1f}")
        k3.metric("p-Value", f"{res['p_value']:.4e}")
        k4.metric("Cohen's d", f"{res['cohens_d']:+.2f}")
        k5.metric("Decision", "Reject H₀" if res["reject_null"] else "Fail to Reject")

        var_assumption_text = "Equal variances assumed (Student's t)" if res['equal_var_assumed'] else "Unequal variances: Welch's t-test applied"
        st.markdown(f"""
        <div class="callout-box">
            <strong>Statistical Decision:</strong> {res['decision']}<br><br>
            <strong>Levene's Variance Test:</strong> Statistic = {res['levene_stat']:.3f}, p = {res['levene_p']:.4f} 
            ({var_assumption_text}).
        </div>
        """, unsafe_allow_html=True)

        # Comparison Visualization
        comp_df = df[df[cat_group_col].isin([g1_name, g2_name])]
        fig_t = px.box(
            comp_df,
            x=cat_group_col,
            y=target_metric,
            color=cat_group_col,
            title=f"Comparison of {target_metric} between {g1_name} and {g2_name}",
            template="plotly_white",
            points="all",
        )
        st.plotly_chart(fig_t, use_container_width=True)

# ==================== TEST 2: ONE-WAY ANOVA ====================
elif "ANOVA" in test_tab:
    st.subheader("📊 One-Way Analysis of Variance (ANOVA)")
    st.markdown("""
    Tests whether the population means of a continuous outcome differ across three or more groups.
    If the omnibus F-test is statistically significant ($p < \\alpha$), **Tukey's Honestly Significant Difference (HSD)**
    post-hoc test is executed to identify pairwise group differences.
    """)

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        anova_group_col = st.selectbox(
            "Select Grouping Factor (3+ groups):",
            cat_cols,
            index=cat_cols.index("Usage_Category") if "Usage_Category" in cat_cols else 0,
        )
    with col_a2:
        anova_metric_col = st.selectbox(
            "Select Continuous Outcome Variable:",
            num_cols,
            index=num_cols.index("Academic_Percentage") if "Academic_Percentage" in num_cols else 0,
        )

    res_anova = run_one_way_anova(df, anova_group_col, anova_metric_col, alpha=alpha_val)

    if "error" in res_anova:
        st.error(res_anova["error"])
    else:
        st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 14px; margin-bottom: 16px;">
            <p style="margin: 0 0 6px 0;"><strong>{res_anova['null_hypothesis']}</strong></p>
            <p style="margin: 0;"><strong>{res_anova['alt_hypothesis']}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("F-Statistic", f"{res_anova['f_statistic']:.3f}")
        k2.metric("DF (Between, Within)", f"{res_anova['df_between']}, {res_anova['df_within']}")
        k3.metric("p-Value", f"{res_anova['p_value']:.4e}")
        k4.metric("Eta-Squared (η²)", f"{res_anova['eta_squared']:.3f}")
        k5.metric("Omnibus Decision", "Reject H₀" if res_anova["reject_null"] else "Fail to Reject")

        st.markdown(f"""
        <div class="callout-box">
            <strong>ANOVA Decision:</strong> {res_anova['decision']}
        </div>
        """, unsafe_allow_html=True)

        # Group Summary Table
        st.markdown("### Group Means & Standard Deviations")
        g_df = pd.DataFrame(res_anova["group_stats"]).T
        st.dataframe(g_df.style.format("{:.2f}"), use_container_width=True)

        # Tukey HSD Post-Hoc Table if available
        if res_anova["tukey_posthoc"]:
            st.markdown("### 🔍 Tukey HSD Post-Hoc Pairwise Comparisons")
            st.markdown("Identifies specifically which group pairs exhibit statistically significant mean differences.")
            st.dataframe(pd.DataFrame(res_anova["tukey_posthoc"]), use_container_width=True)

        fig_anova = px.box(
            df,
            x=anova_group_col,
            y=anova_metric_col,
            color=anova_group_col,
            title=f"{anova_metric_col} Across {anova_group_col} Cohorts",
            template="plotly_white",
            points="outliers",
        )
        st.plotly_chart(fig_anova, use_container_width=True)

# ==================== TEST 3: CHI-SQUARE TEST ====================
else:
    st.subheader("🎲 Pearson's Chi-Square Test of Independence")
    st.markdown("""
    Evaluates whether a statistically significant association exists between two categorical variables.
    Computes cross-tabulation of **Observed vs. Expected Frequencies** and **Cramér's V** for effect size.
    """)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        cat1 = st.selectbox("Select Categorical Variable 1:", cat_cols, index=0)
    with col_c2:
        cat2 = st.selectbox(
            "Select Categorical Variable 2:",
            cat_cols,
            index=1 if len(cat_cols) > 1 else 0,
        )

    if cat1 == cat2:
        st.warning("Please choose two different categorical variables.")
    else:
        res_chi = run_chi_square_test(df, cat1, cat2, alpha=alpha_val)

        st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 14px; margin-bottom: 16px;">
            <p style="margin: 0 0 6px 0;"><strong>{res_chi['null_hypothesis']}</strong></p>
            <p style="margin: 0;"><strong>{res_chi['alt_hypothesis']}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Chi-Square (χ²)", f"{res_chi['chi2_statistic']:.3f}")
        k2.metric("Degrees of Freedom", f"{res_chi['degrees_of_freedom']}")
        k3.metric("p-Value", f"{res_chi['p_value']:.4e}")
        k4.metric("Cramér's V", f"{res_chi['cramers_v']:.3f}")

        st.markdown(f"""
        <div class="callout-box">
            <strong>Chi-Square Conclusion:</strong> {res_chi['decision']}<br>
            <strong>Cochran's Rule Check:</strong> {res_chi['cells_below_5_pct']}% of cells have expected frequency < 5 
            ({'Valid asymptotic test' if res_chi['cells_below_5_pct'] <= 20 else 'Caution: Over 20% cells have expected count < 5'}).
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Cross-Tabulation: Observed Frequencies")
        st.dataframe(res_chi["observed_table"], use_container_width=True)

        st.markdown("### Cross-Tabulation: Expected Frequencies Under Independence (H₀)")
        st.dataframe(res_chi["expected_table"], use_container_width=True)
