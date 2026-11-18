"""Page 6: Insights and Reports.

Synthesizes empirical findings across statistical models, provides downloadable CSV analytics,
and generates publication-quality academic research reports for BCA academic submission.

Course: Probability and Statistics Using Python (26B16CA212)
Institution: Jaypee Institute of Information Technology, Noida
"""

import os
import streamlit as st
import pandas as pd

from src.data_processing import clean_dataset, generate_sample_dataset
from src.report_generator import generate_academic_insights, generate_html_report
from src.statistics_analysis import compute_summary_table

st.set_page_config(page_title="Insights & Reports | JIIT Noida", page_icon="📑", layout="wide")

st.markdown("""
<div class="academic-header">
    <h1>📑 Empirical Insights & Academic Research Report</h1>
    <p>Automated statistical synthesis, analytical takeaways, exportable datasets, and printable reports.</p>
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
insights = generate_academic_insights(df)

# Section 1: Executive Findings Synthesis
st.subheader("💡 1. Automated Empirical Findings")
st.markdown("These takeaways are dynamically computed from the active student dataset:")

for finding in insights["findings"]:
    st.markdown(f"""
    <div class="finding-box" style="background-color: #F8FAFC; border-left: 4px solid #1E3A8A; padding: 12px 18px; margin: 10px 0; border-radius: 0 6px 6px 0;">
        {finding}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Section 2: Downloadable Reports Section
st.subheader("📥 2. Academic Report & Analytical Exports")
st.markdown("Export comprehensive reports formatted for academic evaluations, course submission, or printing:")

col_rep1, col_rep2 = st.columns(2)

with col_rep1:
    st.markdown("#### 📄 Standalone HTML Academic Report")
    st.write("Complete research document with methodology, statistical tables, models, and conclusions. Fully printable to PDF.")
    html_report = generate_html_report(df)
    st.download_button(
        label="📥 Download Academic Research Report (HTML)",
        data=html_report.encode("utf-8"),
        file_name="JIIT_Social_Media_Student_Performance_Report.html",
        mime="text/html",
    )

with col_rep2:
    st.markdown("#### 📊 Analytical Summary Metrics (CSV)")
    st.write("Raw computed metrics table covering central tendency, dispersion, and shape parameters for all variables.")
    summary_df = compute_summary_table(df)
    csv_summary = summary_df.to_csv().encode("utf-8")
    st.download_button(
        label="📥 Download Summary Metrics (CSV)",
        data=csv_summary,
        file_name="descriptive_statistics_matrix.csv",
        mime="text/csv",
    )

st.markdown("---")

# Section 3: Methodology & Statistical Framework
st.subheader("🔬 3. Statistical Methodology")

st.markdown("""
The statistical engine utilizes a rigorous four-phase quantitative pipeline:

1. **Preprocessing & Cleansing:**
   - Detects missing cells and applies configurable imputation ($\mu$, median, mode).
   - Identifies duplicates and filters out-of-bounds measurements.
   - Categorizes daily usage into **Low (< 2h)**, **Moderate (2–4h)**, and **High (> 4h)** cohorts.
   - Detects outliers via John Tukey's Interquartile Range ($[Q_1 - 1.5\\text{IQR}, Q_3 + 1.5\\text{IQR}]$) and Z-Scores ($|Z| > 3$).

2. **Descriptive Exploration:**
   - Computes parametric and non-parametric central tendency ($\bar{x}$, Median, Mode).
   - Unbiased sample variance ($s^2$, $ddof=1$) and standard deviation ($s$).
   - Positional quantiles ($Q_1$, $Q_2$, $Q_3$) and shape metrics (Fisher-Pearson skewness $g_1$, excess kurtosis $g_2$).

3. **Bivariate Association & Inference:**
   - Computes Pearson product-moment ($r$) and Spearman rank ($\rho$) coefficients with exact two-tailed $p$-values.
   - Conducts independent two-sample Welch/Student t-tests with Levene's variance homogeneity check.
   - Conducts One-Way ANOVA with Tukey HSD post-hoc comparisons.
   - Performs Pearson's Chi-Square ($\chi^2$) tests of independence with Cramér's V effect sizes.

4. **Predictive Modeling:**
   - Ordinary Least Squares (OLS) Multiple Linear Regression with 80/20 train-test splitting to eliminate data leakage.
   - Goodness-of-fit diagnostic evaluation: $R^2$, Adjusted $R^2$, MAE, MSE, and RMSE.
   - Residual diagnostics for homoscedasticity and normality.
""")

st.markdown("---")

# Section 4: Academic Limitations & Ethical Statement
st.subheader("⚠️ 4. Research Limitations & Ethical Considerations")

st.markdown("""
<div class="callout-box">
    <strong>Academic Limitations:</strong><br>
    <ul>
        <li><strong>Synthetic Data Origin:</strong> The default dataset is synthesized from realistic probability distributions for pedagogical illustration; it does not replace empirical clinical or longitudinal student surveys.</li>
        <li><strong>Cross-Sectional Limitation:</strong> Statistical associations capture correlational snapshots; they do not establish temporal or causal sequences without controlled experimental designs.</li>
        <li><strong>Unobserved Confounders:</strong> Scholastic performance is influenced by socioeconomic background, prior academic foundation, emotional wellbeing, sleep quality, and pedagogy—factors not fully captured by linear screen-time models.</li>
        <li><strong>Self-Reporting Bias:</strong> In real collegiate contexts, screen time and study duration self-reports frequently suffer from recall inaccuracies.</li>
    </ul>
</div>
""", unsafe_allow_html=True)
