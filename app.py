"""Main Streamlit Application: Social Media Usage & Student Performance Analysis System.

Course: Probability and Statistics Using Python (Course Code: 26B16CA212)
Institution: Jaypee Institute of Information Technology (JIIT), Noida
Curriculum: BCA Semester III
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from src.data_processing import (
    generate_sample_dataset,
    clean_dataset,
    get_dataset_summary,
)
from src.statistics_analysis import calculate_descriptive_stats
from src.visualizations import (
    plot_usage_distribution,
    plot_academic_by_category,
    plot_platform_usage,
    plot_scatter_with_trend,
)
from src.report_generator import generate_academic_insights

# Page Configuration
st.set_page_config(
    page_title="Student Social Media & Performance Analysis | JIIT Noida",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Session State Initialization
if "raw_df" not in st.session_state:
    sample_csv_path = os.path.join(os.path.dirname(__file__), "data", "sample_data.csv")
    if os.path.exists(sample_csv_path):
        st.session_state.raw_df = pd.read_csv(sample_csv_path)
    else:
        st.session_state.raw_df = generate_sample_dataset(300, random_state=42)

if "clean_df" not in st.session_state:
    cleaned, _ = clean_dataset(st.session_state.raw_df)
    st.session_state.clean_df = cleaned

df = st.session_state.clean_df

# Sidebar Navigation & Academic Metadata
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0;">
    <h3 style="color: #1E3A8A; margin-bottom: 2px;">JIIT NOIDA</h3>
    <p style="font-size: 0.8rem; color: #64748B; margin: 0;">Department of Computer Science & IT</p>
    <p style="font-size: 0.75rem; color: #94A3B8; margin: 0;">BCA Semester III • 2026</p>
</div>
<hr style="margin: 10px 0; border: none; border-top: 1px solid #E2E8F0;" />
""", unsafe_allow_html=True)

st.sidebar.info(
    "**Course:** Probability and Statistics Using Python\n\n"
    "**Course Code:** 26B16CA212\n\n"
    "**Academic System:** Empirical analysis of social media screen time vs collegiate metrics."
)

st.sidebar.markdown("### Quick Dataset Filter")
platform_filter = st.sidebar.multiselect(
    "Filter by Platform",
    options=list(df["Most_Used_Platform"].unique()),
    default=list(df["Most_Used_Platform"].unique()),
)

usage_filter = st.sidebar.multiselect(
    "Filter by Usage Category",
    options=["Low", "Moderate", "High"],
    default=["Low", "Moderate", "High"],
)

filtered_df = df[
    (df["Most_Used_Platform"].isin(platform_filter)) &
    (df["Usage_Category"].isin(usage_filter))
]

if filtered_df.empty:
    st.warning("No records match the active filter criteria. Showing full dataset.")
    filtered_df = df

# Header Banner
st.markdown("""
<div class="academic-header">
    <h1>Social Media Usage & Student Mental/Academic Performance Analysis System</h1>
    <p>Empirical statistical modeling of undergraduate time expenditure, cognitive concentration, and scholastic performance.</p>
    <div style="margin-top: 10px;">
        <span class="badge-tag">JIIT Noida</span>
        <span class="badge-tag">Course Code: 26B16CA212</span>
        <span class="badge-tag">BCA Sem III</span>
        <span class="badge-tag">Probability & Statistics</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Synthetic Data Academic Disclosure Callout
st.markdown("""
<div class="callout-box">
    <strong>Academic Notice & Ethical Disclosure:</strong> 
    This application utilizes statistically plausible synthetic student survey data (N = 300) 
    synthesized using realistic probability distributions (Log-Normal, Gamma, and Multivariate Gaussian) 
    strictly for educational and statistical demonstration purposes. It does not represent identifiable student personal data.
</div>
""", unsafe_allow_html=True)

# Key Performance Indicators (KPI Cards)
n_students = len(filtered_df)
avg_sm = filtered_df["Daily_Social_Media_Hours"].mean()
avg_acad = filtered_df["Academic_Percentage"].mean()
avg_study = filtered_df["Study_Hours_Per_Day"].mean()
avg_sleep = filtered_df["Sleep_Hours"].mean()
avg_stress = filtered_df["Stress_Level"].mean()

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-title">Students</div>
        <div class="metric-card-value">{n_students}</div>
        <div class="metric-card-caption">Total records analyzed</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-title">Mean Screen Time</div>
        <div class="metric-card-value">{avg_sm:.1f} <span style="font-size: 1rem; color: #64748B;">h/d</span></div>
        <div class="metric-card-caption">Daily social media</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-title">Academic Score</div>
        <div class="metric-card-value">{avg_acad:.1f}%</div>
        <div class="metric-card-caption">Cohort mean percentage</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-title">Study Time</div>
        <div class="metric-card-value">{avg_study:.1f} <span style="font-size: 1rem; color: #64748B;">h/d</span></div>
        <div class="metric-card-caption">Independent study</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-title">Sleep Duration</div>
        <div class="metric-card-value">{avg_sleep:.1f} <span style="font-size: 1rem; color: #64748B;">h/d</span></div>
        <div class="metric-card-caption">Rest & recovery</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-title">Mean Stress</div>
        <div class="metric-card-value">{avg_stress:.1f} <span style="font-size: 1rem; color: #64748B;">/10</span></div>
        <div class="metric-card-caption">Self-reported scale</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Overview Analytical Visualizations
row1_left, row1_right = st.columns(2)

with row1_left:
    fig_usage = plot_usage_distribution(filtered_df)
    st.plotly_chart(fig_usage, use_container_width=True)

with row1_right:
    fig_acad = plot_academic_by_category(filtered_df)
    st.plotly_chart(fig_acad, use_container_width=True)

row2_left, row2_right = st.columns(2)

with row2_left:
    fig_scatter = plot_scatter_with_trend(
        filtered_df,
        "Daily_Social_Media_Hours",
        "Academic_Percentage",
        "Daily Social Media Hours",
        "Academic Percentage (%)",
        "Social Media Hours vs. Academic Performance (OLS Fit)",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with row2_right:
    fig_platform = plot_platform_usage(filtered_df)
    st.plotly_chart(fig_platform, use_container_width=True)

st.markdown("---")

# Executive Empirical Insights & Summary
st.subheader("📋 Executive Statistical Summary & Automated Findings")
insights = generate_academic_insights(filtered_df)

for finding in insights["findings"]:
    st.markdown(f"- {finding}")

st.markdown("""
<div style="background-color: #F1F5F9; border-radius: 8px; padding: 16px; margin-top: 20px;">
    <h4 style="color: #1E3A8A; margin-top: 0;">Navigation Guide</h4>
    <p style="font-size: 0.9rem; margin-bottom: 0;">
        Use the sidebar navigation to explore dedicated academic modules:
        <strong>1. Dataset Explorer</strong> (Upload & Inspection) • 
        <strong>2. Descriptive Statistics</strong> (Measures of Central Tendency & Dispersion) • 
        <strong>3. Correlation Analysis</strong> (Pearson & Spearman Significance Matrices) • 
        <strong>4. Regression Analysis</strong> (OLS Simple & Multiple Regression Models) • 
        <strong>5. Hypothesis Testing</strong> (t-Tests, ANOVA, Chi-Square Tests) • 
        <strong>6. Insights & Reports</strong> (Automated Academic Findings & Export).
    </p>
</div>
""", unsafe_allow_html=True)
