"""Page 4: Regression Analysis.

Ordinary Least Squares (OLS) Simple and Multiple Linear Regression models,
train-test split validation, goodness-of-fit metrics (R², Adj R², MAE, RMSE),
residual diagnostics, and interactive performance prediction.

Course: Probability and Statistics Using Python (26B16CA212)
Institution: Jaypee Institute of Information Technology, Noida
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from src.data_processing import clean_dataset, generate_sample_dataset
from src.regression_analysis import (
    build_multiple_linear_regression,
    build_simple_linear_regression,
    predict_academic_performance,
)
from src.visualizations import (
    plot_regression_actual_vs_predicted,
    plot_regression_residuals,
    plot_residual_distribution,
)

st.set_page_config(page_title="Regression Analysis | JIIT Noida", page_icon="📉", layout="wide")

st.markdown("""
<div class="academic-header">
    <h1>📉 OLS Linear Regression & Predictive Modeling</h1>
    <p>Train-test split validation, Ordinary Least Squares parameter estimation, residual diagnostics, and inference.</p>
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
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ["Student_ID", "Academic_Percentage"]]

# Model Settings Sidebar / Header
st.subheader("⚙️ 1. Model Configuration")

col_type, col_split = st.columns([1.5, 1])
with col_type:
    model_choice = st.radio(
        "Select Regression Architecture:",
        ["Multiple Linear Regression (Recommended)", "Simple Linear Regression (Univariate)"],
        horizontal=True,
    )
with col_split:
    test_ratio = st.slider("Test Set Split Ratio (%):", min_value=10, max_value=40, value=20, step=5) / 100.0

if "Simple" in model_choice:
    feature_sel = st.selectbox("Select Independent Predictor (X):", num_cols, index=0)
    reg_results = build_simple_linear_regression(
        df,
        feature_col=feature_sel,
        target_col="Academic_Percentage",
        test_size=test_ratio,
        random_state=42,
    )
else:
    feature_sel = st.multiselect(
        "Select Independent Predictors (X):",
        options=num_cols,
        default=["Daily_Social_Media_Hours", "Study_Hours_Per_Day", "Sleep_Hours"],
    )
    if not feature_sel:
        st.warning("Please select at least one predictor feature.")
        st.stop()

    reg_results = build_multiple_linear_regression(
        df,
        feature_cols=feature_sel,
        target_col="Academic_Percentage",
        test_size=test_ratio,
        random_state=42,
    )

st.markdown("---")

# Section 2: Model Performance Metrics
st.subheader("📊 2. Held-Out Test Set Evaluation")
st.info(f"Model evaluated on **{reg_results['test_size']}** unseen test records (Training size: **{reg_results['train_size']}**). Data leakage was strictly prevented.")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("R² Score", f"{reg_results['r2_test']:.3f}")
m2.metric("Adjusted R²", f"{reg_results['adj_r2_test']:.3f}")
m3.metric("MAE", f"{reg_results['mae_test']:.2f}%")
m4.metric("MSE", f"{reg_results['mse_test']:.2f}")
m5.metric("RMSE", f"{reg_results['rmse_test']:.2f}%")

st.markdown(f"**Mathematical Regression Equation:**")
st.code(reg_results["equation"], language="text")

# Coefficients Table
if "Simple" in model_choice:
    coef_df = pd.DataFrame({
        "Parameter": ["Intercept (β₀)", f"Slope β₁ ({reg_results['feature_col']})"],
        "Coefficient Value": [reg_results["intercept"], reg_results["slope"]],
        "Interpretation": [
            "Baseline expected percentage when predictor equals 0.",
            f"Marginal change in academic % per 1 unit increase in {reg_results['feature_col']}.",
        ],
    })
else:
    coef_rows = [{"Parameter": "Intercept (β₀)", "Coefficient Value": reg_results["intercept"], "Interpretation": "Baseline academic percentage."}]
    for col, val in reg_results["coefficients"].items():
        coef_rows.append({
            "Parameter": f"β ({col})",
            "Coefficient Value": val,
            "Interpretation": f"Partial regression coefficient holding other covariates constant.",
        })
    coef_df = pd.DataFrame(coef_rows)

st.dataframe(coef_df, use_container_width=True)

st.markdown(f"""
<div class="callout-box">
    <strong>Pedagogical Model Interpretation:</strong><br>
    {reg_results['interpretation']}
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Section 3: Diagnostic Visualizations
st.subheader("🔍 3. Regression Diagnostic Plots")

row_diag1, row_diag2 = st.columns(2)
with row_diag1:
    fig_parity = plot_regression_actual_vs_predicted(
        reg_results["y_test"], reg_results["y_test_pred"]
    )
    st.plotly_chart(fig_parity, use_container_width=True)

with row_diag2:
    fig_res = plot_regression_residuals(
        reg_results["y_test_pred"], reg_results["residuals_test"]
    )
    st.plotly_chart(fig_res, use_container_width=True)

fig_res_dist = plot_residual_distribution(reg_results["residuals_test"])
st.plotly_chart(fig_res_dist, use_container_width=True)

st.markdown("---")

# Section 4: Interactive Student Academic Simulator
st.subheader("🎯 4. Interactive Academic Outcome Simulator")
st.markdown("Adjust input sliders to estimate predicted collegiate academic score using the trained model.")

pred_inputs = {}
input_cols = st.columns(len(feature_sel)) if isinstance(feature_sel, list) else st.columns(1)

features_to_input = feature_sel if isinstance(feature_sel, list) else [feature_sel]

for idx, fcol in enumerate(features_to_input):
    with input_cols[idx]:
        min_v = float(df[fcol].min())
        max_v = float(df[fcol].max())
        mean_v = float(df[fcol].mean())
        pred_inputs[fcol] = st.slider(
            f"{fcol}:",
            min_value=round(min_v, 1),
            max_value=round(max_v, 1),
            value=round(mean_v, 1),
            step=0.1,
        )

prediction = predict_academic_performance(
    reg_results["raw_model"], features_to_input, pred_inputs
)

st.markdown(f"""
<div style="background-color: #EFF6FF; border: 2px solid #1E3A8A; border-radius: 10px; padding: 20px; text-align: center; margin: 20px 0;">
    <h3 style="color: #64748B; margin-bottom: 4px;">ESTIMATED ACADEMIC PERCENTAGE</h3>
    <div style="font-size: 2.8rem; font-weight: 800; color: #1E3A8A;">{prediction['predicted_percentage']}%</div>
    <p style="font-size: 0.85rem; color: #64748B; margin-top: 6px;">Computed via Ordinary Least Squares Point Estimation</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="background-color: #FEF3C7; border-left: 4px solid #D97706; padding: 12px; border-radius: 0 6px 6px 0; font-size: 0.85rem; color: #92400E;">
    <strong>⚠️ Academic Caution & Ethics:</strong> {prediction['disclaimer']}
</div>
""", unsafe_allow_html=True)
