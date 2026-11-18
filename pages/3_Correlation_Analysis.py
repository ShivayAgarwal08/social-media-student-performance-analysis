"""Page 3: Correlation Analysis.

Bivariate linear and monotonic association modeling, Pearson product-moment,
Spearman rank correlation, two-tailed p-value significance tests, and causality disclaimers.

Course: Probability and Statistics Using Python (26B16CA212)
Institution: Jaypee Institute of Information Technology, Noida
"""

import os
import streamlit as st
import pandas as pd

from src.correlation_analysis import (
    analyze_core_relationships,
    compute_correlation_matrix,
    compute_pairwise_correlation,
)
from src.data_processing import clean_dataset, generate_sample_dataset
from src.visualizations import plot_correlation_heatmap, plot_scatter_with_trend

st.set_page_config(page_title="Correlation Analysis | JIIT Noida", page_icon="🔗", layout="wide")

st.markdown("""
<div class="academic-header">
    <h1>🔗 Correlation Analysis & Significance Testing</h1>
    <p>Assess directional dependencies, Pearson product-moment linearity, Spearman rank monotonicity, and p-values.</p>
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

# Section 1: Multivariate Correlation Matrix & Heatmap
st.subheader("🔥 1. Correlation Matrix & Heatmap")

col_meth, col_disp = st.columns([1, 2])
with col_meth:
    corr_method = st.radio(
        "Select Correlation Formulation:",
        ["Pearson (Linear Association)", "Spearman (Rank Monotonicity)"],
        horizontal=True,
    )

method_key = "spearman" if "Spearman" in corr_method else "pearson"
corr_matrix, p_matrix = compute_correlation_matrix(df, num_cols, method=method_key)

col_heat, col_pvals = st.columns([1.3, 1])

with col_heat:
    fig_heat = plot_correlation_heatmap(corr_matrix, f"{corr_method.split()[0]} Correlation Matrix (r)")
    st.plotly_chart(fig_heat, use_container_width=True)

with col_pvals:
    st.markdown("### Two-Tailed P-Value Matrix")
    st.markdown("Values with $p < 0.05$ indicate statistical significance at $\\alpha = 0.05$.")
    st.dataframe(p_matrix.style.format("{:.4e}", na_rep="N/A"), use_container_width=True, height=480)

st.markdown("---")

# Section 2: Pairwise Bivariate Deep Dive
st.subheader("🔍 2. Pairwise Variable Analyzer")

col_var1, col_var2 = st.columns(2)
with col_var1:
    v1 = st.selectbox("Select Independent Variable (X):", num_cols, index=1)
with col_var2:
    v2 = st.selectbox("Select Dependent Variable (Y):", num_cols, index=4)

if v1 == v2:
    st.warning("Please choose two different variables for bivariate analysis.")
else:
    pair_res = compute_pairwise_correlation(df[v1], df[v2], v1, v2)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pearson r", f"{pair_res['pearson_r']:+.3f}")
    k2.metric("Pearson p-value", f"{pair_res['pearson_p']:.4e}")
    k3.metric("Spearman ρ", f"{pair_res['spearman_rho']:+.3f}")
    k4.metric("Shared Variance (r²)", f"{pair_res['r_squared'] * 100:.1f}%")

    col_scat, col_desc = st.columns([1.3, 1])
    with col_scat:
        fig_pair = plot_scatter_with_trend(
            df, v1, v2, v1, v2, f"{v1} vs. {v2} (OLS Regression Trendline)"
        )
        st.plotly_chart(fig_pair, use_container_width=True)

    with col_desc:
        st.markdown("### 📝 Academic Synthesis")
        st.markdown(f"""
        <div class="callout-box">
            <strong>Statistical Direction:</strong> {pair_res['direction'].title()}<br>
            <strong>Magnitude:</strong> {pair_res['strength'].title()}<br>
            <strong>Significance:</strong> {'Significant (p < 0.05)' if pair_res['is_significant'] else 'Not Significant (p ≥ 0.05)'}<br><br>
            {pair_res['explanation']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background-color: #FEF3C7; border-left: 4px solid #D97706; padding: 12px; border-radius: 0 6px 6px 0; font-size: 0.85rem; color: #92400E;">
            <strong>⚠️ Causality Notice:</strong> {pair_res['causation_disclaimer']}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Section 3: Curriculum Core Pairs
st.subheader("📚 3. Core Syllabus Relationships Summary")
st.markdown("Key collegiate pairs specified in the BCA curriculum guidelines:")

core_results = analyze_core_relationships(df)
core_records = []
for res in core_results:
    core_records.append({
        "Relationship": f"{res['name1']} ↔ {res['name2']}",
        "Pearson r": f"{res['pearson_r']:+.3f}",
        "P-Value": f"{res['pearson_p']:.4e}",
        "Spearman ρ": f"{res['spearman_rho']:+.3f}",
        "Strength": res["strength"].title(),
        "Direction": res["direction"].title(),
        "Significant (α=0.05)": "Yes" if res["is_significant"] else "No",
    })

st.dataframe(pd.DataFrame(core_records), use_container_width=True)
