"""Page 2: Descriptive Statistics.

Univariate and multivariate descriptive metrics, measures of central tendency,
dispersion, position, shape, distribution plots, and academic formula breakdowns.

Course: Probability and Statistics Using Python (26B16CA212)
Institution: Jaypee Institute of Information Technology, Noida
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_processing import clean_dataset, generate_sample_dataset
from src.statistics_analysis import (
    calculate_descriptive_stats,
    compute_summary_table,
    get_formula_guide,
    interpret_distribution,
)

st.set_page_config(page_title="Descriptive Statistics | JIIT Noida", page_icon="📈", layout="wide")

st.markdown("""
<div class="academic-header">
    <h1>📈 Descriptive Statistics & Distribution Analysis</h1>
    <p>Comprehensive measures of central tendency, dispersion, position, and shape across student metrics.</p>
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

# Section 1: Full Descriptive Summary Table
st.subheader("📊 1. Comprehensive Summary Statistics Table")
st.markdown("All calculations are empirically computed using unbiased sample variance (Bessel's correction, $n-1$).")

summary_table = compute_summary_table(df, num_cols)
st.dataframe(summary_table.style.format("{:.3f}", na_rep="N/A"), use_container_width=True)

# Download Summary Table
csv_data = summary_table.to_csv().encode("utf-8")
st.download_button(
    "📥 Download Descriptive Summary Table (CSV)",
    data=csv_data,
    file_name="descriptive_statistics_summary.csv",
    mime="text/csv",
)

st.markdown("---")

# Section 2: Univariate Variable Deep Dive
st.subheader("🔍 2. Univariate Distribution Explorer")
selected_var = st.selectbox("Select Numerical Variable to Analyze:", num_cols, index=1)

stats_dict = calculate_descriptive_stats(df[selected_var])
interp_dict = interpret_distribution(stats_dict)

# KPI Row for Selected Variable
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Sample Mean (x̄)", f"{stats_dict['Mean']:.2f}")
c2.metric("Median (Q2)", f"{stats_dict['Median']:.2f}")
c3.metric("Mode", f"{stats_dict['Mode']:.2f}")
c4.metric("Std Dev (s)", f"{stats_dict['Standard_Deviation']:.2f}")
c5.metric("IQR", f"{stats_dict['IQR']:.2f}")
c6.metric("Skewness (g₁)", f"{stats_dict['Skewness']:+.2f}")

# Distribution Visualizations
col_plot1, col_plot2 = st.columns(2)

with col_plot1:
    fig_hist = px.histogram(
        df,
        x=selected_var,
        nbins=25,
        marginal="rug",
        title=f"Distribution of {selected_var}",
        color_discrete_sequence=["#1E3A8A"],
        template="plotly_white",
    )
    fig_hist.add_vline(x=stats_dict["Mean"], line_dash="solid", line_color="#DC2626", annotation_text="Mean")
    fig_hist.add_vline(x=stats_dict["Median"], line_dash="dash", line_color="#059669", annotation_text="Median")
    fig_hist.update_layout(height=420)
    st.plotly_chart(fig_hist, use_container_width=True)

with col_plot2:
    fig_box = px.box(
        df,
        y=selected_var,
        x="Usage_Category" if "Usage_Category" in df.columns else None,
        color="Usage_Category" if "Usage_Category" in df.columns else None,
        title=f"Boxplot of {selected_var} by Usage Category",
        template="plotly_white",
        color_discrete_map={"Low": "#059669", "Moderate": "#0284C7", "High": "#DC2626"},
    )
    fig_box.update_layout(height=420)
    st.plotly_chart(fig_box, use_container_width=True)

# Distribution Interpretations Box
st.markdown("### 📝 Pedagogical Interpretation of Distribution")
st.markdown(f"""
<div class="callout-box">
    <strong>Symmetry & Skewness:</strong> {interp_dict['Skewness_Interpretation']}<br><br>
    <strong>Tailedness & Kurtosis:</strong> {interp_dict['Kurtosis_Interpretation']}<br><br>
    <strong>Spread & Variability:</strong> {interp_dict['Spread_Interpretation']}
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Section 3: Pedagogical Formula Reference
st.subheader("📚 3. Academic Formula Reference (BCA Syllabus)")
st.markdown("Mathematical definitions implemented in the backend statistical engine:")

formulas = get_formula_guide()
for name, data in formulas.items():
    with st.expander(f"📌 {name}"):
        st.latex(data["formula"])
        st.markdown(f"**Interpretation:** {data['description']}")
