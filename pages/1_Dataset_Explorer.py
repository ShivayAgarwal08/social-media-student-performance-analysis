"""Page 1: Dataset Explorer.

Dataset ingestion, upload validation, schema inspection, missing/duplicate audits,
outlier detection, and data export.

Course: Probability and Statistics Using Python (26B16CA212)
Institution: Jaypee Institute of Information Technology, Noida
"""

import io
import os
import streamlit as st
import pandas as pd

from src.data_processing import (
    clean_dataset,
    detect_outliers,
    generate_sample_dataset,
    get_dataset_summary,
    validate_dataset,
)

st.set_page_config(page_title="Dataset Explorer | JIIT Noida", page_icon="📁", layout="wide")

st.markdown("""
<div class="academic-header">
    <h1>📁 Dataset Explorer & Preprocessing Diagnostics</h1>
    <p>Upload collegiate data, review data integrity, inspect schema definitions, and audit outliers.</p>
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

# File Uploader Section
st.subheader("📤 Data Ingestion (Upload CSV or Excel)")
uploaded_file = st.file_uploader(
    "Upload student survey dataset (.csv, .xlsx, .xls)",
    type=["csv", "xlsx", "xls"],
    help="Upload your own student dataset to run statistical calculations.",
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            user_df = pd.read_csv(uploaded_file)
        else:
            user_df = pd.read_excel(uploaded_file)

        is_valid, errors, warnings = validate_dataset(user_df)

        if not is_valid:
            st.error("❌ Dataset Schema Validation Failed:")
            for err in errors:
                st.write(f"- {err}")
        else:
            if warnings:
                st.warning("⚠️ Schema Warnings:")
                for warn in warnings:
                    st.write(f"- {warn}")

            st.success(f"✅ Successfully ingested '{uploaded_file.name}' with {len(user_df)} student records.")
            cleaned, audit = clean_dataset(user_df)
            st.session_state.raw_df = user_df
            st.session_state.clean_df = cleaned

    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")

col_left, col_right = st.columns([1, 1])
with col_left:
    if st.button("🔄 Reset to Default Synthetic Dataset (300 JIIT Students)"):
        sample_df = generate_sample_dataset(300, random_state=42)
        cleaned, _ = clean_dataset(sample_df)
        st.session_state.raw_df = sample_df
        st.session_state.clean_df = cleaned
        st.rerun()

with col_right:
    # Download sample dataset
    sample_csv = st.session_state.clean_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Active Dataset as CSV",
        data=sample_csv,
        file_name="student_social_media_academic_data.csv",
        mime="text/csv",
    )

st.markdown("---")

df = st.session_state.clean_df
summary = get_dataset_summary(df)

# Dataset Health Summary Metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Records", summary["total_records"])
m2.metric("Total Variables", summary["total_columns"])
m3.metric("Missing Cells", summary["missing_cells"])
m4.metric("Duplicates", summary["duplicate_rows"])
m5.metric("Memory Footprint", f"{summary['memory_usage_kb']} KB")

# Tabbed Explorer
tab_preview, tab_schema, tab_outliers, tab_cleaning = st.tabs([
    "👀 Data Preview & Filters",
    "📋 Schema & Data Types",
    "🔍 Outlier Analysis",
    "🧹 Preprocessing & Cleaning Audit",
])

with tab_preview:
    st.markdown("### Interactive Dataset Preview")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sel_platforms = st.multiselect(
            "Filter Platform:",
            options=sorted(df["Most_Used_Platform"].dropna().unique()),
            default=sorted(df["Most_Used_Platform"].dropna().unique()),
        )
    with col_f2:
        sel_usage = st.multiselect(
            "Filter Usage Tier:",
            options=["Low", "Moderate", "High"],
            default=["Low", "Moderate", "High"],
        )
    with col_f3:
        if "Gender" in df.columns:
            sel_gender = st.multiselect(
                "Filter Gender:",
                options=sorted(df["Gender"].dropna().unique()),
                default=sorted(df["Gender"].dropna().unique()),
            )
        else:
            sel_gender = []

    filtered_df = df[
        (df["Most_Used_Platform"].isin(sel_platforms)) &
        (df["Usage_Category"].isin(sel_usage))
    ]
    if sel_gender and "Gender" in df.columns:
        filtered_df = filtered_df[filtered_df["Gender"].isin(sel_gender)]

    st.write(f"Showing **{len(filtered_df)}** of **{len(df)}** rows matching active filters.")
    st.dataframe(filtered_df, use_container_width=True, height=380)

with tab_schema:
    st.markdown("### Variable Data Dictionary")
    schema_records = []
    for col in df.columns:
        schema_records.append({
            "Variable Name": col,
            "Data Type": str(df[col].dtype),
            "Non-Null Count": int(df[col].count()),
            "Null Count": int(df[col].isnull().sum()),
            "Distinct Values": int(df[col].nunique()),
            "Example Value": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "None",
        })
    st.dataframe(pd.DataFrame(schema_records), use_container_width=True)

with tab_outliers:
    st.markdown("### Statistical Outlier Identification")
    st.info(
        "Outliers are identified using either John Tukey's Interquartile Range (IQR) rule "
        "($[Q_1 - 1.5 \\times \\text{IQR}, Q_3 + 1.5 \\times \\text{IQR}]$) or standard Z-score ($|Z| > 3$)."
    )
    col_out_var, col_out_meth = st.columns(2)
    with col_out_var:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "Student_ID"]
        target_out_col = st.selectbox("Select Numerical Variable to Audit:", num_cols)
    with col_out_meth:
        out_method = st.radio("Detection Methodology:", ["IQR (Tukey's Fences)", "Z-Score (|z| > 3)"], horizontal=True)

    method_key = "iqr" if "IQR" in out_method else "zscore"
    outliers, info = detect_outliers(df, target_out_col, method=method_key)

    if info["outlier_count"] == 0:
        st.success(f"No statistical outliers detected in '{target_out_col}' using {out_method}.")
    else:
        st.warning(f"Detected **{info['outlier_count']}** outlier records ({info['outlier_percentage']}%) in '{target_out_col}'.")
        st.json(info)
        st.dataframe(outliers, use_container_width=True)

with tab_cleaning:
    st.markdown("### Missing Value & Imputation Control")
    st.markdown("""
    Select how missing numerical observations should be handled. In accordance with academic guidelines, 
    records are never discarded silently; all imputation steps are logged transparently.
    """)
    col_imp, col_dup = st.columns(2)
    with col_imp:
        strategy = st.selectbox(
            "Numerical Imputation Strategy:",
            ["mean", "median", "mode", "drop"],
            help="Choose mean for normal data, median for skewed data, or drop to exclude incomplete records.",
        )
    with col_dup:
        remove_dup = st.checkbox("Automatically remove duplicate records", value=True)

    if st.button("Apply Preprocessing Transformation"):
        cleaned, audit = clean_dataset(
            st.session_state.raw_df,
            missing_strategy=strategy,
            remove_duplicates=remove_dup,
        )
        st.session_state.clean_df = cleaned
        st.success("Dataset successfully transformed and updated in session state!")
        st.json(audit)
