# Social Media Usage & Student Mental/Academic Performance Analysis System

An end-to-end, empirical statistical analytics platform developed for **BCA Semester III** students to model, evaluate, and interpret the multi-dimensional relationships between undergraduate social media consumption, study routines, sleep duration, psychological stress, and scholastic achievement.

---

### Academic Affiliation & Course Metadata
- **Institution:** Jaypee Institute of Information Technology (JIIT), Noida
- **Course Name:** Probability and Statistics Using Python
- **Course Code:** `26B16CA212`
- **Program / Semester:** Bachelor of Computer Applications (BCA), Semester III
- **Academic Year:** 2026

---

## 1. Executive Summary & Problem Statement

Collegiate students frequently balance intense scholastic obligations alongside ubiquitous digital screen time. Unmonitored social media engagement can displace academic study hours, impair sleep hygiene, and heighten cognitive stress. Conversely, moderate social media usage may facilitate peer communication and social connectivity.

This project delivers an interactive, mathematically rigorous, reproducible Python and Streamlit system to investigate:
1. **Central Tendency & Dispersion:** What are the distributional characteristics of student screen time, sleep, and academic percentage?
2. **Directional Association:** Does increased daily social media usage correlate with lower academic marks, reduced study hours, and increased stress?
3. **Multivariate Prediction:** How effectively can academic outcomes be predicted from screen time, study duration, and sleep using Ordinary Least Squares (OLS) regression?
4. **Inferential Hypothesis Testing:** Are observed academic score discrepancies between Low-usage (< 2h/day) and High-usage (> 4h/day) cohorts statistically significant, or attributable to random sampling variability?

---

## 2. Key Features

- **Interactive Multi-Page Streamlit Dashboard:** Clean academic typography, responsive KPI cards, and dynamic Plotly visualizations.
- **Dataset Exploration & Preprocessing Engine:**
  - Ingestion of custom CSV and Excel files.
  - Transparent missing-value imputation (mean, median, mode, or row exclusion).
  - Deduplication and out-of-bounds validity checks.
  - Outlier detection via John Tukey's IQR rule ($[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$) and Z-Score thresholding ($|Z| > 3$).
  - Social media usage tiering: **Low (< 2h/day)**, **Moderate (2–4h/day)**, and **High (> 4h/day)**.
- **Descriptive Statistics Suite:**
  - Automated summary tables: Mean, Median, Mode, Min, Max, Range, Variance ($s^2$), Standard Deviation ($s$), $Q_1$, $Q_2$, $Q_3$, IQR, Skewness, Kurtosis, and Count.
  - Pedagogical formula references with KaTeX rendering.
- **Bivariate Correlation Suite:**
  - Pearson product-moment ($r$) and Spearman rank-order ($\rho$) matrices.
  - Two-tailed exact $p$-value significance matrices.
  - Pairwise scatter plots with fitted OLS linear trendlines.
  - Explicit academic differentiation between **correlation and causation**.
- **Regression Analysis & Predictive Diagnostics:**
  - Simple and Multiple Linear Regression models.
  - 80/20 train-test splitting to prevent data leakage.
  - Model evaluation metrics on held-out test data: $R^2$, Adjusted $R^2$, MAE, MSE, and RMSE.
  - Residual diagnostics: Residuals vs. Fitted values (homoscedasticity) and residual normality histogram.
  - Interactive student score simulator with cautionary academic disclaimers.
- **Inferential Hypothesis Testing Suite:**
  - Two-Sample Independent $t$-Test (Student's / Welch's $t$-test with automated Levene's variance homogeneity check).
  - One-Way Analysis of Variance (ANOVA) with Eta-Squared ($\eta^2$) effect size and Tukey HSD post-hoc comparisons.
  - Pearson's Chi-Square ($\chi^2$) Test of Independence with Cramér's V and Cochran's rule check.
- **Insights & Academic Reporting:**
  - Automatically synthesized findings derived from live empirical calculations.
  - One-click downloadable CSV descriptive metrics.
  - Standalone, printable HTML/PDF research report.

---

## 3. Technology Stack

- **Core Language:** Python 3.11+
- **Application Framework:** Streamlit
- **Data Engineering:** Pandas, NumPy, openpyxl
- **Inferential & Applied Statistics:** SciPy (stats), Statsmodels
- **Machine Learning & OLS Regression:** Scikit-learn
- **Data Visualization:** Plotly Express, Plotly Graph Objects, Matplotlib, Seaborn
- **Automated Testing:** Pytest

---

## 4. Project Architecture

```text
social-media-student-performance-analysis/
│
├── app.py                         # Application entry point & Dashboard Overview
├── requirements.txt               # Pinned package dependencies
├── README.md                      # Comprehensive project documentation
├── .gitignore                     # Git exclusion rules (.venv, caches, logs)
│
├── .streamlit/
│   └── config.toml                # Academic theme & server configuration
│
├── data/
│   └── sample_data.csv            # 300 synthetic student records (reproducible seed)
│
├── src/
│   ├── __init__.py                # Package metadata & course identifiers
│   ├── data_processing.py         # Ingestion, validation, cleaning, & outlier detection
│   ├── statistics_analysis.py     # Descriptive statistics & distribution interpretation
│   ├── correlation_analysis.py    # Pearson/Spearman matrices, p-values, & pairwise modeling
│   ├── regression_analysis.py     # Simple/Multiple OLS regression & train-test split
│   ├── hypothesis_testing.py      # Independent t-test, One-Way ANOVA, & Chi-Square test
│   ├── visualizations.py          # Interactive Plotly figures & publication charts
│   └── report_generator.py        # Automated empirical takeaways & HTML/CSV report builder
│
├── pages/
│   ├── 1_Dataset_Explorer.py      # Uploads, data inspection, & outlier detection
│   ├── 2_Descriptive_Statistics.py# Central tendency, dispersion, & KaTeX formulas
│   ├── 3_Correlation_Analysis.py  # Correlation heatmaps, p-values, & core pairs
│   ├── 4_Regression_Analysis.py   # OLS modeling, residual diagnostics, & simulator
│   ├── 5_Hypothesis_Testing.py    # Inferential testing, decision rules, & post-hoc tests
│   └── 6_Insights_and_Reports.py  # Automated findings & downloadable research report
│
├── assets/
│   └── style.css                  # Custom academic styling & KPI scorecard CSS
│
└── tests/
    ├── test_data_processing.py    # Unit tests for cleaning, validation, & outliers
    ├── test_statistics.py         # Unit tests for descriptive formulas & metrics
    ├── test_correlation.py        # Unit tests for Pearson/Spearman & p-values
    ├── test_regression.py         # Unit tests for OLS fitting, evaluation, & prediction
    └── test_hypothesis.py         # Unit tests for t-test, ANOVA, & Chi-Square
```

---

## 5. Dataset Specification

The project includes a synthetic dataset of **300 collegiate student records** (`data/sample_data.csv`) generated with fixed pseudo-random seed (`seed=42`) using realistic statistical distributions:

| Column Name | Data Type | Permissible Range | Distribution / Description |
|:---|:---:|:---:|:---|
| `Student_ID` | String | `STU_1001` - `STU_1300` | Unique student record identifier |
| `Age` | Integer | 18 - 23 years | Undergrad age distribution |
| `Gender` | String | Male, Female, Other | Student gender identity |
| `Daily_Social_Media_Hours`| Float | 0.5 - 8.5 h/day | Right-skewed Gamma/Log-Normal distribution |
| `Most_Used_Platform` | String | Instagram, YouTube, WhatsApp, etc. | Primary platform preference |
| `Study_Hours_Per_Day` | Float | 1.0 - 8.5 h/day | Inversely correlated with screen time ($r \approx -0.42$) |
| `Sleep_Hours` | Float | 4.5 - 9.5 h/day | Mild negative correlation with screen time |
| `Academic_Percentage` | Float | 45.0% - 98.0% | Normal distribution influenced by study, screen, and sleep |
| `Concentration_Level` | Integer | 1 - 10 scale | Self-reported focus capability |
| `Stress_Level` | Integer | 1 - 10 scale | Self-reported psychological tension |
| `Usage_Category` | String | Low, Moderate, High | Derived usage tier (<2h, 2-4h, >4h) |

> **Ethical & Academic Notice:** The dataset is synthetic and constructed strictly for statistical modeling demonstration. It contains no personally identifiable student records.

---

## 6. Mathematical Formulations & Statistical Engine

### A. Descriptive Statistics
- **Sample Mean:**
  $$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$
- **Sample Variance (Bessel's Correction for Unbiased Estimation):**
  $$s^2 = \frac{1}{n - 1}\sum_{i=1}^{n} (x_i - \bar{x})^2$$
- **Sample Standard Deviation:**
  $$s = \sqrt{s^2}$$
- **Interquartile Range (IQR):**
  $$\text{IQR} = Q_3 - Q_1$$
- **Fisher-Pearson Sample Skewness:**
  $$g_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{s}\right)^3$$

### B. Correlation Analysis
- **Pearson Product-Moment Correlation:**
  $$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$
- **Spearman Rank Correlation:**
  $$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$
- **Association vs. Causation:** Statistically significant correlation ($p < 0.05$) demonstrates co-variation; it does not establish causal directionality without experimental controls.

### C. Ordinary Least Squares (OLS) Regression
- **Multiple Linear Model:**
  $$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_k x_k + \epsilon$$
- **Goodness-of-Fit ($R^2$):**
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
- **Adjusted $R^2$ (Penalizing Excess Predictors):**
  $$R^2_{\text{adj}} = 1 - \frac{(1 - R^2)(n - 1)}{n - k - 1}$$
- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

### D. Inferential Hypothesis Testing
- **Two-Sample $t$-Statistic (Welch's Formulation for Unequal Variances):**
  $$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$
- **One-Way ANOVA $F$-Statistic:**
  $$F = \frac{\text{MS}_{\text{Between}}}{\text{MS}_{\text{Within}}} = \frac{\text{SS}_{\text{Between}} / (k - 1)}{\text{SS}_{\text{Within}} / (n - k)}$$
- **Chi-Square ($\chi^2$) Test of Independence:**
  $$\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad E_{ij} = \frac{R_i \times C_j}{N}$$

---

## 7. Installation & Local Execution

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Step-by-Step Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ShivayAgarwal08/social-media-student-performance-analysis.git
   cd social-media-student-performance-analysis
   ```

2. **Create and Activate a Virtual Environment:**
   - **Windows:**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit Application:**
   ```bash
   streamlit run app.py
   ```
   Open your browser and navigate to `http://localhost:8501`.

---

## 8. Automated Testing & Verification

The test suite contains **25 automated unit and statistical integration tests** spanning data cleaning, validation, outlier detection, descriptive formulas, correlation, OLS regression, and hypothesis testing.

Run all tests with detailed verbosity:
```bash
python -m pytest tests/ -v
```

Execution Summary:
```text
tests/test_correlation.py::test_perfect_positive_correlation PASSED
tests/test_correlation.py::test_perfect_negative_correlation PASSED
tests/test_correlation.py::test_compute_correlation_matrix PASSED
tests/test_correlation.py::test_insufficient_sample_size PASSED
tests/test_data_processing.py::test_categorize_social_media PASSED
tests/test_data_processing.py::test_generate_sample_dataset PASSED
tests/test_data_processing.py::test_validate_dataset_valid PASSED
tests/test_data_processing.py::test_validate_dataset_missing_columns PASSED
tests/test_data_processing.py::test_validate_dataset_invalid_values PASSED
tests/test_data_processing.py::test_clean_dataset_imputation_and_duplicates PASSED
tests/test_data_processing.py::test_clean_dataset_drop_strategy PASSED
tests/test_data_processing.py::test_detect_outliers_iqr PASSED
tests/test_data_processing.py::test_detect_outliers_zscore PASSED
tests/test_data_processing.py::test_get_dataset_summary PASSED
tests/test_hypothesis.py::test_independent_t_test_significant PASSED
tests/test_hypothesis.py::test_independent_t_test_non_significant PASSED
tests/test_hypothesis.py::test_one_way_anova_significant PASSED
tests/test_hypothesis.py::test_chi_square_test PASSED
tests/test_regression.py::test_simple_linear_regression_fit PASSED
tests/test_regression.py::test_multiple_linear_regression_fit PASSED
tests/test_regression.py::test_predict_academic_performance PASSED
tests/test_statistics.py::test_calculate_descriptive_stats_known_data PASSED
tests/test_statistics.py::test_calculate_descriptive_stats_empty PASSED
tests/test_statistics.py::test_compute_summary_table PASSED
tests/test_statistics.py::test_interpret_distribution PASSED

============================= 25 passed in 44.19s =============================
```

---

## 9. Research Scope & Limitations

1. **Synthetic Nature:** The default 300 student records are simulated for academic curriculum demonstration.
2. **Cross-Sectional Observations:** Observations demonstrate correlational association rather than verified clinical causality.
3. **Unmodeled Confounders:** Student learning outcomes are mediated by prior academic preparation, instructional quality, economic circumstances, and mental health factors beyond linear screen time.

---

## 10. Course & Academic Inquiries

- **Author / Student:** Shivay Agarwal
- **Course:** Probability and Statistics Using Python (`26B16CA212`)
- **Institution:** Jaypee Institute of Information Technology (JIIT), A-10, Sector-62, Noida, Uttar Pradesh, India
