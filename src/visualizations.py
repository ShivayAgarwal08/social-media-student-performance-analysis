"""Visualization Module.

Generates interactive, publication-quality Plotly figures for statistical analysis,
distributions, bivariate correlations, regression diagnostics, and group comparisons.

Course: Probability and Statistics Using Python (26B16CA212)
Institution: Jaypee Institute of Information Technology, Noida
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Academic styling palette
COLOR_PRIMARY = "#1E3A8A"     # Deep JIIT Navy
COLOR_SECONDARY = "#0284C7"   # Sky Blue
COLOR_ACCENT = "#D97706"      # Warm Amber
COLOR_SUCCESS = "#059669"     # Emerald Green
COLOR_DANGER = "#DC2626"      # Crimson
COLOR_MUTED = "#64748B"       # Slate
TEMPLATE = "plotly_white"


def plot_usage_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot distribution of Daily Social Media Hours with category demarcations."""
    fig = px.histogram(
        df,
        x="Daily_Social_Media_Hours",
        nbins=25,
        marginal="box",
        title="Distribution of Daily Social Media Hours",
        labels={"Daily_Social_Media_Hours": "Daily Social Media Usage (Hours/Day)"},
        color_discrete_sequence=[COLOR_PRIMARY],
        template=TEMPLATE,
    )
    # Add vertical category threshold lines
    fig.add_vline(x=2.0, line_dash="dash", line_color=COLOR_SUCCESS, annotation_text="Low (<2h)")
    fig.add_vline(x=4.0, line_dash="dash", line_color=COLOR_DANGER, annotation_text="High (>4h)")
    fig.update_layout(bargap=0.08, height=450)
    return fig


def plot_platform_usage(df: pd.DataFrame) -> go.Figure:
    """Plot student count and average usage hours across social media platforms."""
    platform_agg = df.groupby("Most_Used_Platform").agg(
        Student_Count=("Student_ID", "count"),
        Avg_Daily_Hours=("Daily_Social_Media_Hours", "mean"),
    ).reset_index().sort_values(by="Student_Count", ascending=False)

    fig = px.bar(
        platform_agg,
        x="Most_Used_Platform",
        y="Student_Count",
        color="Avg_Daily_Hours",
        color_continuous_scale="Blues",
        title="Most Used Platforms by Student Frequency & Mean Usage",
        labels={
            "Most_Used_Platform": "Social Media Platform",
            "Student_Count": "Number of Students",
            "Avg_Daily_Hours": "Mean Daily Hours",
        },
        template=TEMPLATE,
    )
    fig.update_layout(height=420)
    return fig


def plot_academic_by_category(df: pd.DataFrame) -> go.Figure:
    """Plot mean Academic Percentage across Social Media Usage Categories with error bars."""
    agg = df.groupby("Usage_Category")["Academic_Percentage"].agg(
        Mean="mean", Std="std", Count="count"
    ).reindex(["Low", "Moderate", "High"]).reset_index()

    fig = px.bar(
        agg,
        x="Usage_Category",
        y="Mean",
        error_y="Std",
        color="Usage_Category",
        color_discrete_map={"Low": COLOR_SUCCESS, "Moderate": COLOR_SECONDARY, "High": COLOR_DANGER},
        title="Mean Academic Performance (%) by Social Media Usage Category",
        labels={"Usage_Category": "Usage Category", "Mean": "Mean Academic Percentage (%)"},
        template=TEMPLATE,
        text_auto=".1f",
    )
    fig.update_layout(showlegend=False, yaxis_range=[0, 100], height=420)
    return fig


def plot_scatter_with_trend(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    color_by: Optional[str] = "Usage_Category",
) -> go.Figure:
    """Scatter plot with Ordinary Least Squares (OLS) trendline and interactive tooltips."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_by if color_by in df.columns else None,
        trendline="ols",
        trendline_color_override="#0F172A",
        title=title,
        labels={x_col: x_label, y_col: y_label},
        template=TEMPLATE,
        color_discrete_map={"Low": COLOR_SUCCESS, "Moderate": COLOR_SECONDARY, "High": COLOR_DANGER},
        hover_data=["Student_ID", "Study_Hours_Per_Day", "Sleep_Hours"] if "Student_ID" in df.columns else None,
    )
    fig.update_traces(marker=dict(size=8, opacity=0.75))
    fig.update_layout(height=450)
    return fig


def plot_correlation_heatmap(corr_df: pd.DataFrame, title: str = "Correlation Heatmap") -> go.Figure:
    """Publication-ready annotated correlation heatmap."""
    z_values = corr_df.values
    x_labels = list(corr_df.columns)
    y_labels = list(corr_df.index)

    text_values = np.around(z_values, decimals=2).astype(str)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            text=text_values,
            texttemplate="%{text}",
            textfont={"size": 11, "color": "white"},
            colorscale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            colorbar=dict(title="r"),
        )
    )
    fig.update_layout(
        title=title,
        template=TEMPLATE,
        height=520,
        xaxis=dict(tickangle=35),
    )
    return fig


def plot_regression_actual_vs_predicted(y_actual: np.ndarray, y_pred: np.ndarray, title: str = "Actual vs. Predicted Academic Percentage") -> go.Figure:
    """Plot actual vs predicted scatter with ideal 45-degree parity line."""
    fig = go.Figure()

    # Scatter of test predictions
    fig.add_trace(
        go.Scatter(
            x=y_actual,
            y=y_pred,
            mode="markers",
            name="Predictions",
            marker=dict(color=COLOR_SECONDARY, size=8, opacity=0.8),
        )
    )

    # 45-degree ideal parity line
    min_val = min(float(np.min(y_actual)), float(np.min(y_pred))) - 2
    max_val = max(float(np.max(y_actual)), float(np.max(y_pred))) + 2
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal Parity (y = x)",
            line=dict(color=COLOR_DANGER, dash="dash", width=2),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Actual Academic Percentage (%)",
        yaxis_title="Predicted Academic Percentage (%)",
        template=TEMPLATE,
        height=450,
    )
    return fig


def plot_regression_residuals(y_pred: np.ndarray, residuals: np.ndarray) -> go.Figure:
    """Plot model residuals against predicted values for homoscedasticity evaluation."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            name="Residuals (e = y - ŷ)",
            marker=dict(color=COLOR_PRIMARY, size=8, opacity=0.75),
        )
    )

    fig.add_hline(y=0.0, line_dash="dash", line_color=COLOR_DANGER, annotation_text="Zero Residual Line")

    fig.update_layout(
        title="Residuals vs. Fitted Values (Homoscedasticity Check)",
        xaxis_title="Fitted (Predicted) Academic Percentage (%)",
        yaxis_title="Residual (Actual - Predicted)",
        template=TEMPLATE,
        height=420,
    )
    return fig


def plot_residual_distribution(residuals: np.ndarray) -> go.Figure:
    """Histogram of regression residuals to check normality assumption."""
    fig = px.histogram(
        x=residuals,
        nbins=20,
        marginal="rug",
        title="Distribution of Residuals (Normality of Error Terms)",
        labels={"x": "Residual Value (e = y - ŷ)"},
        color_discrete_sequence=[COLOR_SECONDARY],
        template=TEMPLATE,
    )
    fig.add_vline(x=0.0, line_dash="dash", line_color=COLOR_DANGER)
    fig.update_layout(bargap=0.08, height=420)
    return fig


def plot_metric_distribution(df: pd.DataFrame, column: str, title: str) -> go.Figure:
    """Univariate distribution plot with box plot marginal for any numeric variable."""
    fig = px.histogram(
        df,
        x=column,
        nbins=20,
        marginal="box",
        title=title,
        color_discrete_sequence=[COLOR_PRIMARY],
        template=TEMPLATE,
    )
    fig.update_layout(bargap=0.08, height=400)
    return fig
