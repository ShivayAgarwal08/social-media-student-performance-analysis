"""Tests for Hypothesis Testing Module.

Course: Probability and Statistics Using Python (26B16CA212)
Institution: Jaypee Institute of Information Technology, Noida
"""

import numpy as np
import pandas as pd
import pytest

from src.hypothesis_testing import (
    run_chi_square_test,
    run_independent_t_test,
    run_one_way_anova,
)


def test_independent_t_test_significant():
    np.random.seed(42)
    # Distinct groups
    g1 = pd.Series(np.random.normal(85, 4, 50))
    g2 = pd.Series(np.random.normal(65, 4, 50))

    res = run_independent_t_test(g1, g2, "High Group", "Low Group", "Percentage", alpha=0.05)

    assert bool(res["reject_null"]) is True
    assert res["p_value"] < 0.001
    assert res["t_statistic"] > 0
    assert "Reject the Null Hypothesis" in res["decision"]


def test_independent_t_test_non_significant():
    np.random.seed(42)
    # Identical population distributions
    g1 = pd.Series(np.random.normal(70, 5, 40))
    g2 = pd.Series(np.random.normal(70.2, 5, 40))

    res = run_independent_t_test(g1, g2, "Group A", "Group B", "Percentage", alpha=0.05)

    assert res["reject_null"] is False
    assert res["p_value"] > 0.05
    assert "Fail to Reject" in res["decision"]


def test_one_way_anova_significant():
    np.random.seed(42)
    df = pd.DataFrame({
        "Category": ["Low"] * 30 + ["Moderate"] * 30 + ["High"] * 30,
        "Score": np.concatenate([
            np.random.normal(85, 3, 30),
            np.random.normal(75, 3, 30),
            np.random.normal(60, 3, 30),
        ]),
    })

    res = run_one_way_anova(df, group_col="Category", metric_col="Score", alpha=0.05)

    assert bool(res["reject_null"]) is True
    assert res["p_value"] < 0.001
    assert res["df_between"] == 2
    assert res["df_within"] == 87
    assert res["eta_squared"] > 0.5


def test_chi_square_test():
    # Contingency table with known dependent categories
    df = pd.DataFrame({
        "Tier": ["Low"] * 40 + ["High"] * 40,
        "Platform": ["Instagram"] * 35 + ["YouTube"] * 5 + ["Instagram"] * 5 + ["YouTube"] * 35,
    })

    res = run_chi_square_test(df, "Tier", "Platform", alpha=0.05)

    assert bool(res["reject_null"]) is True
    assert res["p_value"] < 0.001
    assert res["degrees_of_freedom"] == 1
    assert res["cramers_v"] > 0.5
