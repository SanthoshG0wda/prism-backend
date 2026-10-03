"""
Tests for deterministic analytical functions.
"""

import pandas as pd
import pytest
from src.tools.analysis import (
    aggregate_metric,
    calculate_correlation,
    time_series_trend,
    top_k_analysis,
)


@pytest.fixture
def sample_sales_df():
    return pd.DataFrame({
        "region": ["North", "South", "North", "South", "East"],
        "product": ["Laptop", "Mouse", "Keyboard", "Laptop", "Mouse"],
        "revenue": [1000.0, 50.0, 100.0, 1200.0, 75.0],
        "profit": [200.0, 15.0, 30.0, 250.0, 20.0],
        "date": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15", "2024-03-01"],
    })


def test_top_k_analysis(sample_sales_df):
    res = top_k_analysis(sample_sales_df, group_col="region", metric_col="revenue", k=2, ascending=False)
    assert len(res["records"]) == 2
    # South total = 1200 + 50 = 1250, North = 1000 + 100 = 1100
    assert res["records"][0]["region"] == "South"
    assert res["records"][0]["revenue"] == 1250.0
    assert "df.groupby" in res["pandas_code"]


def test_aggregate_metric(sample_sales_df):
    res = aggregate_metric(sample_sales_df, group_by=["region"], metric_col="revenue", agg_func="sum")
    assert len(res["records"]) == 3
    regions = {r["region"] for r in res["records"]}
    assert regions == {"North", "South", "East"}


def test_calculate_correlation(sample_sales_df):
    res = calculate_correlation(sample_sales_df, numeric_cols=["revenue", "profit"])
    assert "matrix" in res
    assert len(res["top_correlated_pairs"]) >= 1
    pair = res["top_correlated_pairs"][0]
    assert pair["correlation"] > 0.9  # Revenue and profit are strongly correlated in sample


def test_time_series_trend(sample_sales_df):
    res = time_series_trend(sample_sales_df, date_col="date", metric_col="revenue", freq="ME")
    assert len(res["records"]) >= 3
