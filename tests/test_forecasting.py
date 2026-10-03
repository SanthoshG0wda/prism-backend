"""
Tests for time-series forecasting tool.
"""

import pandas as pd
import pytest
from src.tools.forecasting import forecast_metric


@pytest.fixture
def monthly_sales_df():
    return pd.DataFrame({
        "date": ["2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15", "2024-05-15"],
        "revenue": [10000.0, 12000.0, 15000.0, 18000.0, 22000.0],
    })


def test_forecast_metric(monthly_sales_df):
    res = forecast_metric(monthly_sales_df, date_col="date", metric_col="revenue", periods=3)
    assert res["metric_col"] == "revenue"
    assert len(res["forecast_records"]) == 3

    # Check forecast bounds
    first = res["forecast_records"][0]
    assert "forecast" in first
    assert "lower_bound_95" in first
    assert "upper_bound_95" in first
    assert first["lower_bound_95"] <= first["forecast"] <= first["upper_bound_95"]

    # Check plotly spec exists
    assert "plotly_spec" in res
    assert "data" in res["plotly_spec"]


def test_forecast_insufficient_data():
    short_df = pd.DataFrame({
        "date": ["2024-01-01", "2024-02-01"],
        "revenue": [100, 200],
    })
    with pytest.raises(ValueError, match="at least 3 historical"):
        forecast_metric(short_df, date_col="date", metric_col="revenue")
