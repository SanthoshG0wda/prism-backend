"""
Tests for Plotly chart generation.
"""

import pandas as pd
import pytest
from src.tools.charts import generate_chart


@pytest.fixture
def chart_data():
    return pd.DataFrame({
        "region": ["North", "South", "East", "West"],
        "revenue": [5000, 7000, 3000, 9000],
        "profit": [1000, 1500, 500, 2200],
    })


def test_bar_chart_generation(chart_data):
    res = generate_chart(chart_data, chart_type="bar", x="region", y="revenue", title="Sales by Region")
    assert res["chart_type"] == "bar"
    assert "plotly_spec" in res
    assert "data" in res["plotly_spec"]
    assert "px.bar" in res["pandas_code"]


def test_scatter_chart_generation(chart_data):
    res = generate_chart(chart_data, chart_type="scatter", x="revenue", y="profit", title="Profit vs Revenue")
    assert res["chart_type"] == "scatter"
    assert "plotly_spec" in res


def test_invalid_chart_type(chart_data):
    with pytest.raises(ValueError, match="Unsupported chart type"):
        generate_chart(chart_data, chart_type="invalid_type", x="region")
