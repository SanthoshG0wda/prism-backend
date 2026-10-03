"""
Tests for profiling and data quality checks.
"""

import pandas as pd
import pytest
from src.tools.profiling import check_data_quality, profile_dataframe


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "region": ["North", "South", "East", "West", "North"],
        "revenue": [1000.0, 2500.0, 3000.0, 1500.0, 1000.0],
        "units": [10, 25, 30, 15, 10],
        "category": ["A", "B", "A", None, "A"],
    })


def test_profile_dataframe(sample_df):
    meta = profile_dataframe(sample_df, "test_table")
    assert meta.table_name == "test_table"
    assert meta.row_count == 5
    assert meta.column_count == 4
    assert len(meta.columns) == 4

    col_map = {c.name: c for c in meta.columns}
    assert "revenue" in col_map
    assert col_map["revenue"].min_value == 1000.0
    assert col_map["revenue"].max_value == 3000.0
    assert col_map["category"].null_count == 1
    assert col_map["category"].null_percentage == 20.0


def test_check_data_quality(sample_df):
    report = check_data_quality(sample_df, "test_table")
    assert report.table_name == "test_table"
    assert report.row_count == 5
    # 1 duplicate row (North, 1000, 10, A)
    assert report.duplicate_rows == 1
    assert report.missing_cells == 1
    assert report.completeness_score < 100.0
    assert len(report.quality_issues) > 0
