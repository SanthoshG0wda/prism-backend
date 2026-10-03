"""
Tests for safe DuckDB SQL execution and read-only validation.
"""

import duckdb
import pandas as pd
import pytest
from src.tools.sql import execute_sql, validate_sql_safety


@pytest.fixture
def duck_conn():
    conn = duckdb.connect(database=":memory:")
    df = pd.DataFrame({
        "region": ["North", "South", "East"],
        "sales": [500, 700, 1200]
    })
    conn.register("sales_data", df)
    return conn


def test_safe_select_query(duck_conn):
    query = "SELECT region, sales FROM sales_data WHERE sales > 600 ORDER BY sales DESC"
    result = execute_sql(query, duck_conn)
    assert result["row_count"] == 2
    assert len(result["records"]) == 2
    assert result["records"][0]["region"] == "East"
    assert result["records"][0]["sales"] == 1200


def test_multi_table_join(duck_conn):
    cust_df = pd.DataFrame({
        "region": ["North", "South", "East"],
        "manager": ["Alice", "Bob", "Charlie"],
    })
    duck_conn.register("regions_meta", cust_df)
    query = """
    SELECT s.region, s.sales, r.manager 
    FROM sales_data s 
    JOIN regions_meta r ON s.region = r.region 
    WHERE s.sales > 600
    """
    result = execute_sql(query, duck_conn)
    assert result["row_count"] == 2
    assert "manager" in result["columns"]


def test_disallowed_sql_keywords():
    with pytest.raises(ValueError, match="Security violation"):
        validate_sql_safety("DROP TABLE sales_data")

    with pytest.raises(ValueError, match="Security violation"):
        validate_sql_safety("DELETE FROM sales_data WHERE sales > 0")

    with pytest.raises(ValueError, match="Security violation"):
        validate_sql_safety("INSERT INTO sales_data VALUES ('West', 400)")

    with pytest.raises(ValueError, match="Security violation"):
        validate_sql_safety("UPDATE sales_data SET sales = 0")


def test_empty_query():
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_sql_safety("   ")
