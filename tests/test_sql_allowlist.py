"""
Table-driven allowlist tests for the sqlglot-based SQL validator.

ATTACKS must raise (ValueError/SQLValidationError, "Security violation").
VALID queries must execute and return rows — including joins, CTEs,
window functions, and date_trunc.
"""

import duckdb
import pandas as pd
import pytest

from src.tools.sql import (
    MAX_SQL_LENGTH,
    execute_sql,
    harden_duckdb_connection,
    quote_ident,
    validate_sql_safety,
)

ATTACKS = [
    # File exfiltration via table functions (keyword blocklists miss these)
    "SELECT * FROM read_csv_auto('/app/.env')",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM read_parquet('x.parquet')",
    "SELECT * FROM read_json_auto('x.json')",
    "SELECT * FROM glob('*')",
    "SELECT * FROM pragma_database_list()",
    "SELECT * FROM duckdb_tables()",
    # Banned functions in SELECT position (info disclosure / nondeterminism)
    "SELECT pragma_version()",
    "SELECT current_setting('memory_limit')",
    "SELECT version()",
    "SELECT current_database()",
    "SELECT md5('x')",
    "SELECT random()",
    # Statements that must never run
    "CALL pragma_database_list()",
    "COPY sales TO '/tmp/x.csv'",
    "ATTACH 'f.db' AS aux",
    "DETACH aux",
    "INSTALL httpfs",
    "LOAD spatial",
    "SET memory_limit='1GB'",
    "VACUUM sales",
    "CHECKPOINT",
    "DROP TABLE sales",
    "DELETE FROM sales",
    "INSERT INTO sales VALUES ('X', 1)",
    "UPDATE sales SET revenue = 0",
    "CREATE TABLE evil AS SELECT 1",
    "SELECT * INTO newt FROM sales",
    "SELECT a FROM sales UNION SELECT a FROM sales",
    "SELECT 1; DROP TABLE sales",
    "SELECT 1; SELECT 2",
    # Unknown / malformed references
    "SELECT password FROM users",
    "SELECT nope FROM sales",
    "SELECT * FROM 'secret.csv'",
    'SELECT "weird col" FROM sales',
    "",
    "   ",
    "-- just a comment",
    "SELECT * FROM " + "sales_data_very_long_name_" * 600,
]

VALID = [
    "SELECT region, revenue FROM sales WHERE revenue > 600 ORDER BY revenue DESC",
    "SELECT s.region, SUM(s.revenue) AS total FROM sales s GROUP BY s.region ORDER BY total DESC LIMIT 5",
    "SELECT s.region, r.manager FROM sales s JOIN regions_meta r ON s.region = r.region WHERE s.revenue > 600",
    "WITH ranked AS (SELECT region, SUM(revenue) AS total FROM sales GROUP BY region) SELECT region, total FROM ranked ORDER BY total DESC",
    "SELECT region, ROW_NUMBER() OVER (PARTITION BY region ORDER BY revenue DESC) AS rn FROM sales",
    "SELECT region, LAG(revenue) OVER (ORDER BY revenue) AS prev FROM sales",
    "SELECT date_trunc('month', day) AS m, SUM(revenue) AS total FROM sales GROUP BY 1 ORDER BY 1",
    "SELECT strftime(day, '%Y-%m') AS m, COUNT(*) AS n FROM sales GROUP BY m",
    "SELECT * FROM sales WHERE region IN ('North', 'South') AND revenue IS NOT NULL",
    "SELECT CASE WHEN revenue > 5000 THEN 'big' ELSE 'small' END AS size, CAST(revenue AS DOUBLE) AS r FROM sales",
    'SELECT "region", SUM("revenue") AS "total" FROM "sales" GROUP BY "region"',
    "SELECT DISTINCT region FROM sales ORDER BY region",
    "SELECT region FROM sales WHERE EXISTS (SELECT 1 FROM regions_meta WHERE regions_meta.region = sales.region)",
    "SELECT * FROM sales LIMIT 100",
]


@pytest.fixture
def catalog_conn():
    conn = duckdb.connect(database=":memory:")
    harden_duckdb_connection(conn)
    conn.register(
        "sales",
        pd.DataFrame({
            "region": ["North", "South", "East", "West", "North"],
            "revenue": [5000.0, 7000.0, 3000.0, 100000.0, 6000.0],
            "day": pd.to_datetime(["2024-01-15"] * 5),
        }),
    )
    conn.register(
        "regions_meta",
        pd.DataFrame({
            "region": ["North", "South", "East", "West"],
            "manager": ["Alice", "Bob", "Charlie", "Dana"],
        }),
    )
    return conn


@pytest.mark.parametrize("query", ATTACKS)
def test_attacks_rejected_without_touching_engine(query):
    with pytest.raises(ValueError, match="[Ss]ecurity violation|cannot be empty|exceeds"):
        validate_sql_safety(
            query,
            allowed_tables=["sales", "regions_meta"],
            allowed_columns=["region", "revenue", "manager", "day"],
        )


@pytest.mark.parametrize("query", VALID)
def test_valid_queries_execute(catalog_conn, query):
    result = execute_sql(query, catalog_conn)
    assert result["row_count"] >= 1
    assert len(result["records"]) >= 1


def test_join_returns_manager(catalog_conn):
    result = execute_sql(
        "SELECT s.region, r.manager FROM sales s "
        "JOIN regions_meta r ON s.region = r.region ORDER BY s.region",
        catalog_conn,
    )
    assert "manager" in result["columns"]
    assert result["row_count"] == 5


def test_cte_and_window_execute(catalog_conn):
    result = execute_sql(
        "WITH ranked AS (SELECT region, SUM(revenue) AS total FROM sales GROUP BY region) "
        "SELECT region, total, ROW_NUMBER() OVER (ORDER BY total DESC) AS rn FROM ranked",
        catalog_conn,
    )
    assert result["records"][0]["region"] == "West"
    assert result["records"][0]["rn"] == 1


def test_quote_ident_escapes():
    assert quote_ident("region") == '"region"'
    assert quote_ident('a"; DROP TABLE t; --') == '"a""; DROP TABLE t; --"'
    with pytest.raises(ValueError):
        quote_ident("")
    with pytest.raises(ValueError):
        quote_ident(None)


def test_malicious_header_cannot_inject(catalog_conn):
    evil = 'a"; DROP TABLE t; --'
    df = pd.DataFrame({evil: [1, 2], "revenue": [10.0, 20.0]})
    catalog_conn.register("evil_t", df)
    quoted = quote_ident(evil)
    # Quoted, the header is one inert identifier; the validator rejects the
    # raw quote chars, and the engine only ever sees a failed column lookup.
    with pytest.raises(ValueError, match="[Ss]ecurity violation"):
        validate_sql_safety(
            f"SELECT {quoted} FROM evil_t", allowed_tables=["evil_t"]
        )
    tables_before = {r[0] for r in catalog_conn.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}
    with pytest.raises(Exception):
        # Even past validation shape, unquoted hostile text is just an error
        # (DuckDB raises its own Error types, not a successful injection).
        catalog_conn.execute(f"SELECT {evil} FROM evil_t").fetchall()
    tables_after = {r[0] for r in catalog_conn.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}
    assert tables_before == tables_after


def test_hardening_blocks_raw_file_access_and_relock():
    conn = duckdb.connect(database=":memory:")
    harden_duckdb_connection(conn)
    with pytest.raises(Exception):
        conn.execute("SELECT * FROM read_csv_auto('/etc/hostname')").fetchall()
    with pytest.raises(Exception):
        conn.execute("SET memory_limit='2GB'")
    # ...while normal analytics keep working
    conn.register("t", pd.DataFrame({"a": [1, 2, 3]}))
    assert conn.execute("SELECT SUM(a) FROM t").fetchall() == [(6,)]


def test_query_timeout_cancels():
    import threading
    from src.tools.sql import _execute_with_timeout

    interrupted = threading.Event()

    class BlockingConn:
        def execute(self, query):
            interrupted.wait(timeout=5)
            return self

        def df(self):
            return None

        def interrupt(self):
            interrupted.set()

    with pytest.raises(RuntimeError, match="timeout"):
        _execute_with_timeout(BlockingConn(), "SELECT 1", timeout_seconds=0.05)
    assert interrupted.is_set()


def test_oversized_query_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        validate_sql_safety("SELECT " + "1," * (MAX_SQL_LENGTH // 2) + "1 FROM t",
                            allowed_tables=["t"])
