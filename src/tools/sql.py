"""
Safe SQL execution using DuckDB for deterministic in-memory analytics.

Security model (allowlist, DuckDB dialect via sqlglot):
1. Exactly one statement, which must be SELECT (or WITH...SELECT, which parses
   as Select). Everything else — CALL, VACUUM, CHECKPOINT, SET, DDL, DML,
   stacked statements — is rejected before touching the engine.
2. The AST is walked: rejected node types (UNION, VALUES, PIVOT, LATERAL,
   UNNEST, table samples, locks, commands, …) fail closed.
3. Every table reference must be a registered catalog table (or a CTE defined
   in the query). In particular, table-valued functions (read_csv*,
   read_parquet*, read_json*, glob, pragma_*, duckdb_*, …) parse as
   Table(this=Anonymous) and are rejected — closing the file-read hole that a
   keyword blocklist cannot cover. Table/column identifiers must additionally
   match ^[A-Za-z_][A-Za-z0-9_]*$ (all registered names are sanitized to this).
4. Every function call must be in a small allowlist of aggregate / scalar /
   date / string / window functions (checked via sqlglot's canonical sql_name,
   plus the raw name for Anonymous nodes).
5. Engine hardening (defense in depth): hardened connections disable external
   file access and lock configuration; queries run under a client-side timeout
   with conn.interrupt() (DuckDB exposes no statement-timeout knob).
"""

import concurrent.futures
import os
import re
from typing import Any, Collection, Dict, List, Optional, Set, Tuple

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp

from src.tools.registry import register_tool
from src.utils.logging import get_logger

logger = get_logger(__name__)

MAX_SQL_LENGTH = 10_000
MAX_ROWS_HARD_CAP = 5000
DEFAULT_QUERY_TIMEOUT_S = 30
DUCKDB_MEMORY_LIMIT = "1GB"
DUCKDB_THREADS = 4

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Canonical sqlglot sql_name() values (upper-case) plus raw names of Anonymous
# nodes we deliberately support. Pure, deterministic, read-only functions only:
# no version(), current_setting(), random(), md5(), regex engines, etc.
ALLOWED_FUNCTIONS = frozenset({
    # Aggregates
    "SUM", "AVG", "MEAN", "COUNT", "COUNT_IF", "MIN", "MAX", "MEDIAN", "MODE",
    "QUANTILE", "STDDEV", "STDDEV_POP", "STDDEV_SAMP", "VARIANCE",
    "VARIANCE_POP", "VARIANCE_SAMP", "VAR_POP", "VAR_SAMP", "STRING_AGG",
    "GROUP_CONCAT", "ARRAY_AGG", "FIRST", "LAST", "ARBITRARY",
    "APPROX_COUNT_DISTINCT", "APPROX_DISTINCT", "CORR", "SKEWNESS", "KURTOSIS",
    # Math / conditional / cast
    "ABS", "ROUND", "CEIL", "CEILING", "FLOOR", "SQRT", "POWER", "EXP", "LN",
    "LOG", "COALESCE", "NULLIF", "GREATEST", "LEAST", "IFNULL", "SIGN",
    "CAST", "TRY_CAST", "CASE", "IF", "EXISTS",
    # String
    "UPPER", "LOWER", "TRIM", "LTRIM", "RTRIM", "SUBSTRING", "SUBSTR", "LEFT",
    "RIGHT", "CONCAT", "CONCAT_WS", "REPLACE", "LENGTH", "LEN", "STARTS_WITH",
    "ENDS_WITH", "CONTAINS", "STRPOS", "POSITION", "SPLIT_PART",
    # Date / time
    "STRFTIME", "TIME_TO_STR", "DATE_TRUNC", "TIMESTAMP_TRUNC", "DATE_PART",
    "YEAR", "MONTH", "DAY", "EPOCH", "TIME_TO_UNIX", "NOW", "CURRENT_DATE",
    "CURRENT_TIMESTAMP", "CURRENT_TIME", "TODAY", "DATEDIFF", "DATE_DIFF",
    "EXTRACT",
    # Window
    "ROW_NUMBER", "RANK", "DENSE_RANK", "LAG", "LEAD", "FIRST_VALUE",
    "LAST_VALUE", "NTH_VALUE", "NTILE", "CUME_DIST", "PERCENT_RANK",
})


def _rejected_node_types() -> Tuple[type, ...]:
    """Node classes that never appear in a legitimate read-only SELECT."""
    candidates = [
        "Command", "Create", "Drop", "AlterTable", "TruncateTable", "Delete",
        "Insert", "Update", "Copy", "Attach", "Detach", "Load", "Cache",
        "Set", "Pragma", "Transaction", "Vacuum", "Checkpoint", "Grant",
        "Union", "Except", "Intersect", "Values", "Lateral", "Pivot",
        "Unnest", "TableSample", "TableFunc", "Lock",
    ]
    resolved = []
    for name in candidates:
        cls = getattr(exp, name, None)
        if cls is not None:
            resolved.append(cls)
    return tuple(resolved)


REJECTED_NODE_TYPES = _rejected_node_types()


class SQLValidationError(ValueError):
    """Raised when a SQL query fails allowlist validation (a ValueError)."""


def quote_ident(name: str) -> str:
    """Quote an identifier for DuckDB, doubling embedded quotes.

    Use everywhere SQL text is built from catalog/column names so hostile
    headers (e.g. 'a"; DROP TABLE t; --') become a single inert identifier.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Invalid SQL identifier.")
    return '"' + name.replace('"', '""') + '"'


def _func_name(fn: exp.Func) -> str:
    if isinstance(fn, exp.Anonymous):
        raw = fn.this
        return raw.upper() if isinstance(raw, str) else "ANONYMOUS"
    try:
        return fn.sql_name().upper()
    except Exception:
        return type(fn).__name__.upper()


def validate_sql_safety(
    query: str,
    allowed_tables: Optional[Collection[str]] = None,
    allowed_columns: Optional[Collection[str]] = None,
) -> None:
    """Validates a SQL query against the allowlist. Raises SQLValidationError.

    allowed_tables: catalog table names permitted (case-insensitive). When None,
    table membership is skipped (callers are expected to introspect instead).
    allowed_columns: known column names (case-insensitive); unknown unqualified
    columns and columns of real tables are rejected. CTE/derived-table
    references and SELECT aliases are permitted.
    """
    if not isinstance(query, str) or not query.strip():
        raise SQLValidationError("Security violation: SQL query cannot be empty.")
    if len(query) > MAX_SQL_LENGTH:
        raise SQLValidationError(
            f"Security violation: query exceeds {MAX_SQL_LENGTH} characters."
        )

    try:
        parsed = sqlglot.parse(query, read="duckdb")
    except Exception as exc:
        raise SQLValidationError(
            f"Security violation: query failed to parse as DuckDB SQL: {str(exc)[:200]}"
        ) from exc

    statements = [s for s in parsed if s is not None]
    if not statements:
        raise SQLValidationError("Security violation: SQL query cannot be empty.")
    if len(statements) != 1:
        raise SQLValidationError(
            "Security violation: exactly one SELECT statement is required; "
            "multiple/stacked statements are forbidden."
        )

    stmt = statements[0]
    if not isinstance(stmt, (exp.Select, exp.With)):
        raise SQLValidationError(
            "Security violation: only read-only SELECT (or WITH...SELECT) queries "
            f"are permitted, got {type(stmt).__name__}."
        )

    for node in stmt.walk():
        if isinstance(node, REJECTED_NODE_TYPES):
            raise SQLValidationError(
                f"Security violation: forbidden construct '{type(node).__name__}' "
                "is not permitted in read-only queries."
            )
    if stmt.args.get("into") is not None:
        raise SQLValidationError(
            "Security violation: SELECT...INTO (table creation) is forbidden."
        )

    cte_names = {
        c.alias_or_name.lower()
        for c in stmt.find_all(exp.CTE)
        if c.alias_or_name
    }

    allowed_lower = {t.lower() for t in allowed_tables} if allowed_tables is not None else None
    catalog_cols = {c.lower() for c in allowed_columns} if allowed_columns is not None else None
    real_tables: Set[str] = set()
    alias_to_table: Dict[str, str] = {}
    for table in stmt.find_all(exp.Table):
        inner = table.this
        if not isinstance(inner, exp.Identifier):
            raise SQLValidationError(
                "Security violation: table-valued functions (read_csv*, read_parquet*, "
                "glob, pragma_*, …) are forbidden; only registered tables may be queried."
            )
        tname = table.name
        # Shape checks always apply, even without a catalog allowlist.
        if not tname or not IDENT_RE.match(tname):
            raise SQLValidationError(
                f"Security violation: invalid table reference '{tname}'."
            )
        if table.args.get("db") or table.args.get("catalog"):
            raise SQLValidationError(
                f"Security violation: qualified table reference '{tname}' is forbidden; "
                "use unqualified registered table names."
            )
        tname_lower = tname.lower()
        if allowed_lower is not None and tname_lower not in allowed_lower and tname_lower not in cte_names:
            raise SQLValidationError(
                f"Security violation: unknown table '{tname}'. Query may only reference "
                "registered catalog tables."
            )
        real_tables.add(tname_lower)
        alias = table.args.get("alias")
        alias_name = alias.alias_or_name if alias is not None else ""
        if alias_name:
            alias_to_table[alias_name.lower()] = tname_lower

    derived_aliases = {
        (s.args.get("alias").alias_or_name.lower())
        for s in stmt.find_all(exp.Subquery)
        if s.args.get("alias") is not None and s.args.get("alias").alias_or_name
    }
    select_aliases = {
        a.alias_or_name.lower()
        for a in stmt.find_all(exp.Alias)
        if a.alias_or_name
    }

    # Column shape (identifier charset) always applies; membership needs a catalog.
    for col in stmt.find_all(exp.Column):
        if isinstance(col.this, exp.Star):
            continue  # `*` / `t.*` — tables already allowlisted
        cname = col.name
        if catalog_cols is not None and cname and cname.lower() in catalog_cols:
            pass
        elif not cname or not IDENT_RE.match(cname):
            raise SQLValidationError(
                f"Security violation: invalid column reference '{cname}'."
            )
        if catalog_cols is None:
            continue
        cname_lower = cname.lower()
        qualifier = (col.table or "").lower()
        if qualifier:
            if qualifier in cte_names or qualifier in derived_aliases:
                continue  # CTE / derived-table outputs are query-defined
            if qualifier in alias_to_table or qualifier in real_tables:
                if cname_lower not in catalog_cols:
                    raise SQLValidationError(
                        f"Security violation: unknown column '{cname}'."
                    )
                continue
            if cname_lower not in catalog_cols and cname_lower not in select_aliases:
                raise SQLValidationError(
                    f"Security violation: unknown column '{cname}'."
                )
        elif cname_lower not in catalog_cols and cname_lower not in select_aliases:
            raise SQLValidationError(
                f"Security violation: unknown column '{cname}'."
            )

    for fn in stmt.find_all(exp.Func):
        # Structural null-handling wrappers carry no function call.
        if type(fn).__name__ in ("IgnoreNulls", "RespectNulls"):
            continue
        # Boolean connectors (AND/OR) subclass Func but call nothing.
        if isinstance(fn, exp.Connector):
            continue
        fname = _func_name(fn)
        if fname not in ALLOWED_FUNCTIONS:
            raise SQLValidationError(
                f"Security violation: function '{fname}' is not in the read-only "
                "allowlist (table functions, pragmas, and introspection helpers "
                "are forbidden)."
            )


def _catalog_from_connection(
    conn: duckdb.DuckDBPyConnection,
) -> Tuple[List[str], Set[str]]:
    """Introspect registered tables/views and their columns (internal query)."""
    tables = [
        r[0]
        for r in conn.execute("SELECT table_name FROM information_schema.tables").fetchall()
    ]
    cols: Set[str] = set()
    for t in tables:
        try:
            desc = conn.execute(f"SELECT * FROM {quote_ident(t)} LIMIT 0").description
            cols.update((d[0] or "").lower() for d in (desc or []))
        except Exception:
            continue
    return tables, cols


def harden_duckdb_connection(
    conn: duckdb.DuckDBPyConnection,
    memory_limit: str = DUCKDB_MEMORY_LIMIT,
    threads: int = DUCKDB_THREADS,
) -> duckdb.DuckDBPyConnection:
    """Apply engine-level hardening. Must run before any untrusted SQL.

    Disables all external file access, then locks configuration so it cannot
    be re-enabled. register()/normal SELECTs keep working (verified).
    """
    conn.execute(f"SET memory_limit='{memory_limit}'")
    conn.execute(f"SET threads TO {int(threads)}")
    conn.execute("SET enable_external_access=false")
    conn.execute("SET lock_configuration=true")
    return conn


def _execute_with_timeout(
    conn: duckdb.DuckDBPyConnection, query: str, timeout_seconds: int
) -> pd.DataFrame:
    """Run a query with a client-side timeout (DuckDB has no timeout knob).

    On timeout the running query is cancelled via conn.interrupt().
    """
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="duckdb-timeout"
    ) as pool:
        future = pool.submit(lambda: conn.execute(query).df())
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            try:
                conn.interrupt()
            except Exception:
                pass
            raise RuntimeError(
                f"DuckDB SQL error: query exceeded {timeout_seconds}s timeout and was cancelled."
            ) from exc


def execute_sql(
    query: str,
    conn: duckdb.DuckDBPyConnection,
    max_rows: int = 500,
    allowed_tables: Optional[Collection[str]] = None,
    allowed_columns: Optional[Collection[str]] = None,
    timeout_seconds: int = DEFAULT_QUERY_TIMEOUT_S,
) -> Dict[str, Any]:
    """Executes an allowlist-validated read-only SQL query and returns records.

    When allowed_tables/columns are omitted they are introspected from the
    connection, so legacy callers keep working with full enforcement.
    """
    import time

    if allowed_tables is None or allowed_columns is None:
        catalog_tables, catalog_cols = _catalog_from_connection(conn)
        if allowed_tables is None:
            allowed_tables = catalog_tables
        if allowed_columns is None:
            allowed_columns = catalog_cols
    validate_sql_safety(query, allowed_tables=allowed_tables, allowed_columns=allowed_columns)

    try:
        max_rows = min(int(max_rows or 500), MAX_ROWS_HARD_CAP)
    except (TypeError, ValueError):
        max_rows = 500
    start_time = time.perf_counter()

    logger.info(f"Running SQL query: {query}")
    try:
        df_result: pd.DataFrame = _execute_with_timeout(conn, query, timeout_seconds)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        total_rows = len(df_result)
        truncated = False
        if total_rows > max_rows:
            df_result = df_result.head(max_rows)
            truncated = True

        # Convert timestamps and NaNs to serializable objects
        records: List[Dict[str, Any]] = df_result.to_dict(orient="records")

        # Clean NaN/inf for strict JSON serialization
        for row in records:
            for k, v in row.items():
                if pd.isna(v):
                    row[k] = None
                elif hasattr(v, "isoformat"):
                    row[k] = v.isoformat()

        columns = list(df_result.columns)

        summary = f"SQL executed in {elapsed_ms:.2f}ms. Returned {total_rows} row(s)."
        if truncated:
            summary += f" Truncated to first {max_rows} rows for display."

        return {
            "query": query,
            "columns": columns,
            "records": records,
            "row_count": total_rows,
            "truncated": truncated,
            "execution_time_ms": elapsed_ms,
            "summary": summary,
        }
    except Exception as exc:
        logger.error(f"SQL execution failed: {str(exc)}")
        if isinstance(exc, RuntimeError) and "timeout" in str(exc):
            raise
        raise RuntimeError(f"DuckDB SQL error: {str(exc)}") from exc


@register_tool(
    name="execute_sql_query",
    description="Executes a safe, read-only SQL query against the loaded datasets using DuckDB and returns the tabulated result.",
    parameter_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The read-only SQL query to execute (e.g. SELECT region, SUM(revenue) FROM sales GROUP BY region ORDER BY SUM(revenue) DESC).",
            },
            "max_rows": {
                "type": "integer",
                "description": "Maximum number of rows to return (default: 500).",
                "default": 500,
            }
        },
        "required": ["query"],
    }
)
def tool_execute_sql_query(
    query: str,
    conn: duckdb.DuckDBPyConnection,
    max_rows: int = 500,
) -> Dict[str, Any]:
    """Tool wrapper for safe SQL query execution."""
    return execute_sql(query=query, conn=conn, max_rows=max_rows)
