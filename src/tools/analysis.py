"""
Deterministic analytical calculations module using Pandas.
Provides functions for aggregations, top-k rankings, correlations, and trend analysis.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from src.tools.registry import register_tool
from src.utils.logging import get_logger

logger = get_logger(__name__)


def top_k_analysis(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    k: int = 5,
    ascending: bool = False,
    agg_func: str = "sum",
) -> Dict[str, Any]:
    """
    Computes top-k or bottom-k groups based on an aggregated metric.
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataset columns: {list(df.columns)}")
    if metric_col not in df.columns:
        raise ValueError(f"Metric column '{metric_col}' not found in dataset columns: {list(df.columns)}")

    if not pd.api.types.is_numeric_dtype(df[metric_col]):
        if pd.api.types.is_numeric_dtype(df[group_col]):
            group_col, metric_col = metric_col, group_col
        else:
            raise ValueError(f"Metric column '{metric_col}' must be numeric.")

    grouped = df.groupby(group_col)[metric_col].agg(agg_func).reset_index()
    sorted_df = grouped.sort_values(by=metric_col, ascending=ascending).head(k)

    records = sorted_df.to_dict(orient="records")
    direction = "ascending (lowest)" if ascending else "descending (highest)"
    summary = f"Top {k} by '{metric_col}' ({agg_func}) grouped by '{group_col}' in {direction} order."

    # Generate equivalent Pandas code for transparency
    pandas_code = (
        f"df.groupby('{group_col}')['{metric_col}'].{agg_func}()"
        f".reset_index().sort_values(by='{metric_col}', ascending={ascending}).head({k})"
    )

    return {
        "group_col": group_col,
        "metric_col": metric_col,
        "agg_func": agg_func,
        "k": k,
        "ascending": ascending,
        "records": records,
        "pandas_code": pandas_code,
        "summary": summary,
    }


def aggregate_metric(
    df: pd.DataFrame,
    group_by: List[str],
    metric_col: str,
    agg_func: str = "sum",
) -> Dict[str, Any]:
    """
    Groups by specified dimensions and applies an aggregation function (sum, mean, median, min, max, count).
    """
    for col in group_by:
        if col not in df.columns:
            raise ValueError(f"Group-by column '{col}' not found in dataset columns.")
    if metric_col not in df.columns and agg_func != "count":
        raise ValueError(f"Metric column '{metric_col}' not found in dataset columns.")

    allowed_funcs = ["sum", "mean", "median", "min", "max", "count", "std"]
    if agg_func.lower() not in allowed_funcs:
        raise ValueError(f"Unsupported aggregation function: '{agg_func}'. Allowed: {allowed_funcs}")

    grouped = df.groupby(group_by)[metric_col].agg(agg_func.lower()).reset_index()
    records = grouped.to_dict(orient="records")

    pandas_code = f"df.groupby({group_by})['{metric_col}'].{agg_func.lower()}().reset_index()"

    return {
        "group_by": group_by,
        "metric_col": metric_col,
        "agg_func": agg_func,
        "records": records,
        "pandas_code": pandas_code,
        "summary": f"Aggregated '{metric_col}' by {group_by} using {agg_func.upper()}.",
    }


def calculate_correlation(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes Pearson correlation matrix between numeric columns in the dataset.
    """
    if numeric_cols:
        missing = [c for c in numeric_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        selected_df = df[numeric_cols]
    else:
        selected_df = df.select_dtypes(include=["number"])

    if selected_df.empty or selected_df.shape[1] < 2:
        raise ValueError("Correlation requires at least two numeric columns.")

    corr_df = selected_df.corr().round(4)
    corr_dict = corr_df.to_dict()

    # Find highest non-diagonal correlation pairs
    pairs = []
    cols = list(corr_df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            val = float(corr_df.loc[c1, c2])
            if not pd.isna(val):
                pairs.append({"col1": c1, "col2": c2, "correlation": val})

    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    pandas_code = f"df[{list(selected_df.columns)}].corr().round(4)"

    return {
        "columns": cols,
        "matrix": corr_dict,
        "top_correlated_pairs": pairs[:5],
        "pandas_code": pandas_code,
        "summary": f"Calculated correlation matrix across {len(cols)} numeric columns.",
    }


def time_series_trend(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    freq: str = "ME",
    agg_func: str = "sum",
) -> Dict[str, Any]:
    """
    Resamples time-series data at specified frequency (e.g. 'ME' for month-end, 'W' for week, 'YE' for year)
    and aggregates metrics to identify trends.
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found.")
    if metric_col not in df.columns:
        raise ValueError(f"Metric column '{metric_col}' not found.")

    temp_df = df.copy()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
    temp_df = temp_df.dropna(subset=[date_col])

    resampled = (
        temp_df.set_index(date_col)
        .resample(freq)[metric_col]
        .agg(agg_func)
        .reset_index()
    )
    resampled[date_col] = resampled[date_col].dt.strftime("%Y-%m-%d")

    records = resampled.to_dict(orient="records")

    pandas_code = (
        f"df.assign({date_col}=pd.to_datetime(df['{date_col}']))"
        f".set_index('{date_col}').resample('{freq}')['{metric_col}'].{agg_func}().reset_index()"
    )

    return {
        "date_col": date_col,
        "metric_col": metric_col,
        "frequency": freq,
        "agg_func": agg_func,
        "records": records,
        "pandas_code": pandas_code,
        "summary": f"Time series trend for '{metric_col}' resampled by '{freq}' with {agg_func.upper()}.",
    }


@register_tool(
    name="top_k_analysis",
    description="Finds top-K or bottom-K ranked groups by aggregating a numeric metric (e.g. top 5 customers by revenue).",
    parameter_schema={
        "type": "object",
        "properties": {
            "group_col": {"type": "string", "description": "Column name to group by (e.g. 'region', 'product')."},
            "metric_col": {"type": "string", "description": "Numeric column to aggregate (e.g. 'revenue', 'profit')."},
            "k": {"type": "integer", "description": "Number of top results to return (default: 5).", "default": 5},
            "ascending": {"type": "boolean", "description": "If true, returns lowest values (bottom-k).", "default": False},
            "agg_func": {"type": "string", "description": "Aggregation function ('sum', 'mean', etc.).", "default": "sum"},
        },
        "required": ["group_col", "metric_col"],
    }
)
def tool_top_k_analysis(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    k: int = 5,
    ascending: bool = False,
    agg_func: str = "sum",
) -> Dict[str, Any]:
    return top_k_analysis(df=df, group_col=group_col, metric_col=metric_col, k=k, ascending=ascending, agg_func=agg_func)


@register_tool(
    name="aggregate_metric",
    description="Aggregates a numeric column grouped by one or more categorical dimensions.",
    parameter_schema={
        "type": "object",
        "properties": {
            "group_by": {"type": "array", "items": {"type": "string"}, "description": "List of columns to group by."},
            "metric_col": {"type": "string", "description": "Numeric column to aggregate."},
            "agg_func": {"type": "string", "description": "Aggregation type (sum, mean, median, min, max, count).", "default": "sum"},
        },
        "required": ["group_by", "metric_col"],
    }
)
def tool_aggregate_metric(
    df: pd.DataFrame,
    group_by: List[str],
    metric_col: str,
    agg_func: str = "sum",
) -> Dict[str, Any]:
    return aggregate_metric(df=df, group_by=group_by, metric_col=metric_col, agg_func=agg_func)


@register_tool(
    name="calculate_correlation",
    description="Calculates the correlation matrix between numerical features.",
    parameter_schema={
        "type": "object",
        "properties": {
            "numeric_cols": {"type": "array", "items": {"type": "string"}, "description": "Optional subset of numeric columns to correlate."}
        },
    }
)
def tool_calculate_correlation(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return calculate_correlation(df=df, numeric_cols=numeric_cols)


@register_tool(
    name="time_series_trend",
    description="Analyzes time series trend over intervals (monthly, weekly, daily) for a metric.",
    parameter_schema={
        "type": "object",
        "properties": {
            "date_col": {"type": "string", "description": "Date or timestamp column name."},
            "metric_col": {"type": "string", "description": "Numeric metric column to track."},
            "freq": {"type": "string", "description": "Frequency alias (e.g. 'ME' for month-end, 'W' for week, 'YE' for year).", "default": "ME"},
            "agg_func": {"type": "string", "description": "Aggregation function.", "default": "sum"},
        },
        "required": ["date_col", "metric_col"],
    }
)
def tool_time_series_trend(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    freq: str = "ME",
    agg_func: str = "sum",
) -> Dict[str, Any]:
    return time_series_trend(df=df, date_col=date_col, metric_col=metric_col, freq=freq, agg_func=agg_func)
