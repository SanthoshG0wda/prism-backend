"""
Tools package exports.
Imports all individual tool modules to ensure registration in global_registry.
"""

from .registry import ToolRegistry, global_registry, register_tool
from .profiling import profile_dataframe, check_data_quality, tool_profile_dataset, tool_check_data_quality
from .sql import execute_sql, validate_sql_safety, tool_execute_sql_query
from .analysis import (
    top_k_analysis,
    aggregate_metric,
    calculate_correlation,
    time_series_trend,
    tool_top_k_analysis,
    tool_aggregate_metric,
    tool_calculate_correlation,
    tool_time_series_trend,
)
from .charts import generate_chart, tool_generate_chart
from .anomalies import (
    detect_anomalies_iqr,
    detect_anomalies_zscore,
    tool_detect_anomalies,
)
from .forecasting import forecast_metric, tool_forecast_metric
from .export import generate_executive_html_report

__all__ = [
    "ToolRegistry",
    "global_registry",
    "register_tool",
    "profile_dataframe",
    "check_data_quality",
    "tool_profile_dataset",
    "tool_check_data_quality",
    "execute_sql",
    "validate_sql_safety",
    "tool_execute_sql_query",
    "top_k_analysis",
    "aggregate_metric",
    "calculate_correlation",
    "time_series_trend",
    "tool_top_k_analysis",
    "tool_aggregate_metric",
    "tool_calculate_correlation",
    "tool_time_series_trend",
    "generate_chart",
    "tool_generate_chart",
    "detect_anomalies_iqr",
    "detect_anomalies_zscore",
    "tool_detect_anomalies",
    "forecast_metric",
    "tool_forecast_metric",
    "generate_executive_html_report",
]
