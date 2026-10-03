"""
Data Profiling and Data Quality validation tool.
Calculates statistical summaries, null percentages, distinct counts, and quality metrics.
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np
from src.models.schemas import ColumnProfile, DatasetMetadata, DataQualityReport
from src.tools.registry import register_tool
from src.utils.logging import get_logger

logger = get_logger(__name__)


def profile_dataframe(df: pd.DataFrame, table_name: str) -> DatasetMetadata:
    """
    Deterministically computes a comprehensive statistical profile of a pandas DataFrame.
    """
    row_count, col_count = df.shape
    memory_bytes = int(df.memory_usage(deep=True).sum())
    column_profiles: List[ColumnProfile] = []

    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = round((null_count / row_count * 100.0) if row_count > 0 else 0.0, 2)
        distinct_count = int(series.nunique(dropna=True))

        # Sample values (up to 5 distinct non-null values)
        non_null_samples = series.dropna().unique()[:5].tolist()
        # Convert non-serializable objects (like timestamps or numpy types) to python primitives, skipping inf/nan
        sample_values = []
        for val in non_null_samples:
            if pd.isna(val):
                continue
            if hasattr(val, "isoformat"):
                sample_values.append(val.isoformat())
            elif hasattr(val, "item"):
                item_val = val.item()
                if isinstance(item_val, float) and (np.isnan(item_val) or np.isinf(item_val)):
                    continue
                sample_values.append(item_val)
            elif isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                continue
            else:
                sample_values.append(val)

        min_val = None
        max_val = None
        mean_val = None
        std_val = None

        if pd.api.types.is_numeric_dtype(series):
            clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
            if not clean_series.empty:
                min_val = float(clean_series.min())
                max_val = float(clean_series.max())
                mean_val = round(float(clean_series.mean()), 4)
                std_val = round(float(clean_series.std()), 4) if len(clean_series) > 1 else 0.0

        column_profiles.append(
            ColumnProfile(
                name=str(col),
                dtype=str(series.dtype),
                null_count=null_count,
                null_percentage=null_pct,
                distinct_count=distinct_count,
                sample_values=sample_values,
                min_value=min_val,
                max_value=max_val,
                mean=mean_val,
                std=std_val,
            )
        )

    return DatasetMetadata(
        table_name=table_name,
        original_filename=table_name,
        row_count=row_count,
        column_count=col_count,
        memory_bytes=memory_bytes,
        columns=column_profiles,
    )


def check_data_quality(df: pd.DataFrame, table_name: str) -> DataQualityReport:
    """
    Evaluates dataset quality: completeness, duplicates, missing cells, and structural warnings.
    """
    row_count, col_count = df.shape
    total_cells = row_count * col_count
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    completeness = round(
        ((total_cells - missing_cells) / total_cells * 100.0) if total_cells > 0 else 100.0,
        2
    )

    issues: List[str] = []
    if duplicate_rows > 0:
        issues.append(f"Found {duplicate_rows} duplicate row(s) ({(duplicate_rows / row_count * 100):.1f}% of data).")

    if missing_cells > 0:
        issues.append(f"Total missing values: {missing_cells} across {col_count} columns.")

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            null_pct = (null_count / row_count) * 100.0
            if null_pct > 30.0:
                issues.append(f"Column '{col}' has high missingness ({null_pct:.1f}% missing).")

        # Check for constant/zero-variance column
        if df[col].nunique(dropna=False) == 1:
            issues.append(f"Column '{col}' is constant (has only a single unique value).")

    if not issues:
        issues.append("No critical data quality issues identified. Dataset is well-formed.")

    profile = profile_dataframe(df, table_name)

    return DataQualityReport(
        table_name=table_name,
        row_count=row_count,
        completeness_score=completeness,
        duplicate_rows=duplicate_rows,
        missing_cells=missing_cells,
        quality_issues=issues,
        column_profiles=profile.columns,
    )


@register_tool(
    name="profile_dataset",
    description="Generates statistical summary, column schemas, null rates, and sample values for a table.",
    parameter_schema={
        "type": "object",
        "properties": {
            "table_name": {
                "type": "string",
                "description": "Name of the table to profile.",
            }
        },
        "required": ["table_name"],
    }
)
def tool_profile_dataset(df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
    """Tool wrapper for profile_dataframe."""
    metadata = profile_dataframe(df, table_name)
    return {
        "metadata": metadata.model_dump(),
        "summary": f"Profiled '{table_name}': {metadata.row_count} rows, {metadata.column_count} columns.",
    }


@register_tool(
    name="check_data_quality",
    description="Performs automated data quality checks including completeness score, duplicates, and column warnings.",
    parameter_schema={
        "type": "object",
        "properties": {
            "table_name": {
                "type": "string",
                "description": "Name of the table to run quality checks on.",
            }
        },
        "required": ["table_name"],
    }
)
def tool_check_data_quality(df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
    """Tool wrapper for check_data_quality."""
    report = check_data_quality(df, table_name)
    return {
        "quality_report": report.model_dump(),
        "summary": f"Quality check for '{table_name}': Completeness {report.completeness_score}%, {report.duplicate_rows} duplicates, {len(report.quality_issues)} findings.",
    }
