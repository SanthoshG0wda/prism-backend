"""
Deterministic Anomaly Detection module.
Implements statistical outlier detection via IQR (Interquartile Range) and Z-Score,
providing transparent mathematical explanations for every flagged data point.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.models.schemas import AnomalyDetectionResult, AnomalyItem
from src.tools.registry import register_tool
from src.utils.logging import get_logger

logger = get_logger(__name__)


def detect_anomalies_iqr(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5,
) -> AnomalyDetectionResult:
    """
    Identifies statistical anomalies using Tukey's Interquartile Range (IQR) method.
    Flag condition: value < (Q1 - multiplier * IQR) OR value > (Q3 + multiplier * IQR).
    """
    if column not in df.columns:
        matched = [c for c in df.columns if c.strip().lower() == column.strip().lower()]
        if matched:
            column = matched[0]
        else:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                column = num_cols[0]
            else:
                raise ValueError(f"Column '{column}' does not exist in dataset.")

    series = df[column].dropna()
    if not pd.api.types.is_numeric_dtype(series):
        try:
            converted = pd.to_numeric(series.astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').dropna()
            if len(converted) >= max(3, int(len(series) * 0.5)):
                series = converted
            else:
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if num_cols:
                    column = num_cols[0]
                    series = df[column].dropna()
                else:
                    raise ValueError(f"Column '{column}' must be numeric for anomaly detection.")
        except Exception:
            raise ValueError(f"Column '{column}' must be numeric for anomaly detection.")

    series = series.replace([np.inf, -np.inf], np.nan).dropna()

    if len(series) < 4:
        return AnomalyDetectionResult(
            table_name="active_dataset",
            column_analyzed=column,
            method="iqr",
            threshold=multiplier,
            anomalies_found=0,
            anomalies=[],
            summary=f"Insufficient sample size ({len(series)} values) to compute IQR anomalies.",
        )

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1

    lower_bound = q1 - (multiplier * iqr)
    upper_bound = q3 + (multiplier * iqr)
    median_val = float(series.median())

    anomalies: List[AnomalyItem] = []

    for idx, val in series.items():
        val_float = float(val)
        if val_float < lower_bound:
            dist = round(lower_bound - val_float, 2)
            anomalies.append(
                AnomalyItem(
                    column=column,
                    row_index=int(idx),
                    value=val_float,
                    method="iqr",
                    score=dist,
                    explanation=(
                        f"Value {val_float} is {dist} below the lower fence ({lower_bound:.2f}). "
                        f"[Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, Median={median_val:.2f}]"
                    ),
                )
            )
        elif val_float > upper_bound:
            dist = round(val_float - upper_bound, 2)
            anomalies.append(
                AnomalyItem(
                    column=column,
                    row_index=int(idx),
                    value=val_float,
                    method="iqr",
                    score=dist,
                    explanation=(
                        f"Value {val_float} is {dist} above the upper fence ({upper_bound:.2f}). "
                        f"[Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, Median={median_val:.2f}]"
                    ),
                )
            )

    summary = (
        f"IQR detection on '{column}': Found {len(anomalies)} outlier(s) "
        f"outside bounds [{lower_bound:.2f}, {upper_bound:.2f}]."
    )

    return AnomalyDetectionResult(
        table_name="active_dataset",
        column_analyzed=column,
        method="iqr",
        threshold=multiplier,
        anomalies_found=len(anomalies),
        anomalies=anomalies,
        summary=summary,
    )


def detect_anomalies_zscore(
    df: pd.DataFrame,
    column: str,
    threshold: float = 3.0,
) -> AnomalyDetectionResult:
    """
    Identifies statistical outliers using Standard Z-Score.
    Flag condition: |(value - mean) / std| > threshold.
    """
    if column not in df.columns:
        matched = [c for c in df.columns if c.strip().lower() == column.strip().lower()]
        if matched:
            column = matched[0]
        else:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                column = num_cols[0]
            else:
                raise ValueError(f"Column '{column}' does not exist in dataset.")

    series = df[column].dropna()
    if not pd.api.types.is_numeric_dtype(series):
        try:
            converted = pd.to_numeric(series.astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').dropna()
            if len(converted) >= max(3, int(len(series) * 0.5)):
                series = converted
            else:
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if num_cols:
                    column = num_cols[0]
                    series = df[column].dropna()
                else:
                    raise ValueError(f"Column '{column}' must be numeric for Z-Score anomaly detection.")
        except Exception:
            raise ValueError(f"Column '{column}' must be numeric for Z-Score anomaly detection.")

    series = series.replace([np.inf, -np.inf], np.nan).dropna()

    if len(series) < 3:
        return AnomalyDetectionResult(
            table_name="active_dataset",
            column_analyzed=column,
            method="z_score",
            threshold=threshold,
            anomalies_found=0,
            anomalies=[],
            summary=f"Insufficient sample size ({len(series)} values) to compute Z-Scores.",
        )

    mean_val = float(series.mean())
    std_val = float(series.std())

    if std_val == 0.0 or np.isnan(std_val):
        return AnomalyDetectionResult(
            table_name="active_dataset",
            column_analyzed=column,
            method="z_score",
            threshold=threshold,
            anomalies_found=0,
            anomalies=[],
            summary=f"Column '{column}' has zero variance (all values are identical). No anomalies.",
        )

    anomalies: List[AnomalyItem] = []

    for idx, val in series.items():
        val_float = float(val)
        z = (val_float - mean_val) / std_val
        if abs(z) > threshold:
            anomalies.append(
                AnomalyItem(
                    column=column,
                    row_index=int(idx),
                    value=val_float,
                    method="z_score",
                    score=round(float(z), 3),
                    explanation=(
                        f"Value {val_float} deviates by {abs(z):.2f} standard deviations from mean "
                        f"(Mean={mean_val:.2f}, Std={std_val:.2f}, Threshold={threshold:.1f}\u03c3)."
                    ),
                )
            )

    summary = (
        f"Z-Score detection on '{column}': Found {len(anomalies)} outlier(s) "
        f"exceeding |Z| > {threshold:.1f} (Mean={mean_val:.2f}, Std={std_val:.2f})."
    )

    return AnomalyDetectionResult(
        table_name="active_dataset",
        column_analyzed=column,
        method="z_score",
        threshold=threshold,
        anomalies_found=len(anomalies),
        anomalies=anomalies,
        summary=summary,
    )


@register_tool(
    name="detect_anomalies",
    description="Detects statistical anomalies and outliers in a numeric column using IQR or Z-Score with detailed explanations.",
    parameter_schema={
        "type": "object",
        "properties": {
            "column": {
                "type": "string",
                "description": "The numeric column to inspect for outliers.",
            },
            "method": {
                "type": "string",
                "enum": ["iqr", "z_score"],
                "default": "iqr",
                "description": "Detection algorithm: 'iqr' (robust to extreme values) or 'z_score'.",
            },
            "threshold": {
                "type": "number",
                "default": 1.5,
                "description": "IQR multiplier (default: 1.5) or Z-score threshold (default: 3.0).",
            },
        },
        "required": ["column"],
    }
)
def tool_detect_anomalies(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Tool wrapper for anomaly detection."""
    method_clean = method.lower().strip()
    if method_clean == "z_score":
        thresh = threshold if threshold is not None else 3.0
        res = detect_anomalies_zscore(df, column, threshold=thresh)
    else:
        thresh = threshold if threshold is not None else 1.5
        res = detect_anomalies_iqr(df, column, multiplier=thresh)

    return {
        "result": res.model_dump(),
        "summary": res.summary,
        "anomalies": [a.model_dump() for a in res.anomalies],
    }
