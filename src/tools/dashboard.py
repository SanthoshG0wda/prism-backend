"""
Generic executive dashboard builder.

Replaces hardcoded region/revenue/product/profit assumptions with
schema-aware inference so ANY uploaded CSV gets useful KPIs + charts.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from src.tools.profiling import check_data_quality
from src.utils.json_safe import json_safe


def _infer_column_roles(df: pd.DataFrame) -> Dict[str, Any]:
    """Infer categorical, numeric, and datetime roles from dtypes + cardinality."""
    import re
    import warnings
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    datetime_cols: List[str] = []
    date_name_hints = ("date", "time", "month", "year", "day", "created", "timestamp")
    date_value_re = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}")

    def _looks_datetime(col: str) -> bool:
        if any(h in col.lower() for h in date_name_hints):
            return True
        try:
            sample = df[col].dropna().astype(str).head(20)
            hits = sum(1 for v in sample if date_value_re.search(v))
            return hits >= max(3, len(sample) // 2)
        except Exception:
            return False

    for col in df.columns:
        if col in numeric_cols or col in object_cols:
            continue
        if not _looks_datetime(col):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() / max(len(df), 1) > 0.8:
                datetime_cols.append(col)
        except Exception:
            continue
    # Also detect datetime-like object columns
    for col in list(object_cols):
        if not _looks_datetime(col):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() / max(len(df), 1) > 0.8 and df[col].nunique() > 5:
                datetime_cols.append(col)
        except Exception:
            pass

    # Prefer low-cardinality categoricals for grouping (2..20 distinct values)
    cat_candidates = sorted(
        [c for c in object_cols if c not in datetime_cols],
        key=lambda c: df[c].nunique(dropna=True),
    )
    group_col: Optional[str] = None
    for c in cat_candidates:
        n = df[c].nunique(dropna=True)
        if 1 < n <= 20:
            group_col = c
            break
    if group_col is None and cat_candidates:
        group_col = cat_candidates[0]

    # Primary metric: largest-sum numeric column (revenue-like names preferred)
    metric_col: Optional[str] = None
    if numeric_cols:
        preferred = [c for c in numeric_cols if c.lower() in
                     ("revenue", "sales", "profit", "amount", "total", "price", "value", "units_sold", "quantity")]
        metric_col = preferred[0] if preferred else numeric_cols[0]

    secondary_metric: Optional[str] = None
    if len(numeric_cols) > 1:
        for c in numeric_cols:
            if c != metric_col:
                secondary_metric = c
                break

    date_col: Optional[str] = None
    if datetime_cols:
        date_col = datetime_cols[0]
    else:
        for c in df.columns:
            if "date" in c.lower() or "time" in c.lower() or "month" in c.lower():
                try:
                    if pd.to_datetime(df[c], errors="coerce").notna().sum() > 0:
                        date_col = c
                        break
                except Exception:
                    continue

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": object_cols,
        "datetime_cols": datetime_cols,
        "group_col": group_col,
        "metric_col": metric_col,
        "secondary_metric": secondary_metric,
        "date_col": date_col,
    }


def build_dashboard_data(df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
    """Build deterministic KPI + chart payload for any tabular DataFrame."""
    quality_report = check_data_quality(df, table_name)
    roles = _infer_column_roles(df)
    numeric_cols: List[str] = roles["numeric_cols"]
    group_col = roles["group_col"]
    metric_col = roles["metric_col"]
    secondary_metric = roles["secondary_metric"]

    # Generic metric totals (first two numeric cols) — no hardcoded names
    metric_totals: Dict[str, float] = {}
    for c in numeric_cols[:4]:
        try:
            metric_totals[c] = float(df[c].sum())
        except Exception:
            metric_totals[c] = 0.0

    primary_total = metric_totals.get(metric_col or "", 0.0)
    secondary_total = metric_totals.get(secondary_metric or "", 0.0)

    charts: List[Dict[str, Any]] = []

    # Chart 1: group-by bar (top 10 groups)
    if group_col and metric_col and group_col in df.columns and metric_col in df.columns:
        try:
            grouped = df.groupby(group_col, dropna=False)[metric_col].sum().reset_index()
            grouped = grouped.sort_values(by=metric_col, ascending=False).head(10)
            charts.append({
                "title": f"{metric_col} by {group_col}",
                "type": "bar",
                "data": grouped.to_dict(orient="records"),
                "x_key": group_col,
                "y_key": metric_col,
            })
        except Exception:
            pass

    # Chart 2: share pie (only when 2..8 groups to stay readable) else histogram/scatter fallback
    if group_col and metric_col and len(charts) == 1:
        try:
            n_groups = df[group_col].nunique(dropna=True)
            if 2 <= n_groups <= 8:
                grouped_p = df.groupby(group_col, dropna=False)[metric_col].sum().reset_index()
                charts.append({
                    "title": f"{metric_col} share by {group_col}",
                    "type": "pie",
                    "data": grouped_p.to_dict(orient="records"),
                    "x_key": group_col,
                    "y_key": metric_col,
                })
            elif metric_col in df.columns:
                hist = df[[metric_col]].dropna().head(500).to_dict(orient="records")
                charts.append({
                    "title": f"Distribution of {metric_col}",
                    "type": "histogram",
                    "data": hist,
                    "x_key": metric_col,
                    "y_key": metric_col,
                })
        except Exception:
            pass

    # Fallback when no categorical grouping exists: numeric distribution
    if not charts and metric_col:
        try:
            hist = df[[metric_col]].dropna().head(500).to_dict(orient="records")
            charts.append({
                "title": f"Distribution of {metric_col}",
                "type": "histogram",
                "data": hist,
                "x_key": metric_col,
                "y_key": metric_col,
            })
        except Exception:
            pass

    return json_safe({
        "table_name": table_name,
        "kpis": {
            "total_rows": len(df),
            "column_count": len(df.columns),
            "completeness_score": quality_report.completeness_score,
            "duplicate_rows": quality_report.duplicate_rows,
            "numeric_columns_count": len(numeric_cols),
            "primary_metric": metric_col,
            "primary_metric_total": primary_total,
            "secondary_metric": secondary_metric,
            "secondary_metric_total": secondary_total,
            "metric_totals": metric_totals,
            # Back-compat keys for existing React dashboard view
            "total_revenue": primary_total,
            "total_profit": secondary_total if (secondary_metric or "").lower() == "profit" else 0.0,
        },
        "column_roles": roles,
        "quality_report": quality_report.model_dump(),
        "charts": charts,
    })
