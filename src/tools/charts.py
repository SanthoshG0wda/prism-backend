"""
Visualization tool using Plotly.
Generates interactive bar, line, pie, scatter, histogram, and box charts with modern styling.
"""

from typing import Any, Dict, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.tools.registry import register_tool
from src.utils.logging import get_logger

logger = get_logger(__name__)

PLOTLY_TEMPLATE = "plotly_dark"


def generate_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: str,
    y: Optional[str] = None,
    color: Optional[str] = None,
    title: Optional[str] = None,
    orientation: str = "v",
) -> Dict[str, Any]:
    """
    Deterministically builds a Plotly figure dictionary based on user parameters.
    """
    chart_type_clean = chart_type.lower().strip()
    chart_title = title or f"{chart_type.capitalize()} Chart: {y or ''} by {x}".strip()

    def _find_col(col_name: Optional[str]) -> Optional[str]:
        if not col_name:
            return None
        if col_name in df.columns:
            return col_name
        col_clean = str(col_name).strip().lower()
        for c in df.columns:
            if str(c).strip().lower() == col_clean:
                return c
        return col_name

    x = _find_col(x) or x
    if y:
        y = _find_col(y)
    if color:
        color = _find_col(color)

    if x not in df.columns:
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        x = cat_cols[0] if cat_cols else df.columns[0]
    if y and y not in df.columns:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != x]
        y = num_cols[0] if num_cols else None

    # Auto-swap x and y if x is numeric and y is categorical for bar or pie charts
    if y and chart_type_clean in ("bar", "pie"):
        if pd.api.types.is_numeric_dtype(df[x]) and not pd.api.types.is_numeric_dtype(df[y]):
            x, y = y, x

    if color and color not in df.columns:
        color = None

    aggregated = False
    # Aggregate raw event-level data before plotting so bar/pie charts stay readable.
    # E.g. 1000 raw sales rows grouped by region instead of 1000 individual bars.
    if (
        y
        and chart_type_clean in ("bar", "pie")
        and len(df) > 50
        and pd.api.types.is_numeric_dtype(df[y])
        and df[x].nunique(dropna=True) <= 30
    ):
        try:
            df = df.groupby(x, dropna=False)[y].sum().reset_index()
            df = df.sort_values(by=y, ascending=False)
            if chart_type_clean == "pie" and len(df) > 8:
                df = df.head(8)
            elif len(df) > 15:
                df = df.head(15)
            aggregated = True
        except Exception:
            pass

    fig: go.Figure

    if chart_type_clean == "bar":
        fig = px.bar(
            df,
            x=x,
            y=y,
            color=color,
            title=chart_title,
            template=PLOTLY_TEMPLATE,
            orientation=orientation,
        )
        pandas_code = f"px.bar(df, x='{x}', y='{y}', color={repr(color)}, title={repr(chart_title)})"

    elif chart_type_clean == "line":
        # Sort by x if it's dates
        sorted_df = df.copy()
        if pd.api.types.is_datetime64_any_dtype(sorted_df[x]) or "date" in x.lower():
            sorted_df[x] = pd.to_datetime(sorted_df[x], errors="coerce")
            sorted_df = sorted_df.sort_values(by=x)

        fig = px.line(
            sorted_df,
            x=x,
            y=y,
            color=color,
            markers=True,
            title=chart_title,
            template=PLOTLY_TEMPLATE,
        )
        pandas_code = f"px.line(df, x='{x}', y='{y}', color={repr(color)}, markers=True, title={repr(chart_title)})"

    elif chart_type_clean == "pie":
        fig = px.pie(
            df,
            names=x,
            values=y,
            title=chart_title,
            template=PLOTLY_TEMPLATE,
            hole=0.35,  # Donut style for modern appearance
        )
        pandas_code = f"px.pie(df, names='{x}', values='{y}', hole=0.35, title={repr(chart_title)})"

    elif chart_type_clean == "scatter":
        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=color,
            title=chart_title,
            template=PLOTLY_TEMPLATE,
        )
        pandas_code = f"px.scatter(df, x='{x}', y='{y}', color={repr(color)}, title={repr(chart_title)})"

    elif chart_type_clean == "histogram":
        fig = px.histogram(
            df,
            x=x,
            color=color,
            title=chart_title,
            template=PLOTLY_TEMPLATE,
        )
        pandas_code = f"px.histogram(df, x='{x}', color={repr(color)}, title={repr(chart_title)})"

    elif chart_type_clean == "box":
        fig = px.box(
            df,
            x=x,
            y=y,
            color=color,
            title=chart_title,
            template=PLOTLY_TEMPLATE,
        )
        pandas_code = f"px.box(df, x='{x}', y='{y}', color={repr(color)}, title={repr(chart_title)})"

    else:
        raise ValueError(
            f"Unsupported chart type '{chart_type}'. Choose from: bar, line, pie, scatter, histogram, box."
        )

    # Apply modern layout tweaks
    fig.update_layout(
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=13),
        title_font=dict(size=16),
        hoverlabel=dict(font_size=12),
    )

    return {
        "chart_type": chart_type_clean,
        "title": chart_title,
        "plotly_spec": fig.to_dict(),
        "pandas_code": pandas_code,
        "summary": f"Created {chart_type_clean} chart with x='{x}' and y='{y}'."
        + (" Pre-aggregated raw rows with SUM grouped by x." if aggregated else ""),
    }


@register_tool(
    name="generate_chart",
    description="Generates an interactive Plotly chart (bar, line, pie, scatter, histogram, box) from dataframe columns.",
    parameter_schema={
        "type": "object",
        "properties": {
            "chart_type": {
                "type": "string",
                "enum": ["bar", "line", "pie", "scatter", "histogram", "box"],
                "description": "The type of chart to produce.",
            },
            "x": {
                "type": "string",
                "description": "Column name for the X-axis (or category names for pie chart).",
            },
            "y": {
                "type": "string",
                "description": "Column name for the Y-axis (or quantitative values for pie chart).",
            },
            "color": {
                "type": "string",
                "description": "Optional column name to group or color points by.",
            },
            "title": {
                "type": "string",
                "description": "Custom title for the chart.",
            }
        },
        "required": ["chart_type", "x"],
    }
)
def tool_generate_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: str,
    y: Optional[str] = None,
    color: Optional[str] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    return generate_chart(df=df, chart_type=chart_type, x=x, y=y, color=color, title=title)
