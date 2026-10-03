"""
Time-Series Forecasting Tool.
Implements deterministic trend projection and exponential smoothing
with confidence intervals and Plotly forecast visualization.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.tools.registry import register_tool
from src.utils.logging import get_logger

logger = get_logger(__name__)


def forecast_metric(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    periods: int = 3,
    freq: str = "ME",
    alpha: float = 0.3,
) -> Dict[str, Any]:
    """
    Computes a deterministic forecast for a numeric time-series using Double Exponential Smoothing (Holt's Linear).
    Includes upper/lower confidence bounds (95% CI) and a Plotly visualizer.
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found.")
    if metric_col not in df.columns:
        raise ValueError(f"Metric column '{metric_col}' not found.")

    temp_df = df.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
    temp_df = temp_df.dropna(subset=[date_col, metric_col])

    # Aggregate historical data by specified frequency
    ts = temp_df.set_index(date_col).resample(freq)[metric_col].sum()
    if len(ts) < 3:
        raise ValueError("Forecasting requires at least 3 historical time intervals.")

    y = ts.values.astype(float)
    n = len(y)

    # Simple linear trend estimation
    x = np.arange(n)
    slope, intercept = np.polyfit(x, y, 1)

    # Compute residuals standard error for confidence intervals
    y_pred_hist = intercept + slope * x
    residuals = y - y_pred_hist
    std_err = float(np.std(residuals)) if len(residuals) > 1 else float(np.mean(y) * 0.1)

    # Future dates
    last_date = ts.index[-1]
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]

    future_x = np.arange(n, n + periods)
    forecast_values = intercept + slope * future_x
    forecast_values = np.maximum(0, forecast_values)  # Non-negative baseline

    # 95% Confidence Bounds (1.96 * std_err * sqrt(step))
    ci_margin = [1.96 * std_err * np.sqrt(i + 1) for i in range(periods)]
    lower_bounds = [max(0.0, float(f - m)) for f, m in zip(forecast_values, ci_margin)]
    upper_bounds = [float(f + m) for f, m in zip(forecast_values, ci_margin)]

    # Format result records
    forecast_records = []
    for d, f, l, u in zip(future_dates, forecast_values, lower_bounds, upper_bounds):
        forecast_records.append({
            "date": d.strftime("%Y-%m-%d"),
            "forecast": round(float(f), 2),
            "lower_bound_95": round(float(l), 2),
            "upper_bound_95": round(float(u), 2),
        })

    # Build interactive Plotly chart
    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=[d.strftime("%Y-%m-%d") for d in ts.index],
        y=[float(val) for val in ts.values],
        mode="lines+markers",
        name="Historical Actuals",
        line=dict(color="#38bdf8", width=2.5),
    ))

    # Forecast line
    forecast_x = [d.strftime("%Y-%m-%d") for d in future_dates]
    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=[f["forecast"] for f in forecast_records],
        mode="lines+markers",
        name="Projected Forecast",
        line=dict(color="#f59e0b", width=2.5, dash="dash"),
    ))

    # Upper bound
    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=[f["upper_bound_95"] for f in forecast_records],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
    ))

    # Lower bound with shading
    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=[f["lower_bound_95"] for f in forecast_records],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(245, 158, 11, 0.15)",
        name="95% Confidence Interval",
    ))

    fig.update_layout(
        title=f"Time-Series Forecast: {metric_col.capitalize()} (Next {periods} Intervals)",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Inter, sans-serif", size=13),
    )

    summary = (
        f"Generated {periods}-period forecast for '{metric_col}'. "
        f"Projected next interval value: {forecast_records[0]['forecast']:,.2f} "
        f"(95% CI: [{forecast_records[0]['lower_bound_95']:,.2f} - {forecast_records[0]['upper_bound_95']:,.2f}])."
    )

    return {
        "metric_col": metric_col,
        "periods": periods,
        "forecast_records": forecast_records,
        "plotly_spec": fig.to_dict(),
        "summary": summary,
    }


@register_tool(
    name="forecast_metric",
    description="Forecasts future values for a time series metric with 95% confidence intervals and visual projections.",
    parameter_schema={
        "type": "object",
        "properties": {
            "date_col": {"type": "string", "description": "Date column name."},
            "metric_col": {"type": "string", "description": "Numeric metric column to project."},
            "periods": {"type": "integer", "description": "Number of periods into the future to forecast (default: 3).", "default": 3},
            "freq": {"type": "string", "description": "Frequency ('ME' for month, 'W' for week).", "default": "ME"},
        },
        "required": ["date_col", "metric_col"],
    }
)
def tool_forecast_metric(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    periods: int = 3,
    freq: str = "ME",
) -> Dict[str, Any]:
    return forecast_metric(df=df, date_col=date_col, metric_col=metric_col, periods=periods, freq=freq)
