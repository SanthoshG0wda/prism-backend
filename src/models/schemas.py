"""
Data models and schemas for the AI Data Analyst system using Pydantic v2.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    """Profile of a single dataset column."""
    name: str
    dtype: str
    null_count: int
    null_percentage: float
    distinct_count: int
    sample_values: List[Any] = Field(default_factory=list)
    min_value: Optional[Union[float, int, str]] = None
    max_value: Optional[Union[float, int, str]] = None
    mean: Optional[float] = None
    std: Optional[float] = None


class DatasetMetadata(BaseModel):
    """Metadata describing an uploaded tabular dataset."""
    table_name: str
    original_filename: str
    row_count: int
    column_count: int
    memory_bytes: int
    columns: List[ColumnProfile] = Field(default_factory=list)


class DataQualityReport(BaseModel):
    """Structured report evaluating data quality and anomalies."""
    table_name: str
    row_count: int
    completeness_score: float = Field(
        ..., description="Percentage of non-null cells across the entire dataset (0.0 to 100.0)"
    )
    duplicate_rows: int
    missing_cells: int
    quality_issues: List[str] = Field(default_factory=list)
    column_profiles: List[ColumnProfile] = Field(default_factory=list)


class AnomalyItem(BaseModel):
    """Individual flagged anomaly record with deterministic attribution."""
    column: str
    row_index: int
    value: Union[float, int, str]
    method: Literal["iqr", "z_score"]
    score: float = Field(..., description="Z-score value or distance from IQR boundary")
    explanation: str


class AnomalyDetectionResult(BaseModel):
    """Aggregated anomaly detection outcome for a dataset."""
    table_name: str
    column_analyzed: str
    method: Literal["iqr", "z_score"]
    threshold: float
    anomalies_found: int
    anomalies: List[AnomalyItem] = Field(default_factory=list)
    summary: str


class ToolCall(BaseModel):
    """Represents a planned tool invocation selected by the agent."""
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    purpose: str = ""


class ToolExecutionResult(BaseModel):
    """Standardized result returned by any deterministic analysis tool."""
    tool_name: str
    success: bool
    result_data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    summary: str = ""


class QueryPlan(BaseModel):
    """The structured analysis plan formulated by the agent."""
    user_intent: str
    reasoning: str
    selected_tool: str
    tool_parameters: Dict[str, Any] = Field(default_factory=dict)
    generated_sql: Optional[str] = None
    generated_pandas_code: Optional[str] = None


class AgentResponse(BaseModel):
    """The complete response contract returned to the UI layer."""
    question: str
    answer: str
    steps_explanation: List[str] = Field(
        default_factory=list,
        description="Transparent breakdown of steps taken to obtain the answer"
    )
    tool_used: Optional[str] = None
    tool_result: Optional[Any] = None
    generated_sql: Optional[str] = None
    generated_pandas_code: Optional[str] = None
    chart_spec: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Plotly JSON-compatible figure dictionary"
    )
    anomalies: Optional[List[AnomalyItem]] = None
    data_quality: Optional[DataQualityReport] = None
    artifact: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Claude-style interactive artifact (e.g. dashboard, dataset visualizer, report)"
    )
    execution_time_ms: float = 0.0


class ChatMessage(BaseModel):
    """Represents a message in the analyst conversation session."""
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
