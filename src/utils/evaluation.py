"""
Automated Evaluation and Benchmarking Framework for AI Data Analyst.
Evaluates agent tool selection accuracy, deterministic computation validity, and response latency.
"""

import time
from typing import Any, Dict, List
import pandas as pd
from pydantic import BaseModel, Field
from src.agent.analyst import DataAnalystAgent
from src.agent.state import SessionState
from src.services.llm import LLMService, LLMSettings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TestCase(BaseModel):
    """Single benchmark evaluation test case."""
    name: str
    query: str
    expected_tool: str
    requires_table_loaded: bool = True


class BenchmarkResult(BaseModel):
    """Result of running an evaluation test suite."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    accuracy_percentage: float
    avg_latency_ms: float
    detailed_results: List[Dict[str, Any]] = Field(default_factory=list)


BENCHMARK_SUITE: List[TestCase] = [
    TestCase(
        name="Top Entities Query",
        query="Which region generated the highest revenue?",
        expected_tool="top_k_analysis",
    ),
    TestCase(
        name="Outlier Detection Query",
        query="Detect anomalies in revenue and explain why they were flagged",
        expected_tool="detect_anomalies",
    ),
    TestCase(
        name="Time Series Trend Query",
        query="Show the monthly sales trend",
        expected_tool="time_series_trend",
    ),
    TestCase(
        name="Forecasting Query",
        query="Forecast revenue for next 3 months",
        expected_tool="forecast_metric",
    ),
    TestCase(
        name="Data Quality Audit Query",
        query="Run a data quality check on the dataset",
        expected_tool="check_data_quality",
    ),
    TestCase(
        name="Visualization Query",
        query="Generate a bar chart of profit across regions",
        expected_tool="generate_chart",
    ),
    TestCase(
        name="SQL Query Generation",
        query="Generate a SQL query for regional sales",
        expected_tool="execute_sql_query",
    ),
]


def run_benchmark(sample_df: pd.DataFrame) -> BenchmarkResult:
    """
    Executes benchmark test cases through the agent and computes accuracy and latency metrics.
    """
    state = SessionState()
    state.register_dataset("sales_data", sample_df)

    llm = LLMService(LLMSettings(LLM_PROVIDER="mock"))
    agent = DataAnalystAgent(session_state=state, llm_service=llm)

    passed = 0
    total = len(BENCHMARK_SUITE)
    latencies = []
    details = []

    for test in BENCHMARK_SUITE:
        start = time.perf_counter()
        resp = agent.run(test.query)
        elapsed = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed)

        is_correct = resp.tool_used == test.expected_tool
        if is_correct:
            passed += 1

        details.append({
            "test_name": test.name,
            "query": test.query,
            "expected_tool": test.expected_tool,
            "actual_tool": resp.tool_used,
            "passed": is_correct,
            "latency_ms": round(elapsed, 2),
            "tool_success": resp.tool_result is not None,
        })

    accuracy = round((passed / total) * 100.0, 1)
    avg_lat = round(sum(latencies) / len(latencies), 2)

    return BenchmarkResult(
        total_tests=total,
        passed_tests=passed,
        failed_tests=total - passed,
        accuracy_percentage=accuracy,
        avg_latency_ms=avg_lat,
        detailed_results=details,
    )
