"""
Tests for the agent evaluation and benchmark framework.
"""

import pandas as pd
from src.utils.evaluation import run_benchmark


def test_agent_benchmark_accuracy():
    sample_df = pd.DataFrame({
        "date": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
        "region": ["North", "South", "East", "West"],
        "revenue": [5000.0, 7000.0, 3000.0, 100000.0],
        "profit": [1000.0, 1500.0, 500.0, 20000.0],
        "product": ["Laptops", "Keyboards", "Monitors", "Laptops"],
    })

    result = run_benchmark(sample_df)
    assert result.total_tests == 7
    assert result.passed_tests == 7
    assert result.accuracy_percentage == 100.0
    assert result.avg_latency_ms > 0.0
