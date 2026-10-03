"""
Tests for deterministic anomaly detection (IQR and Z-Score).
"""

import pandas as pd
import pytest
from src.tools.anomalies import detect_anomalies_iqr, detect_anomalies_zscore


@pytest.fixture
def df_with_outlier():
    # Regular values around 100, one massive outlier at 10,000
    values = [100.0, 102.0, 98.0, 105.0, 99.0, 101.0, 97.0, 103.0, 10000.0]
    return pd.DataFrame({"revenue": values})


def test_iqr_anomaly_detection(df_with_outlier):
    result = detect_anomalies_iqr(df_with_outlier, column="revenue", multiplier=1.5)
    assert result.anomalies_found == 1
    anom = result.anomalies[0]
    assert anom.value == 10000.0
    assert anom.method == "iqr"
    assert "above the upper fence" in anom.explanation


def test_zscore_anomaly_detection(df_with_outlier):
    result = detect_anomalies_zscore(df_with_outlier, column="revenue", threshold=2.0)
    assert result.anomalies_found == 1
    anom = result.anomalies[0]
    assert anom.value == 10000.0
    assert anom.method == "z_score"
    assert anom.score > 2.0
    assert "standard deviations" in anom.explanation


def test_no_anomalies_uniform_data():
    df_uniform = pd.DataFrame({"revenue": [10.0, 11.0, 10.5, 9.8, 10.2, 10.1]})
    result = detect_anomalies_iqr(df_uniform, column="revenue", multiplier=1.5)
    assert result.anomalies_found == 0
