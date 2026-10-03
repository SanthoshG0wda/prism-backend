"""
Comprehensive test suite verifying that PRISM works reliably with ANY CSV file.
Tests 5 distinct dataset domains:
1. Financial / Stock Market Data (OHLCV)
2. HR / Compensation Data (Column names containing spaces)
3. IoT / Sensor Readings (Timestamps, floats, missing data)
4. European Delimited Data (Semicolon ';' delimited)
5. Messy Real-World Data (Whitespace in headers, NaNs, special symbols)
"""

import io
import pytest
import pandas as pd
import numpy as np
import duckdb

from src.utils.csv import read_csv_bytes
from src.tools.profiling import profile_dataframe, check_data_quality
from src.tools.sql import execute_sql, validate_sql_safety
from src.tools.analysis import top_k_analysis, calculate_correlation
from src.tools.anomalies import detect_anomalies_iqr, detect_anomalies_zscore
from src.tools.charts import generate_chart
from src.services.llm import LLMService


# ==============================================================================
# 1. Financial / Stock Market Dataset
# ==============================================================================
STOCK_CSV = b"""Date,Ticker,Open,High,Low,Close,Volume,Dividends
2024-01-02,AAPL,187.15,188.44,183.89,185.64,82488700,0.0
2024-01-03,AAPL,184.22,185.88,183.43,184.25,58414500,0.0
2024-01-04,AAPL,182.15,183.09,180.88,181.91,71983600,0.0
2024-01-02,MSFT,373.86,376.22,370.18,370.87,25258600,0.0
2024-01-03,MSFT,369.01,373.26,368.51,370.60,23083500,0.0
2024-01-04,MSFT,370.67,373.10,367.17,367.94,20970100,0.0
2024-01-02,NVDA,492.23,499.97,476.90,481.68,41125400,0.0
2024-01-03,NVDA,474.85,481.84,470.20,475.69,32077900,0.0
2024-01-04,NVDA,477.67,485.00,475.08,479.98,30653500,0.0
2024-01-05,NVDA,484.62,495.47,483.06,490.97,999999999,0.0
"""

def test_financial_stock_dataset_end_to_end():
    df = read_csv_bytes(STOCK_CSV, "stocks.csv")
    assert len(df) == 10
    assert "Volume" in df.columns
    assert "Ticker" in df.columns

    # 1. Profile
    profile = profile_dataframe(df, "stocks")
    assert profile.row_count == 10
    assert profile.column_count == 8

    # 2. Quality
    quality = check_data_quality(df, "stocks")
    assert quality.completeness_score == 100.0

    # 3. SQL Query
    conn = duckdb.connect(database=":memory:")
    conn.register("active_dataset", df)
    res = execute_sql("SELECT Ticker, SUM(Volume) AS total_vol FROM active_dataset GROUP BY Ticker ORDER BY total_vol DESC", conn)
    assert len(res["records"]) == 3
    assert res["records"][0]["Ticker"] == "NVDA"

    # 4. Top-k Analysis (auto-corrects if user or prompt swaps categorical & metric)
    top_k = top_k_analysis(df, group_col="Ticker", metric_col="Volume", k=3)
    assert len(top_k["records"]) == 3
    assert top_k["records"][0]["Ticker"] == "NVDA"

    # Inverted args auto-repair
    top_k_swapped = top_k_analysis(df, group_col="Volume", metric_col="Ticker", k=3)
    assert len(top_k_swapped["records"]) == 3

    # 5. Outlier Detection on Volume (NVDA spike on 2024-01-05)
    anom = detect_anomalies_iqr(df, "Volume", multiplier=1.5)
    assert anom.anomalies_found >= 1
    assert any(a.value == 999999999 for a in anom.anomalies)

    # 6. Chart generation
    chart = generate_chart(df, chart_type="line", x="Date", y="Close", color="Ticker")
    assert chart["chart_type"] == "line"
    assert "data" in chart["plotly_spec"]


# ==============================================================================
# 2. HR / Compensation Dataset with Column Names Containing Spaces
# ==============================================================================
HR_CSV = b"""Employee ID,Full Name,Department Name,Job Title,Annual Salary,Performance Score,Hire Date
E001,Alice Smith,Engineering,Lead Architect,145000,4.8,2020-03-15
E002,Bob Jones,Engineering,Senior Backend Engineer,120000,4.2,2021-06-01
E003,Carol White,Marketing,Growth Director,130000,4.6,2019-11-12
E004,David Brown,Sales,Account Executive,95000,3.9,2022-01-10
E005,Emma Green,Sales,VP Sales,180000,4.9,2018-05-20
E006,Frank Black,Marketing,Copywriter,65000,3.5,2023-04-01
E007,Grace Miller,HR,People Operations Lead,90000,4.1,2021-08-15
E008,Henry Taylor,Engineering,Junior Developer,75000,3.8,2023-09-01
E009,Isabella Martinez,Sales,Sales Associate,70000,3.7,2022-10-05
E010,Jack Wilson,Executive,CEO,550000,5.0,2015-01-01
"""

def test_hr_dataset_with_spaced_columns():
    df = read_csv_bytes(HR_CSV, "hr_employees.csv")
    assert "Full Name" in df.columns
    assert "Annual Salary" in df.columns
    assert "Department Name" in df.columns

    # 1. SQL with quoted identifiers containing spaces
    conn = duckdb.connect(database=":memory:")
    conn.register("active_dataset", df)
    sql = 'SELECT "Department Name", AVG("Annual Salary") AS avg_sal FROM active_dataset GROUP BY "Department Name" ORDER BY avg_sal DESC'
    validate_sql_safety(sql, allowed_tables=["active_dataset"], allowed_columns=list(df.columns) + ["avg_sal"])
    res = execute_sql(sql, conn)
    assert len(res["records"]) == 5

    # 2. Top-k with spaces
    top_dept = top_k_analysis(df, group_col="Department Name", metric_col="Annual Salary", k=3)
    assert len(top_dept["records"]) == 3

    # 3. Anomaly detection (CEO salary is an outlier)
    anom = detect_anomalies_iqr(df, "Annual Salary", multiplier=1.5)
    assert anom.anomalies_found >= 1
    assert any(a.value == 550000 for a in anom.anomalies)

    # 4. Chart generation
    chart = generate_chart(df, chart_type="bar", x="Department Name", y="Annual Salary")
    assert chart["chart_type"] == "bar"


# ==============================================================================
# 3. IoT Sensor Dataset with Floating Points & Missing Values
# ==============================================================================
IOT_CSV = b"""timestamp,sensor_id,temperature_c,humidity_pct,pressure_hpa,battery_mv
2024-03-01T00:00:00Z,temp_node_1,21.5,45.2,1013.25,3290
2024-03-01T00:01:00Z,temp_node_1,21.6,45.1,1013.20,3289
2024-03-01T00:02:00Z,temp_node_1,,45.0,1013.18,3288
2024-03-01T00:03:00Z,temp_node_1,21.8,45.3,1013.22,3287
2024-03-01T00:04:00Z,temp_node_1,89.5,12.1,980.00,2100
2024-03-01T00:05:00Z,temp_node_1,21.7,45.0,1013.21,3285
2024-03-01T00:06:00Z,temp_node_1,21.5,,1013.23,3284
2024-03-01T00:07:00Z,temp_node_1,21.6,45.2,1013.24,3283
"""

def test_iot_sensor_dataset():
    df = read_csv_bytes(IOT_CSV, "sensors.csv")
    assert len(df) == 8

    # 1. Quality report detects missing cells
    quality = check_data_quality(df, "sensors")
    assert quality.missing_cells == 2

    # 2. Anomaly detection catches heat spike (89.5 C)
    anom = detect_anomalies_iqr(df, "temperature_c")
    assert anom.anomalies_found == 1
    assert anom.anomalies[0].value == 89.5

    # 3. Correlation analysis
    corr = calculate_correlation(df, numeric_cols=["temperature_c", "humidity_pct"])
    assert "top_correlated_pairs" in corr
    assert len(corr["top_correlated_pairs"]) >= 1


# ==============================================================================
# 4. European Semicolon-Delimited Dataset
# ==============================================================================
EUROPEAN_CSV = b"""order_id;client_name;city;sales_eur;tax_rate;order_date
1001;M\xfcller GmbH;Berlin;1520.50;0.19;2024-02-01
1002;Schmidt AG;Munich;3400.00;0.19;2024-02-02
1003;Weber KG;Hamburg;890.25;0.19;2024-02-03
1004;Meyer Logistik;Cologne;4500.80;0.07;2024-02-04
1005;Wagner Tech;Frankfurt;12300.00;0.19;2024-02-05
"""

def test_european_semicolon_delimited_dataset():
    df = read_csv_bytes(EUROPEAN_CSV, "german_orders.csv")
    assert len(df.columns) == 6
    assert "order_id" in df.columns
    assert "sales_eur" in df.columns
    assert len(df) == 5

    conn = duckdb.connect(database=":memory:")
    conn.register("active_dataset", df)
    res = execute_sql("SELECT city, SUM(sales_eur) AS total FROM active_dataset GROUP BY city ORDER BY total DESC", conn)
    assert len(res["records"]) == 5
    assert res["records"][0]["city"] == "Frankfurt"


# ==============================================================================
# 5. Messy CSV with Whitespace in Headers, Currency Symbols & Mixed Cases
# ==============================================================================
MESSY_CSV = b"""  customer_code  ,   region   ,   spend_amount  , active_flag 
CUST_001 , North , 1200.50 , True
CUST_002 , South , 450.00 , True
CUST_003 , East  , 3200.75 , False
CUST_004 , West  , 9999.00 , True
CUST_005 , North , 150.25 , True
"""

def test_messy_whitespace_csv():
    df = read_csv_bytes(MESSY_CSV, "messy.csv")
    # Headers should be trimmed automatically
    assert "customer_code" in df.columns
    assert "region" in df.columns
    assert "spend_amount" in df.columns

    # Offline LLM heuristic recognizes spend_amount as metric and region as group
    llm = LLMService()
    prompt = f"""
### Schema
| Column | Type |
| `customer_code` | VARCHAR |
| `region` | VARCHAR |
| `spend_amount` | DOUBLE |
| `active_flag` | BOOLEAN |

USER QUESTION: What is the top region by spend amount?
"""
    completion = llm._heuristic_offline_completion(prompt, system_prompt="You must select a selected_tool from the registry", json_mode=True)
    import json
    parsed = json.loads(completion)
    assert parsed["selected_tool"] in ("top_k_analysis", "execute_sql_query")
    if parsed["selected_tool"] == "top_k_analysis":
        assert parsed["tool_parameters"]["group_col"] == "region"
        assert parsed["tool_parameters"]["metric_col"] == "spend_amount"
