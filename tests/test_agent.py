"""
End-to-end test for DataAnalystAgent deterministic workflow.
"""

import pandas as pd
import pytest
from src.agent.analyst import DataAnalystAgent
from src.agent.state import SessionState
from src.services.llm import LLMService, LLMSettings


@pytest.fixture
def agent_with_data():
    state = SessionState()
    df = pd.DataFrame({
        "region": ["North", "South", "East", "West"],
        "revenue": [5000.0, 7000.0, 3000.0, 100000.0],  # West is an anomaly
        "profit": [1000.0, 1500.0, 500.0, 20000.0],
    })
    state.register_dataset("sales", df)

    # Use offline deterministic LLM heuristic service
    settings = LLMSettings(LLM_PROVIDER="mock")
    llm = LLMService(settings)
    return DataAnalystAgent(session_state=state, llm_service=llm)


def test_agent_top_customers_query(agent_with_data):
    response = agent_with_data.run("Which region generated the highest revenue?")
    assert response.tool_used == "top_k_analysis"
    assert len(response.steps_explanation) >= 5
    assert response.tool_result is not None
    assert response.tool_result["records"][0]["region"] == "West"


def test_agent_anomaly_query(agent_with_data):
    response = agent_with_data.run("Detect anomalies in revenue and explain why they were flagged")
    assert response.tool_used == "detect_anomalies"
    assert response.anomalies is not None
    assert len(response.anomalies) == 1
    assert response.anomalies[0].value == 100000.0


def test_agent_chart_query(agent_with_data):
    response = agent_with_data.run("Generate a bar chart of revenue by region")
    assert response.tool_used == "generate_chart"
    assert response.chart_spec is not None
    assert "data" in response.chart_spec


def test_agent_sql_query(agent_with_data):
    response = agent_with_data.run("Generate SQL query for region sales")
    assert response.tool_used == "execute_sql_query"
    assert response.generated_sql is not None
    assert "SELECT" in response.generated_sql


def test_agent_conversational_concept_query(agent_with_data):
    response = agent_with_data.run("What is DuckDB?")
    assert response.tool_used == "conversational_agent"
    assert "DuckDB" in response.answer
    assert response.tool_result is None


def test_agent_special_abilities_query(agent_with_data):
    response = agent_with_data.run("What are your special abilities?")
    assert response.tool_used == "conversational_greeting"
    # Offline deterministic greeting: compact capabilities + live dataset facts.
    assert "capabilities" in response.answer.lower()
    assert "No dataset" not in response.answer
    assert "Attach a CSV" not in response.answer
    assert "Special Analytical Superpowers" not in response.answer


def test_agent_conversational_when_no_data():
    empty_state = SessionState()
    settings = LLMSettings(LLM_PROVIDER="mock")
    llm = LLMService(settings)
    agent = DataAnalystAgent(session_state=empty_state, llm_service=llm)

    response = agent.run("Explain how Tukey IQR anomaly detection works")
    assert response.tool_used == "conversational_agent"
    assert "Tukey" in response.answer or "IQR" in response.answer


def test_agent_dataset_required_notice_when_no_data():
    empty_state = SessionState()
    settings = LLMSettings(LLM_PROVIDER="mock")
    llm = LLMService(settings)
    agent = DataAnalystAgent(session_state=empty_state, llm_service=llm)

    response = agent.run("What are the top 5 customers by revenue?")
    assert response.tool_used == "dataset_required_notice"
    assert "upload a CSV" in response.answer


def test_agent_dashboard_artifact(agent_with_data):
    response = agent_with_data.run("Generate an Executive Dashboard for sales")
    assert response.tool_used == "generate_dashboard_artifact"
    assert response.artifact is not None
    assert response.artifact["type"] == "dashboard"
    assert "kpis" in response.artifact["data"]
    assert response.artifact["table_name"] == "sales"


def test_agent_dashboard_specific_table_resolution():
    state = SessionState()
    df_sales = pd.DataFrame({"sales_col": [1, 2, 3]})
    df_cust = pd.DataFrame({"cust_col": ["Alice", "Bob"], "spend": [100, 200]})
    state.register_dataset("sales_data", df_sales)
    state.register_dataset("customers", df_cust)

    settings = LLMSettings(LLM_PROVIDER="mock")
    llm = LLMService(settings)
    agent = DataAnalystAgent(session_state=state, llm_service=llm)

    response = agent.run("Generate an Executive Dashboard artifact for customers")
    assert response.tool_used == "generate_dashboard_artifact"
    assert response.artifact is not None
    assert response.artifact["table_name"] == "customers"
    assert response.artifact["data"]["kpis"]["total_rows"] == 2


def test_agent_list_all_rows(agent_with_data):
    response = agent_with_data.run("list all rows")
    assert response.tool_used == "execute_sql_query"
    assert response.tool_result is not None
    assert "records" in response.tool_result
    assert len(response.tool_result["records"]) == 4


def test_agent_list_all_rows_in_sales_data():
    state = SessionState()
    df_sales = pd.DataFrame({"region": ["A", "B"], "revenue": [10, 20]})
    state.register_dataset("sales_data", df_sales)

    settings = LLMSettings(LLM_PROVIDER="mock")
    llm = LLMService(settings)
    agent = DataAnalystAgent(session_state=state, llm_service=llm)

    response = agent.run("list all rows in the sales data file")
    assert response.tool_used == "execute_sql_query"
    assert response.tool_result is not None
    assert "records" in response.tool_result
    assert len(response.tool_result["records"]) == 2


