"""
Tests for the agentic workflow: follow-ups, retry, JOINs, scan-all,
auto-chart, SQL validation, and grounded offline synthesis.
"""

import pandas as pd
import pytest

from src.agent.analyst import DataAnalystAgent
from src.agent.state import SessionState
from src.services.llm import LLMService, LLMSettings


@pytest.fixture
def sales_state():
    state = SessionState()
    sales = pd.DataFrame({
        "region": ["North", "South", "East", "West", "North", "South"],
        "revenue": [5000.0, 7000.0, 3000.0, 100000.0, 6000.0, 8000.0],
        "profit": [1000.0, 1500.0, 500.0, 20000.0, 1200.0, 1600.0],
    })
    cust = pd.DataFrame({
        "region": ["North", "South", "East", "West"],
        "manager": ["Alice", "Bob", "Charlie", "Dana"],
    })
    state.register_dataset("sales_data", sales)
    state.register_dataset("customers", cust)
    state.set_active_dataset("sales_data")
    return state


@pytest.fixture
def agent(sales_state):
    llm = LLMService(LLMSettings(LLM_PROVIDER="mock"))
    return DataAnalystAgent(session_state=sales_state, llm_service=llm)


def test_followup_inherits_prior_columns(agent):
    first = agent.run("Which region generated the highest revenue?")
    assert first.tool_used == "top_k_analysis"
    second = agent.run("show it as a pie chart")
    assert second.tool_used == "generate_chart"
    assert second.tool_result is not None


def test_retry_and_fallback_never_returns_none_tool_result(agent):
    # Nonsense column reference must be repaired, not crash to None.
    resp = agent.run("Which zzz_nonexistent generated the highest revenue?")
    assert resp.tool_result is not None


def test_join_two_tables(agent):
    resp = agent.run("Join sales_data with customers on region")
    assert resp.tool_used == "execute_sql_query"
    assert resp.tool_result is not None
    assert resp.tool_result["row_count"] >= 4
    assert "manager" in resp.tool_result["columns"]


def test_anomaly_scan_all_without_column(agent):
    resp = agent.run("Detect anomalies in the dataset.")
    assert resp.tool_used == "detect_anomalies"
    assert resp.tool_result is not None
    assert "columns_scanned" in resp.tool_result
    assert len(resp.tool_result["columns_scanned"]) >= 2
    assert resp.anomalies is not None and len(resp.anomalies) >= 1


def test_ranking_auto_attaches_chart(agent):
    resp = agent.run("Which region generated the highest revenue?")
    assert resp.tool_used == "top_k_analysis"
    assert resp.chart_spec is not None
    assert "data" in resp.chart_spec


def test_sql_is_validated_and_scoped(agent):
    resp = agent.run("Generate SQL for sales by region.")
    assert resp.tool_used == "execute_sql_query"
    assert resp.generated_sql is not None
    assert "sales_data" in resp.generated_sql or "active_dataset" not in resp.generated_sql
    assert "LIMIT" in resp.generated_sql.upper()


def test_offline_synthesis_contains_real_numbers(agent):
    resp = agent.run("Which region generated the highest revenue?")
    assert "West" in resp.answer
    assert "100000" in resp.answer.replace(",", "").replace(".0", "")


def test_all_assignment_example_questions(sales_state):
    llm = LLMService(LLMSettings(LLM_PROVIDER="mock"))
    agent = DataAnalystAgent(session_state=sales_state, llm_service=llm)
    questions = [
        "Which region generated the highest revenue?",
        "Show monthly sales trends.",
        "Which products are underperforming?",
        "What are the top five customers?",
        "Generate SQL for this analysis.",
        "Detect anomalies in the dataset.",
    ]
    for q in questions:
        resp = agent.run(q)
        assert resp.tool_result is not None, f"No result for: {q}"
        assert resp.answer and len(resp.answer) > 20, f"Empty answer for: {q}"


def test_run_stream_matches_run(agent):
    streamed = [e for e in agent.run_stream("Which region generated the highest revenue?")]
    types = [e["type"] for e in streamed]
    assert "status" in types
    assert "token" in types
    results = [e for e in streamed if e["type"] == "result"]
    assert len(results) == 1
    final = results[0]["response"]
    direct = agent.run("Which region generated the highest revenue?")
    assert final.tool_used == direct.tool_used == "top_k_analysis"
    assert "".join(e.get("text", "") for e in streamed if e["type"] == "token").strip() == final.answer.strip()


def test_run_stream_greeting_streams(agent):
    events = [e for e in agent.run_stream("hello")]
    assert events[0]["type"] == "status"
    assert any(e["type"] == "token" for e in events)
    assert events[-1]["type"] == "result"
    assert events[-1]["response"].tool_used == "conversational_greeting"


def test_equiv_routing_top_customers_on_sample_schema():
    # "customers" must resolve to CUSTOMERNAME (ratio 0.76 < fuzzy 0.8).
    from src.agent.state import SessionState as _S
    st = _S()
    st.register_dataset("s", pd.DataFrame({"CUSTOMERNAME": ["A", "B"], "SALES": [10.0, 20.0]}))
    agent = DataAnalystAgent(session_state=st, llm_service=LLMService(LLMSettings(LLM_PROVIDER="mock")))
    assert agent._is_dataset_query("What are the top five customers?")
    resp = agent.run("What are the top five customers?")
    assert resp.tool_used == "top_k_analysis"
    assert resp.tool_result is not None


def test_short_table_name_no_false_positive():
    from src.agent.state import SessionState as _S
    st = _S()
    st.register_dataset("s", pd.DataFrame({"a": [1]}))
    agent = DataAnalystAgent(session_state=st, llm_service=LLMService(LLMSettings(LLM_PROVIDER="mock")))
    assert not agent._is_dataset_query("What is DuckDB?")
    assert agent._is_dataset_query("list all rows in s")


def test_region_revenue_maps_to_territory_sales():
    from src.agent.state import SessionState as _S
    st = _S()
    st.register_dataset(
        "s",
        pd.DataFrame({"TERRITORY": ["EMEA", "APAC"], "SALES": [100.0, 50.0],
                      "ORDERNUMBER": ["1", "2"], "CITY": ["X", "Y"]}),
    )
    agent = DataAnalystAgent(session_state=st, llm_service=LLMService(LLMSettings(LLM_PROVIDER="mock")))
    resp = agent.run("Which region generated the highest revenue?")
    recs = resp.tool_result["records"]
    assert recs[0]["TERRITORY"] == "EMEA"
    assert "EMEA" in resp.answer