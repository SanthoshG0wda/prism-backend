"""
Assignment-flow tests: sessions start EMPTY (no sample auto-loading).
The user uploads CSVs first; analysis only runs on user-provided data.
"""

import pytest
import pandas as pd

from src.agent.analyst import DataAnalystAgent
from src.agent.sessions import SessionManager
from src.services.llm import LLMService, LLMSettings
from src.utils.sqlite_store import SQLiteSessionStore


def make_agent(state):
    llm = LLMService(LLMSettings(LLM_PROVIDER="mock"))
    return DataAnalystAgent(session_state=state, llm_service=llm)


@pytest.fixture
def mgr(tmp_path):
    store = SQLiteSessionStore(str(tmp_path / "test_sessions.db"))
    return SessionManager(store)


def test_fresh_session_starts_empty(mgr):
    _, state = mgr.get_or_create("never-seen-before")
    assert state.datasets == {}
    assert state.active_dataset_name is None
    assert state.conversation_history == []


def test_default_session_starts_empty(mgr):
    _, state = mgr.get_or_create(None)
    assert state.datasets == {}


def test_sessions_are_isolated(mgr):
    _, a = mgr.get_or_create("user-a")
    _, b = mgr.get_or_create("user-b")
    a.register_dataset("mine", pd.DataFrame({"x": [1, 2]}))
    assert "mine" in a.datasets
    assert "mine" not in b.datasets


def test_analytical_query_with_no_upload_asks_for_csv(mgr):
    _, state = mgr.get_or_create("empty-user")
    resp = make_agent(state).run("Which region generated the highest revenue?")
    assert resp.tool_used == "dataset_required_notice"
    assert resp.tool_result is None
    assert "upload" in resp.answer.lower()


def test_upload_then_analyze_flow(mgr):
    """Simulates the assignment flow: upload CSV -> ask questions."""
    _, state = mgr.get_or_create("uploader")
    # 1. Empty at first
    assert make_agent(state).run("Show monthly sales trends.").tool_used == "dataset_required_notice"
    # 2. User uploads a CSV
    df = pd.DataFrame({
        "region": ["North", "South", "East"],
        "revenue": [500.0, 700.0, 1200.0],
    })
    state.register_dataset("my_sales", df)
    # 3. Same question now analyzes the uploaded data
    resp = make_agent(state).run("Which region generated the highest revenue?")
    assert resp.tool_used == "top_k_analysis"
    assert resp.tool_result is not None
    assert resp.tool_result["records"][0]["region"] == "East"


def test_greeting_omits_dataset_and_upload_state(mgr):
    _, state = mgr.get_or_create("fresh-greeter")
    resp = make_agent(state).run("hello")
    assert resp.tool_used == "conversational_greeting"
    assert "No dataset" not in resp.answer
    assert "Attach a CSV" not in resp.answer
    assert "My capabilities" not in resp.answer
    assert len(resp.answer) < 300


def test_capability_question_still_lists_capabilities(mgr):
    from src.agent.state import SessionState as _S
    _, state = mgr.get_or_create("caps-asker")
    resp = make_agent(state).run("what can you do")
    assert resp.tool_used == "conversational_greeting"
    assert "capabilities" in resp.answer.lower()
    assert "No dataset" not in resp.answer
    assert "Try next" not in resp.answer