"""
Persistence tests: uploads + conversation context survive restarts via SQLite.
"""

import pandas as pd

from src.agent.analyst import DataAnalystAgent
from src.agent.sessions import SessionManager
from src.services.llm import LLMService, LLMSettings
from src.utils.sqlite_store import SQLiteSessionStore


def _managers(tmp_path):
    db = str(tmp_path / "sessions.db")
    return SessionManager(SQLiteSessionStore(db)), SessionManager(SQLiteSessionStore(db))


def test_fresh_session_still_starts_empty(tmp_path):
    mgr, _ = _managers(tmp_path)
    _, state = mgr.get_or_create("brand-new")
    assert state.datasets == {}
    assert state.conversation_history == []


def test_upload_and_history_survive_manager_restart(tmp_path):
    mgr1, mgr2 = _managers(tmp_path)
    _, s1 = mgr1.get_or_create("persist-me")
    raw = "region,revenue\nNorth,500\nSouth,700\n".encode("utf-8")
    s1.register_dataset("sales", pd.read_csv(__import__("io").BytesIO(raw)),
                        source_bytes=raw, filename="sales.csv")
    s1.add_message("user", "Which region is best?")
    s1.add_message("assistant", "South.", metadata={"tool_used": "top_k_analysis"})

    # Simulate server restart: brand-new manager, same DB file.
    _, s2 = mgr2.get_or_create("persist-me")
    assert set(s2.datasets) == {"sales"}
    assert len(s2.datasets["sales"]) == 2
    assert s2.active_dataset_name == "sales"
    assert [m.role for m in s2.conversation_history] == ["user", "assistant"]
    assert s2.conversation_history[1].metadata["tool_used"] == "top_k_analysis"
    # Restored frame is queryable in-memory (DuckDB re-registered).
    assert s2.get_active_df() is not None


def test_followup_context_survives_restart(tmp_path):
    mgr1, mgr2 = _managers(tmp_path)
    _, s1 = mgr1.get_or_create("ctx")
    raw = "region,revenue\nNorth,500\nSouth,700\n".encode("utf-8")
    s1.register_dataset("sales", pd.read_csv(__import__("io").BytesIO(raw)),
                        source_bytes=raw, filename="sales.csv")
    llm = LLMService(LLMSettings(LLM_PROVIDER="mock"))
    DataAnalystAgent(session_state=s1, llm_service=llm).run(
        "Which region generated the highest revenue?"
    )
    _, s2 = mgr2.get_or_create("ctx")
    agent2 = DataAnalystAgent(session_state=s2, llm_service=llm)
    assert agent2._last_dataset_context() is not None
    resp = agent2.run("show it as a pie chart")
    assert resp.tool_used == "generate_chart"
    assert resp.tool_result is not None


def test_clear_deletes_persisted_session(tmp_path):
    mgr, mgr2 = _managers(tmp_path)
    _, s1 = mgr.get_or_create("doomed")
    raw = "a\n1\n".encode()
    s1.register_dataset("t", pd.read_csv(__import__("io").BytesIO(raw)),
                        source_bytes=raw, filename="t.csv")
    assert mgr.clear("doomed")
    _, s2 = mgr2.get_or_create("doomed")
    assert s2.datasets == {}
    assert s2.conversation_history == []


def test_unbound_state_does_not_touch_db(tmp_path):
    # Plain SessionState (tests/tools) works with no store attached.
    from src.agent.state import SessionState
    st = SessionState()
    st.register_dataset("t", pd.DataFrame({"a": [1]}))
    st.add_message("user", "hi")
    st.clear_history()
    assert st.conversation_history == []


def test_agent_response_with_numpy_payload_serializes():
    """Regression: plotly/pandas numpy types must not 500 response validation."""
    import numpy as np
    from src.models.schemas import AgentResponse
    from src.utils.json_safe import json_safe
    payload = {
        "records": [{"g": "A", "v": np.int64(5)}],
        "plotly_spec": {"data": [{"y": np.array([1, 2])}]},
    }
    resp = AgentResponse(
        question="q",
        answer="a",
        tool_used="top_k_analysis",
        tool_result=json_safe(payload),
        chart_spec=json_safe(payload["plotly_spec"]),
    )
    # This is what FastAPI response_model validation does.
    AgentResponse.model_validate(resp.model_dump(mode="json"))
