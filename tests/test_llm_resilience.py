"""
Regression tests for the live-LLM failure modes seen in production logs:
- reasoning models returning content=null (muse-glimmer-30b separates reasoning output)
- malformed bodies, empty answers (AgentResponse 500: answer=None)
- upload-language routing ("analyze this file")
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agent.analyst import DataAnalystAgent
from src.agent.state import SessionState
from src.services.llm import LLMService, LLMSettings


def _configured_service():
    return LLMService(LLMSettings(
        LLM_PROVIDER="nvidia",
        LLM_API_KEY="nvapi-test-key-12345",
        LLM_MODEL="meta/muse-glimmer-30b",
        LLM_TIMEOUT_SECONDS=1,
    ))


def _mock_response(payload, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.text = str(payload)
    return resp


def test_reasoning_trace_never_served_as_answer():
    # Null content + thinking trace must raise (deterministic fallback engages),
    # never leak the private trace into the chat.
    svc = _configured_service()
    payload = {"choices": [{"message": {"content": None, "reasoning_content": "We need to greet briefly..."}}]}
    with patch("src.services.llm.requests.post", return_value=_mock_response(payload)):
        with pytest.raises(RuntimeError, match="empty content"):
            svc.generate("hi")


def test_null_content_without_reasoning_raises_not_none():
    svc = _configured_service()
    payload = {"choices": [{"message": {"content": None}}]}
    with patch("src.services.llm.requests.post", return_value=_mock_response(payload)):
        with pytest.raises(RuntimeError, match="empty content"):
            svc.generate("hi")


def test_malformed_body_raises_not_none():
    svc = _configured_service()
    with patch("src.services.llm.requests.post", return_value=_mock_response({"oops": 1})):
        with pytest.raises(RuntimeError, match="no choices"):
            svc.generate("hi")


def test_none_answer_never_reaches_response():
    state = SessionState()
    state.register_dataset("t", pd.DataFrame({"region": ["A", "B"], "revenue": [1.0, 2.0]}))
    agent = DataAnalystAgent(session_state=state, llm_service=_configured_service())
    with patch.object(LLMService, "generate_structured", side_effect=ValueError("plan boom")), \
         patch.object(LLMService, "generate", return_value=None):
        resp = agent.run("Which region generated the highest revenue?")
    assert isinstance(resp.answer, str) and resp.answer.strip()
    assert resp.tool_result is not None


def test_analyze_this_file_routes_to_dataset():
    state = SessionState()
    state.register_dataset("sales_data_sample", pd.DataFrame({"a": [1]}))
    agent = DataAnalystAgent(
        session_state=state,
        llm_service=LLMService(LLMSettings(LLM_PROVIDER="mock")),
    )
    assert agent._is_dataset_query("analye this file")
    assert agent._is_dataset_query("summarize the sales data")


def test_reasoning_only_stream_raises():
    # A stream of pure thinking deltas must raise, not stream thoughts.
    from unittest.mock import MagicMock
    svc = _configured_service()
    lines = [
        'data: {"choices": [{"delta": {"reasoning_content": "We need to greet"}}]}',
        'data: {"choices": [{"delta": {"reasoning_content": " briefly."}}]}',
        'data: [DONE]',
    ]
    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = lines
    resp.__enter__.return_value = resp
    with patch("src.services.llm.requests.post", return_value=resp):
        with pytest.raises(RuntimeError, match="no answer content"):
            list(svc.generate_stream("hi"))


def test_default_budget_fits_reasoning_models():
    svc = _configured_service()
    assert svc.settings.max_tokens >= 2048

def test_unicode_survives_without_charset_header():
    # NIM omits charset: requests would default to Latin-1 and mangle ≈/—.
    import json as _json
    from unittest.mock import patch
    from requests.models import Response
    body = _json.dumps(
        {"choices": [{"message": {"content": "Total ≈ $4,979,272.41 — done"}}]},
        ensure_ascii=False,
    ).encode("utf-8")
    resp = Response()
    resp.status_code = 200
    resp._content = body
    resp.headers["Content-Type"] = "application/json"  # no charset!
    assert resp.encoding is None or "utf" not in (resp.encoding or "").lower()
    svc = _configured_service()
    with patch("src.services.llm.requests.post", return_value=resp):
        out = svc.generate("hi")
    assert "≈" in out and "—" in out
    assert "â" not in out