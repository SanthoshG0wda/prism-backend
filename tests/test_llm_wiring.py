"""
Tests for meta/muse-glimmer-30b wiring on NVIDIA NIM:
model alias resolution, LLM status reporting, and honest
online/offline greeting.
"""

import pandas as pd

from src.agent.analyst import DataAnalystAgent
from src.agent.state import SessionState
from src.services.llm import LLMService, LLMSettings, MODEL_ALIASES


def test_shorthand_alias_resolves_to_glimmer():
    svc = LLMService(LLMSettings(LLM_PROVIDER="nvidia", LLM_MODEL="muse-glimmer"))
    assert svc.settings.model == "meta/muse-glimmer-30b"


def test_glimmer_alias_map_covers_variants():
    assert MODEL_ALIASES["muse-glimmer"] == "meta/muse-glimmer-30b"
    assert MODEL_ALIASES["glimmer"] == "meta/muse-glimmer-30b"


def test_default_nvidia_model_is_glimmer():
    svc = LLMService(LLMSettings(LLM_PROVIDER="nvidia", LLM_MODEL=""))
    assert svc.settings.model == "meta/muse-glimmer-30b"


def test_unconfigured_service_is_offline():
    svc = LLMService(LLMSettings(LLM_PROVIDER="nvidia", LLM_API_KEY=""))
    assert not svc.is_configured()


def test_offline_greeting_discloses_heuristic_mode():
    state = SessionState()
    agent = DataAnalystAgent(
        session_state=state,
        llm_service=LLMService(LLMSettings(LLM_PROVIDER="mock")),
    )
    resp = agent.run("hello")
    assert resp.tool_used == "conversational_greeting"
    assert "offline" in resp.answer.lower()
    assert "My capabilities" not in resp.answer


def test_online_greeting_is_llm_generated_not_template():
    from unittest.mock import patch
    state = SessionState()
    agent = DataAnalystAgent(
        session_state=state,
        llm_service=LLMService(LLMSettings(
            LLM_PROVIDER="nvidia",
            LLM_API_KEY="nvapi-test-key-12345",
            LLM_MODEL="muse-glimmer",
        )),
    )
    assert agent.llm.is_configured()
    assert agent.llm.settings.model == "meta/muse-glimmer-30b"

    def _fake_stream(prompt, system_prompt=None):
        yield from ["Hello from ", "meta/muse-glimmer-30b, at your service!"]

    with patch.object(LLMService, "generate_stream", side_effect=_fake_stream):
        resp = agent.run("hello")
    assert resp.tool_used == "conversational_greeting"
    assert resp.answer == "Hello from meta/muse-glimmer-30b, at your service!"
    assert "Special Analytical Superpowers" not in resp.answer
    assert "offline" not in resp.answer.lower()


def test_online_greeting_streams_tokens():
    from unittest.mock import patch
    state = SessionState()
    agent = DataAnalystAgent(
        session_state=state,
        llm_service=LLMService(LLMSettings(
            LLM_PROVIDER="nvidia",
            LLM_API_KEY="nvapi-test-key-12345",
        )),
    )

    def _fake_stream(prompt, system_prompt=None):
        yield from ["Hi", " there!"]

    with patch.object(LLMService, "generate_stream", side_effect=_fake_stream):
        events = list(agent.run_stream("hi"))
    tokens = "".join(e.get("text", "") for e in events if e["type"] == "token")
    assert tokens == "Hi there!"
    assert events[-1]["type"] == "result"


def _mock_get(payload=None, status=200):
    from unittest.mock import MagicMock
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload or {}
    resp.text = str(payload)
    return resp


def test_check_connection_ok_and_model_listed():
    from unittest.mock import patch
    svc = LLMService(LLMSettings(
        LLM_PROVIDER="nvidia", LLM_API_KEY="nvapi-good-key-12345", LLM_MODEL="muse-glimmer",
    ))
    payload = {"data": [{"id": "meta/muse-glimmer-30b"}, {"id": "meta/llama-3.3-70b-instruct"}]}
    completion = {"choices": [{"message": {"content": "ok"}}]}
    with patch("src.services.llm.requests.get", return_value=_mock_get(payload)), \
         patch("src.services.llm.requests.post", return_value=_mock_get(completion)):
        out = svc.check_connection()
    assert out["ok"] is True
    assert out["key_valid"] is True
    assert out["model_listed"] is True
    assert out["models_count"] == 2


def test_check_connection_rejects_bad_key():
    from unittest.mock import patch
    import pytest as _pytest
    svc = LLMService(LLMSettings(LLM_PROVIDER="nvidia", LLM_API_KEY="nvapi-bad-key-1"))
    payload = {"data": [{"id": "meta/muse-glimmer-30b"}]}
    with patch("src.services.llm.requests.get", return_value=_mock_get(payload)), \
         patch("src.services.llm.requests.post", return_value=_mock_get(status=401)):
        with _pytest.raises(RuntimeError, match="Key rejected"):
            svc.check_connection()


def test_check_connection_requires_key():
    import pytest as _pytest
    svc = LLMService(LLMSettings(LLM_PROVIDER="nvidia", LLM_API_KEY=""))
    with _pytest.raises(RuntimeError, match="No API key"):
        svc.check_connection()


def test_check_connection_flags_unlisted_model():
    from unittest.mock import patch
    svc = LLMService(LLMSettings(
        LLM_PROVIDER="nvidia", LLM_API_KEY="nvapi-good-key-12345",
        LLM_MODEL="meta/llama-3.3-70b-instruct",
    ))
    payload = {"data": [{"id": "meta/muse-glimmer-30b"}]}
    completion = {"choices": [{"message": {"content": "ok"}}]}
    with patch("src.services.llm.requests.get", return_value=_mock_get(payload)), \
         patch("src.services.llm.requests.post", return_value=_mock_get(completion)):
        out = svc.check_connection()
    assert out["ok"] is True
    assert out["key_valid"] is True
    assert out["model_listed"] is False

def test_strip_prompt_echo_removes_scaffolding_and_thinking():
    raw = (
        "ENGINE: NVIDIA NIM (meta/muse-glimmer-30b)\n"
        "SESSION DATASETS:\nNo dataset currently uploaded.\n"
        "We need to greet briefly, summarize capabilities compactly.\n"
        "System prompt says: greet briefly.\n"
        "Let\u2019s produce\n"
        "Output.\n"
        "Hello! I am your AI Analyst.\n"
        "No dataset currently uploaded."
    )
    from src.agent.analyst import DataAnalystAgent
    cleaned = DataAnalystAgent._strip_prompt_echo(raw)
    assert "ENGINE:" not in cleaned
    assert "We need to" not in cleaned
    assert "System prompt says" not in cleaned
    assert "Hello! I am your AI Analyst." in cleaned
    assert "No dataset currently uploaded." in cleaned


def test_online_echoing_greeting_is_cleaned():
    from unittest.mock import patch
    state = SessionState()
    agent = DataAnalystAgent(
        session_state=state,
        llm_service=LLMService(LLMSettings(
            LLM_PROVIDER="nvidia",
            LLM_API_KEY="nvapi-test-key-12345",
        )),
    )

    def _echo_stream(prompt, system_prompt=None):
        yield from [
            "ENGINE: NVIDIA NIM\nWe need to greet briefly.\n",
            "Hello! Ready to analyze.",
        ]

    with patch.object(LLMService, "generate_stream", side_effect=_echo_stream):
        resp = agent.run("hello")
    assert "ENGINE:" not in resp.answer
    assert "We need to" not in resp.answer
    assert "Hello! Ready to analyze." in resp.answer

def test_strip_prompt_echo_removes_try_next_block():
    from src.agent.analyst import DataAnalystAgent
    raw = (
        "Hello! I am your AI Analyst.\n"
        "Current datasets: No dataset currently uploaded.\n"
        "\n"
        "Try next:\n"
        "\n"
        "    Upload one or more CSV files to start querying them\n"
        "    Ask for a top/bottom-k ranking once data is loaded\n"
        "    Request a chart or KPI dashboard\n"
    )
    cleaned = DataAnalystAgent._strip_prompt_echo(raw)
    assert "Try next" not in cleaned
    assert "Upload one or more CSV" not in cleaned
    assert "Hello! I am your AI Analyst." in cleaned
    assert "No dataset currently uploaded." in cleaned


def test_offline_greeting_has_no_try_next():
    state = SessionState()
    agent = DataAnalystAgent(
        session_state=state,
        llm_service=LLMService(LLMSettings(LLM_PROVIDER="mock")),
    )
    resp = agent.run("hello")
    assert "try next" not in resp.answer.lower()

BANNED_TERMS = ["digital back office", "nvidia", "nim", "muse", "glimmer",
                "llama", "mistral", "nemotron"]


def test_agent_answers_name_no_vendor_or_model():
    from src.agent.state import SessionState as _S
    llm = LLMService(LLMSettings(LLM_PROVIDER="mock"))
    for q in ["hello", "what can you do", "what are your special abilities?",
              "tell me about dbo", "thanks"]:
        state = _S()
        agent = DataAnalystAgent(session_state=state, llm_service=llm)
        ans = agent.run(q).answer.lower()
        for term in BANNED_TERMS:
            assert term not in ans, f"{term!r} leaked in answer to {q!r}"


def test_export_report_names_no_vendor():
    from src.agent.state import SessionState as _S
    from src.tools.export import generate_executive_html_report
    html = generate_executive_html_report(_S()).lower()
    for term in BANNED_TERMS:
        assert term not in html, f"{term!r} leaked in export report"