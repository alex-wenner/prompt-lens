from types import SimpleNamespace
from typing import Any

import pytest

from promptlens.adapters import CopilotAdapter


class _FakeSession:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.prompts: list[str] = []
        self.disconnected = False

    async def send_and_wait(self, prompt: str, *, timeout: float) -> Any:
        self.prompts.append(prompt)
        return self._response

    async def disconnect(self) -> None:
        self.disconnected = True


class _FakeClient:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.created_sessions: list[dict[str, Any]] = []
        self.sessions: list[_FakeSession] = []
        self.stopped = False

    async def create_session(self, **kwargs: Any) -> _FakeSession:
        self.created_sessions.append(kwargs)
        session = _FakeSession(self._response)
        self.sessions.append(session)
        return session

    async def stop(self) -> None:
        self.stopped = True


def _assistant_event(content: str, tool_requests: list[Any] | None = None) -> Any:
    data = SimpleNamespace(content=content, tool_requests=tool_requests)
    return SimpleNamespace(data=data)


def test_complete_extracts_text_and_uses_model() -> None:
    client = _FakeClient(_assistant_event("hello from copilot"))
    adapter = CopilotAdapter(model="gpt-4.1", client=client)
    try:
        output = adapter.complete("hi")
    finally:
        adapter.close()

    assert output.text == "hello from copilot"
    assert output.tool_calls == []
    assert client.created_sessions[0]["model"] == "gpt-4.1"
    assert client.sessions[0].prompts == ["hi"]
    assert client.sessions[0].disconnected is True


def test_complete_maps_tool_requests() -> None:
    tool_request = SimpleNamespace(
        tool_call_id="call-1", name="get_weather", arguments={"city": "Seattle"}
    )
    client = _FakeClient(_assistant_event("", [tool_request]))
    adapter = CopilotAdapter(model="gpt-4.1", client=client)
    try:
        output = adapter.complete("weather?")
    finally:
        adapter.close()

    assert output.tool_calls == [
        {"id": "call-1", "name": "get_weather", "arguments": {"city": "Seattle"}}
    ]


def test_complete_handles_missing_response() -> None:
    client = _FakeClient(None)
    adapter = CopilotAdapter(model="gpt-4.1", client=client)
    try:
        output = adapter.complete("hi")
    finally:
        adapter.close()

    assert output.text == ""
    assert output.tool_calls == []


def test_each_complete_uses_a_fresh_session() -> None:
    client = _FakeClient(_assistant_event("ok"))
    adapter = CopilotAdapter(model="gpt-4.1", client=client)
    try:
        adapter.complete("first")
        adapter.complete("second")
    finally:
        adapter.close()

    assert len(client.sessions) == 2
    assert client.sessions[0].prompts == ["first"]
    assert client.sessions[1].prompts == ["second"]


def test_injected_client_is_not_stopped_on_close() -> None:
    client = _FakeClient(_assistant_event("ok"))
    adapter = CopilotAdapter(model="gpt-4.1", client=client)
    adapter.complete("hi")
    adapter.close()

    assert client.stopped is False


def test_invalid_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="timeout must be > 0"):
        CopilotAdapter(model="gpt-4.1", timeout=0.0)
