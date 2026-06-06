from types import SimpleNamespace
from typing import Any

from promptlens.adapters import GrokAdapter


class _FakeChat:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.appended: list[Any] = []

    def append(self, message: Any) -> None:
        self.appended.append(message)

    def sample(self) -> Any:
        return self._response


class _FakeChatNamespace:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.created: list[dict[str, Any]] = []
        self.chats: list[_FakeChat] = []

    def create(self, **kwargs: Any) -> _FakeChat:
        self.created.append(kwargs)
        chat = _FakeChat(self._response)
        self.chats.append(chat)
        return chat


class _FakeClient:
    def __init__(self, response: Any) -> None:
        self.chat = _FakeChatNamespace(response)


def test_complete_extracts_text_and_uses_model() -> None:
    client = _FakeClient(SimpleNamespace(content="hello from grok", tool_calls=None))
    adapter = GrokAdapter(model="grok-4", temperature=0.2, client=client)

    output = adapter.complete("hi")

    assert output.text == "hello from grok"
    assert output.tool_calls == []
    assert client.chat.created[0]["model"] == "grok-4"
    assert client.chat.created[0]["temperature"] == 0.2
    assert client.chat.chats[0].appended  # the user turn was appended


def test_complete_maps_tool_calls() -> None:
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="get_weather", arguments='{"city": "Seattle"}'),
    )
    client = _FakeClient(SimpleNamespace(content="", tool_calls=[tool_call]))
    adapter = GrokAdapter(model="grok-4", client=client)

    output = adapter.complete("weather?")

    assert output.tool_calls == [
        {"id": "call-1", "name": "get_weather", "arguments": '{"city": "Seattle"}'}
    ]


def test_complete_forwards_tools() -> None:
    client = _FakeClient(SimpleNamespace(content="ok", tool_calls=None))
    adapter = GrokAdapter(model="grok-4", client=client)
    tools = [{"name": "search"}]

    adapter.complete("hi", tools=tools)

    assert client.chat.created[0]["tools"] == tools


def test_complete_handles_missing_content() -> None:
    client = _FakeClient(SimpleNamespace(content=None, tool_calls=None))
    adapter = GrokAdapter(model="grok-4", client=client)

    output = adapter.complete("hi")

    assert output.text == ""
    assert output.tool_calls == []
