from types import SimpleNamespace
from typing import Any

from promptlens.adapters import GeminiAdapter


class _FakeModels:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._response


class _FakeClient:
    def __init__(self, response: Any) -> None:
        self.models = _FakeModels(response)


def test_complete_extracts_text_and_uses_model() -> None:
    client = _FakeClient(SimpleNamespace(text="hello from gemini", function_calls=None))
    adapter = GeminiAdapter(model="gemini-3.5-flash", temperature=0.4, client=client)

    output = adapter.complete("hi")

    assert output.text == "hello from gemini"
    assert output.tool_calls == []
    assert client.models.calls[0]["model"] == "gemini-3.5-flash"
    assert client.models.calls[0]["contents"] == "hi"
    assert client.models.calls[0]["config"]["temperature"] == 0.4


def test_complete_maps_function_calls() -> None:
    function_call = SimpleNamespace(
        id="call-1", name="get_weather", args={"city": "Seattle"}
    )
    client = _FakeClient(SimpleNamespace(text="", function_calls=[function_call]))
    adapter = GeminiAdapter(model="gemini-3.5-flash", client=client)

    output = adapter.complete("weather?")

    assert output.tool_calls == [
        {"id": "call-1", "name": "get_weather", "arguments": {"city": "Seattle"}}
    ]


def test_complete_forwards_tools() -> None:
    client = _FakeClient(SimpleNamespace(text="ok", function_calls=None))
    adapter = GeminiAdapter(model="gemini-3.5-flash", client=client)
    tools = [{"name": "search"}]

    adapter.complete("hi", tools=tools)

    assert client.models.calls[0]["config"]["tools"] == tools


def test_complete_handles_missing_text() -> None:
    client = _FakeClient(SimpleNamespace(text=None, function_calls=None))
    adapter = GeminiAdapter(model="gemini-3.5-flash", client=client)

    output = adapter.complete("hi")

    assert output.text == ""
    assert output.tool_calls == []
