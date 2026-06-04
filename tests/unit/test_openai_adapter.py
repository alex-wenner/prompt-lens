from types import SimpleNamespace
from typing import Any

from promptlens.adapters import OpenAIAdapter


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        token = SimpleNamespace(logprob=-0.5)
        logprobs = SimpleNamespace(content=[token])
        message = SimpleNamespace(content="hi", tool_calls=None)
        choice = SimpleNamespace(message=message, logprobs=logprobs)
        return SimpleNamespace(choices=[choice])


class _FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_logprobs_requested_only_when_enabled() -> None:
    client = _FakeClient()
    OpenAIAdapter(model="gpt-4o", client=client).complete("hello")
    assert "logprobs" not in client.chat.completions.calls[0]

    client = _FakeClient()
    output = OpenAIAdapter(model="gpt-4o", logprobs=True, client=client).complete("hello")
    assert client.chat.completions.calls[0]["logprobs"] is True
    assert output.logprobs == [-0.5]


def test_tools_omitted_when_none() -> None:
    client = _FakeClient()
    OpenAIAdapter(model="gpt-4o", client=client).complete("hello")
    assert "tools" not in client.chat.completions.calls[0]
