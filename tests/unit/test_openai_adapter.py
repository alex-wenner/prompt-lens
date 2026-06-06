import json
from types import SimpleNamespace
from typing import Any

import pytest

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


def test_logprobs_rejected_for_unsupported_model() -> None:
    client = _FakeClient()
    with pytest.raises(ValueError, match="does not support logprobs"):
        OpenAIAdapter(model="gpt-5.5", logprobs=True, client=client)


def test_unsupported_model_allowed_without_logprobs() -> None:
    client = _FakeClient()
    adapter = OpenAIAdapter(model="gpt-5.5", client=client)
    adapter.complete("hello")
    assert "logprobs" not in client.chat.completions.calls[0]


class _FakeFiles:
    def __init__(self, output_jsonl: str) -> None:
        self.output_jsonl = output_jsonl
        self.created: list[Any] = []

    def create(self, *, file: Any, purpose: str) -> Any:
        self.created.append({"purpose": purpose, "payload": file.read().decode("utf-8")})
        return SimpleNamespace(id="file-in")

    def content(self, file_id: str) -> Any:
        assert file_id == "file-out"
        return SimpleNamespace(text=self.output_jsonl)


class _FakeBatches:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.created.append(kwargs)
        return SimpleNamespace(id="batch-1")

    def retrieve(self, batch_id: str) -> Any:
        return SimpleNamespace(id=batch_id, status="completed", output_file_id="file-out")


class _FakeBatchClient:
    def __init__(self, output_jsonl: str) -> None:
        self.files = _FakeFiles(output_jsonl)
        self.batches = _FakeBatches()
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def _batch_line(custom_id: str, content: str) -> str:
    return json.dumps(
        {
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "body": {"choices": [{"message": {"content": content}}]},
            },
        }
    )


def test_batch_api_uploads_polls_and_orders_results() -> None:
    output = "\n".join(
        [_batch_line("req-1", "second"), _batch_line("req-0", "first")]
    )
    client = _FakeBatchClient(output)
    adapter = OpenAIAdapter(model="gpt-4o", client=client, use_batch_api=True)

    outputs = adapter.complete_batch(["a", "b"])

    assert [out.text for out in outputs] == ["first", "second"]
    assert client.files.created[0]["purpose"] == "batch"
    assert client.batches.created[0]["endpoint"] == "/v1/chat/completions"


def test_batch_api_skipped_for_single_prompt() -> None:
    client = _FakeBatchClient("")
    adapter = OpenAIAdapter(model="gpt-4o", client=client, use_batch_api=True)

    outputs = adapter.complete_batch(["only"])

    assert [out.text for out in outputs] == ["hi"]
    assert client.batches.created == []

