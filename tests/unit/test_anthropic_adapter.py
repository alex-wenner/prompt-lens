from types import SimpleNamespace
from typing import Any

import pytest

from promptlens.adapters import AnthropicAdapter


class _FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        block = SimpleNamespace(type="text", text="sync")
        return SimpleNamespace(content=[block])


def _message(text: str) -> Any:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class _FakeBatches:
    def __init__(self) -> None:
        self.created: list[Any] = []

    def create(self, *, requests: Any) -> Any:
        self.created.append(requests)
        return SimpleNamespace(id="batch-1")

    def retrieve(self, batch_id: str) -> Any:
        return SimpleNamespace(id=batch_id, processing_status="ended")

    def results(self, batch_id: str) -> Any:
        # Deliberately out of order to verify custom_id reordering.
        return [
            SimpleNamespace(
                custom_id="req-1",
                result=SimpleNamespace(type="succeeded", message=_message("second")),
            ),
            SimpleNamespace(
                custom_id="req-0",
                result=SimpleNamespace(type="succeeded", message=_message("first")),
            ),
        ]


class _FakeClient:
    def __init__(self) -> None:
        self.messages = SimpleNamespace(create=_FakeMessages().create, batches=_FakeBatches())


def test_complete_uses_messages_create() -> None:
    client = _FakeClient()
    output = AnthropicAdapter(model="claude", client=client).complete("hello")
    assert output.text == "sync"


def test_batch_api_orders_results_by_custom_id() -> None:
    client = _FakeClient()
    adapter = AnthropicAdapter(model="claude", client=client, use_batch_api=True)

    outputs = adapter.complete_batch(["a", "b"])

    assert [out.text for out in outputs] == ["first", "second"]
    assert len(client.messages.batches.created[0]) == 2


def test_batch_api_skipped_for_single_prompt() -> None:
    client = _FakeClient()
    adapter = AnthropicAdapter(model="claude", client=client, use_batch_api=True)

    outputs = adapter.complete_batch(["only"])

    assert [out.text for out in outputs] == ["sync"]
    assert client.messages.batches.created == []


def test_batch_request_raises_on_failure() -> None:
    class _FailingBatches(_FakeBatches):
        def results(self, batch_id: str) -> Any:
            return [
                SimpleNamespace(
                    custom_id="req-0",
                    result=SimpleNamespace(type="errored", message=None),
                ),
                SimpleNamespace(
                    custom_id="req-1",
                    result=SimpleNamespace(type="succeeded", message=_message("ok")),
                ),
            ]

    client = _FakeClient()
    client.messages = SimpleNamespace(
        create=_FakeMessages().create, batches=_FailingBatches()
    )
    adapter = AnthropicAdapter(model="claude", client=client, use_batch_api=True)

    with pytest.raises(RuntimeError, match="errored"):
        adapter.complete_batch(["a", "b"])


def test_invalid_poll_interval_rejected() -> None:
    with pytest.raises(ValueError, match="poll_interval_seconds"):
        AnthropicAdapter(model="claude", poll_interval_seconds=0)


def test_temperature_sent_for_models_that_accept_it() -> None:
    messages = _FakeMessages()
    client = _FakeClient()
    client.messages = SimpleNamespace(create=messages.create, batches=_FakeBatches())
    AnthropicAdapter(model="claude-sonnet-4-6", temperature=0.3, client=client).complete("hello")

    assert messages.calls[0]["temperature"] == 0.3


def test_temperature_omitted_for_opus_4_7_plus() -> None:
    # Claude Opus 4.7+ removed sampling parameters; sending temperature is a 400.
    messages = _FakeMessages()
    client = _FakeClient()
    client.messages = SimpleNamespace(create=messages.create, batches=_FakeBatches())
    AnthropicAdapter(model="claude-opus-4-8", temperature=0.3, client=client).complete("hello")

    assert "temperature" not in messages.calls[0]
