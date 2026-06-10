"""Anthropic adapter using the official SDK."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from promptlens.adapters.models import supports_temperature
from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions, coerce_tools


class AnthropicAdapter(Adapter):
    """Thin wrapper around Anthropic Messages API.

    Set ``use_batch_api=True`` to route :meth:`complete_batch` through the
    Anthropic Message Batches API (50% cheaper, asynchronous with polling).
    The synchronous :meth:`complete` path is unchanged.

    ``temperature`` is omitted from requests for models that removed sampling
    parameters (Claude Opus 4.7 and later), where sending it returns a 400.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        client: Any | None = None,
        use_batch_api: bool = False,
        poll_interval_seconds: float = 5.0,
        max_concurrency: int = 1,
    ) -> None:
        if poll_interval_seconds <= 0:
            msg = f"poll_interval_seconds must be > 0, got {poll_interval_seconds}"
            raise ValueError(msg)
        if max_concurrency < 1:
            msg = f"max_concurrency must be >= 1, got {max_concurrency}"
            raise ValueError(msg)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = client
        self.use_batch_api = use_batch_api
        self.poll_interval_seconds = poll_interval_seconds
        self.max_concurrency = max_concurrency

    def _request_params(self, prompt: str, tools: ToolDefinitions | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if supports_temperature(self.model):
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = coerce_tools(tools, "anthropic")
        return kwargs

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        client = self._client or _default_client()
        response = client.messages.create(**self._request_params(prompt, tools))
        return _message_to_output(response)

    def count_tokens(self, prompt: str, tools: ToolDefinitions | None = None) -> int:
        """Count input tokens exactly via the Messages API count_tokens endpoint.

        This is a free metering call — it runs no inference and bills no tokens —
        but it does hit the network, so the harness only uses it when an exact
        estimate is explicitly requested.
        """
        client = self._client or _default_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if tools:
            kwargs["tools"] = coerce_tools(tools, "anthropic")
        return int(client.messages.count_tokens(**kwargs).input_tokens)

    def complete_batch(
        self, prompts: Sequence[str], tools: ToolDefinitions | None = None
    ) -> list[CompletionOutput]:
        if not self.use_batch_api or len(prompts) <= 1:
            return super().complete_batch(prompts, tools=tools)
        client = self._client or _default_client()
        requests = [
            {"custom_id": _custom_id(index), "params": self._request_params(prompt, tools)}
            for index, prompt in enumerate(prompts)
        ]
        batch = client.messages.batches.create(requests=requests)
        self._await_batch(client, batch.id)
        outputs_by_id: dict[str, CompletionOutput] = {}
        for entry in client.messages.batches.results(batch.id):
            result = entry.result
            if getattr(result, "type", None) != "succeeded":
                msg = f"Anthropic batch request {entry.custom_id} ended as {result.type}"
                raise RuntimeError(msg)
            outputs_by_id[entry.custom_id] = _message_to_output(result.message)
        return [outputs_by_id[_custom_id(index)] for index in range(len(prompts))]

    def _await_batch(self, client: Any, batch_id: str) -> None:
        while True:
            current = client.messages.batches.retrieve(batch_id)
            if current.processing_status == "ended":
                return
            time.sleep(self.poll_interval_seconds)


def _custom_id(index: int) -> str:
    return f"req-{index}"


def _message_to_output(response: Any) -> CompletionOutput:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in getattr(response, "content", []):
        block_type = getattr(block, "type", "")
        if block_type == "text":
            text_parts.append(str(getattr(block, "text", "")))
        if block_type == "tool_use":
            tool_calls.append(
                {
                    "id": getattr(block, "id", None),
                    "name": getattr(block, "name", None),
                    "arguments": getattr(block, "input", {}),
                }
            )
    return CompletionOutput(text="".join(text_parts), tool_calls=tool_calls, raw=response)


def _default_client() -> Any:
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover
        msg = "Install promptlens[anthropic] to use AnthropicAdapter"
        raise RuntimeError(msg) from exc
    return Anthropic()
