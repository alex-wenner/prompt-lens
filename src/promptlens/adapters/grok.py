"""xAI Grok adapter using the official ``xai-sdk``.

Grok is reached through xAI's own Python SDK rather than the generic
OpenAI-compatible HTTP path. The SDK exposes a stateful chat object: a chat is
created for the model, the user turn is appended, and ``sample()`` returns the
assistant response. This adapter wraps that flow behind the synchronous
:class:`Adapter` interface.
"""

from __future__ import annotations

from typing import Any

from promptlens.core.base import (
    Adapter,
    CompletionOutput,
    TokenUsage,
    ToolDefinitions,
    coerce_tools,
)


class GrokAdapter(Adapter):
    """Thin wrapper around the official xAI ``xai-sdk`` chat API.

    Each :meth:`complete` call creates a fresh, stateless chat so attribution
    coalitions never share conversation memory.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self._client = client

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        client = self._client or _default_client(self.api_key)
        self._client = client
        kwargs: dict[str, Any] = {"model": self.model, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = coerce_tools(tools, "grok")
        chat = client.chat.create(**kwargs)
        chat.append(_user_message(prompt))
        response = chat.sample()
        return _response_to_output(response)


def _default_client(api_key: str | None) -> Any:
    try:
        from xai_sdk import Client
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        msg = "Install promptlens[grok] to use GrokAdapter"
        raise RuntimeError(msg) from exc
    if api_key:
        return Client(api_key=api_key)
    return Client()


def _user_message(prompt: str) -> Any:
    """Build a user turn using the SDK helper, falling back to a plain mapping."""
    try:
        from xai_sdk.chat import user
    except ImportError:  # pragma: no cover - tests inject a fake client
        return {"role": "user", "content": prompt}
    return user(prompt)


def _response_to_output(response: Any) -> CompletionOutput:
    text = getattr(response, "content", "") or ""
    tool_calls = [
        _tool_call_to_dict(call) for call in (getattr(response, "tool_calls", None) or [])
    ]
    return CompletionOutput(
        text=str(text),
        tool_calls=tool_calls,
        usage=_extract_usage(getattr(response, "usage", None)),
        raw=response,
    )


def _extract_usage(usage: Any) -> TokenUsage | None:
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if prompt_tokens is None or completion_tokens is None:
        return None
    return TokenUsage(input_tokens=int(prompt_tokens), output_tokens=int(completion_tokens))


def _tool_call_to_dict(call: Any) -> dict[str, Any]:
    function = getattr(call, "function", None)
    return {
        "id": getattr(call, "id", None),
        "name": getattr(function, "name", None),
        "arguments": getattr(function, "arguments", None),
    }
