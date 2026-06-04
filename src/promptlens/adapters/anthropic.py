"""Anthropic adapter using the official SDK."""

from __future__ import annotations

from typing import Any

from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions


class AnthropicAdapter(Adapter):
    """Thin wrapper around Anthropic Messages API."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = client

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        client = self._client or _default_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if tools:
            kwargs["tools"] = tools
        response = client.messages.create(**kwargs)
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
