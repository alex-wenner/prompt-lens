"""OpenAI adapter using the official SDK."""

from __future__ import annotations

from typing import Any

from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions


class OpenAIAdapter(Adapter):
    """Thin wrapper around OpenAI chat completions."""

    def __init__(self, model: str, temperature: float = 0.0, client: Any | None = None) -> None:
        self.model = model
        self.temperature = temperature
        self._client = client

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        client = self._client or _default_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            temperature=self.temperature,
        )
        choice = response.choices[0]
        message = choice.message
        text = message.content or ""
        tool_calls = [_openai_tool_call_to_dict(call) for call in (message.tool_calls or [])]
        logprobs = _extract_logprobs(choice)
        return CompletionOutput(text=text, tool_calls=tool_calls, logprobs=logprobs, raw=response)


def _default_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        msg = "Install promptlens[openai] to use OpenAIAdapter"
        raise RuntimeError(msg) from exc
    return OpenAI()


def _openai_tool_call_to_dict(call: Any) -> dict[str, Any]:
    function = getattr(call, "function", None)
    return {
        "id": getattr(call, "id", None),
        "name": getattr(function, "name", None),
        "arguments": getattr(function, "arguments", None),
    }


def _extract_logprobs(choice: Any) -> list[float] | None:
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None) if logprobs is not None else None
    if not content:
        return None
    return [float(item.logprob) for item in content]
