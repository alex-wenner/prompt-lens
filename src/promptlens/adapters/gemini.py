"""Google Gemini adapter using the official ``google-genai`` SDK.

Gemini is reached through Google's own Python SDK rather than the generic
OpenAI-compatibility layer. The SDK exposes ``client.models.generate_content``,
which returns the assistant text plus any function calls the model made. This
adapter wraps that flow behind the synchronous :class:`Adapter` interface.
"""

from __future__ import annotations

from typing import Any

from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions, Usage, coerce_tools


class GeminiAdapter(Adapter):
    """Thin wrapper around the official Google ``google-genai`` SDK.

    Each :meth:`complete` call is a fresh, stateless ``generate_content`` request
    so attribution coalitions never share conversation memory.
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
        config: dict[str, Any] = {"temperature": self.temperature}
        if tools:
            config["tools"] = coerce_tools(tools, "gemini")
        response = client.models.generate_content(
            model=self.model, contents=prompt, config=config
        )
        return _response_to_output(response)


def _default_client(api_key: str | None) -> Any:
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        msg = "Install promptlens[gemini] to use GeminiAdapter"
        raise RuntimeError(msg) from exc
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()


def _response_to_output(response: Any) -> CompletionOutput:
    text = getattr(response, "text", "") or ""
    tool_calls = [
        {
            "id": getattr(call, "id", None),
            "name": getattr(call, "name", None),
            "arguments": getattr(call, "args", None),
        }
        for call in (getattr(response, "function_calls", None) or [])
    ]
    metadata = getattr(response, "usage_metadata", None)
    input_tokens = getattr(metadata, "prompt_token_count", None) if metadata else None
    output_tokens = getattr(metadata, "candidates_token_count", None) if metadata else None
    usage = (
        Usage(input_tokens=int(input_tokens), output_tokens=int(output_tokens))
        if input_tokens is not None and output_tokens is not None
        else None
    )
    return CompletionOutput(text=str(text), tool_calls=tool_calls, usage=usage, raw=response)
