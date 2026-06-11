"""Amazon Bedrock adapter."""

from __future__ import annotations

from typing import Any

from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions, Usage, coerce_tools


class BedrockAdapter(Adapter):
    """Thin wrapper around Amazon Bedrock Runtime Converse API."""

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
        request: dict[str, Any] = {
            "modelId": self.model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"temperature": self.temperature, "maxTokens": self.max_tokens},
        }
        if tools:
            request["toolConfig"] = {"tools": coerce_tools(tools, "bedrock")}
        response = client.converse(**request)
        output_message = response.get("output", {}).get("message", {})
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in output_message.get("content", []):
            if "text" in block:
                text_parts.append(str(block["text"]))
            if "toolUse" in block:
                tool_use = block["toolUse"]
                tool_calls.append(
                    {
                        "id": tool_use.get("toolUseId"),
                        "name": tool_use.get("name"),
                        "arguments": tool_use.get("input", {}),
                    }
                )
        usage_data = response.get("usage") or {}
        usage = (
            Usage(
                input_tokens=int(usage_data.get("inputTokens", 0)),
                output_tokens=int(usage_data.get("outputTokens", 0)),
            )
            if usage_data
            else None
        )
        return CompletionOutput(
            text="".join(text_parts), tool_calls=tool_calls, usage=usage, raw=response
        )


def _default_client() -> Any:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover
        msg = "Install promptlens[bedrock] to use BedrockAdapter"
        raise RuntimeError(msg) from exc
    return boto3.client("bedrock-runtime")
