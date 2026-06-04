"""Offline adapter for tests and examples."""

from __future__ import annotations

from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions


class EchoAdapter(Adapter):
    """Return the prompt as the model output without making network calls."""

    def __init__(self, model: str = "echo") -> None:
        self.model = model

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        return CompletionOutput(text=prompt)
