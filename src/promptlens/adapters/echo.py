"""Offline adapter for tests and examples."""

from __future__ import annotations

from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions, Usage


class EchoAdapter(Adapter):
    """Return the prompt as the model output without making network calls.

    Reports synthetic whitespace-token usage so cost-estimation flows that read
    measured baseline usage keep working offline.
    """

    def __init__(self, model: str = "echo") -> None:
        self.model = model

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        tokens = max(1, len(prompt.split()))
        return CompletionOutput(
            text=prompt, usage=Usage(input_tokens=tokens, output_tokens=tokens)
        )
