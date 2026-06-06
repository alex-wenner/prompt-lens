"""Generic adapter for any OpenAI-compatible provider endpoint."""

from __future__ import annotations

from typing import Any

from promptlens.adapters.openai import OpenAIAdapter


class OpenAICompatibleAdapter(OpenAIAdapter):
    """Talk to any provider that exposes an OpenAI-compatible Chat Completions API.

    This is the generic escape hatch for providers promptlens does not ship a
    dedicated adapter for. Point ``base_url`` at the provider gateway and it works
    with xAI Grok (``https://api.x.ai/v1``), Google Gemini's OpenAI-compatibility
    layer (``https://generativelanguage.googleapis.com/v1beta/openai/``), local
    servers such as Ollama or vLLM, and any other OpenAI-compatible endpoint.
    GitHub Copilot has its own dedicated :class:`~promptlens.adapters.CopilotAdapter`
    backed by the official Copilot SDK.

    ``logprobs`` defaults to off because most compatibility layers do not return
    token log probabilities; enable it only for endpoints/models that do.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "not-needed",
        temperature: float = 0.0,
        logprobs: bool = False,
        client: Any | None = None,
    ) -> None:
        if client is None:
            client = _compatible_client(base_url=base_url, api_key=api_key)
        super().__init__(
            model=model, temperature=temperature, logprobs=logprobs, client=client
        )
        self.base_url = base_url


def _compatible_client(base_url: str, api_key: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        msg = "Install promptlens[openai] to use OpenAICompatibleAdapter"
        raise RuntimeError(msg) from exc
    return OpenAI(base_url=base_url, api_key=api_key)
