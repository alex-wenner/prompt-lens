"""OpenAI-compatible adapter for local and open-weight model endpoints."""

from __future__ import annotations

from typing import Any

from promptlens.adapters.openai import OpenAIAdapter


class OpenAICompatibleAdapter(OpenAIAdapter):
    """Use any OpenAI-compatible endpoint, including local open-source model servers."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "not-needed",
        temperature: float = 0.0,
        client: Any | None = None,
    ) -> None:
        if client is None:
            client = _compatible_client(base_url=base_url, api_key=api_key)
        super().__init__(model=model, temperature=temperature, client=client)
        self.base_url = base_url


def _compatible_client(base_url: str, api_key: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        msg = "Install promptlens[openai] to use OpenAICompatibleAdapter"
        raise RuntimeError(msg) from exc
    return OpenAI(base_url=base_url, api_key=api_key)
