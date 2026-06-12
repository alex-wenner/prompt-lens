"""OpenAI (and OpenAI-compatible) embedding client."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class OpenAIEmbeddingClient:
    """Embed text with the OpenAI (or OpenAI-compatible) embeddings API.

    The client is created lazily so constructing the scorer never makes a network
    call; the provider is only contacted the first time :meth:`embed` runs. Pass
    ``client`` to inject a stub in tests, or ``base_url`` to target an
    OpenAI-compatible embeddings endpoint.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        client: Any | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._client = client

    def embed(self, text: str) -> Sequence[float]:
        """Return the embedding vector for ``text`` from the provider."""
        client = self._client or _default_client(self.base_url)
        response = client.embeddings.create(model=self.model, input=text)
        return [float(value) for value in response.data[0].embedding]


def _default_client(base_url: str | None) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        msg = "Install promptlens[openai] to use OpenAIEmbeddingClient"
        raise RuntimeError(msg) from exc
    if base_url:
        return OpenAI(base_url=base_url)
    return OpenAI()
