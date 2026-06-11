"""Hugging Face ``sentence-transformers`` embeddings for the semantic drift scorer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class HuggingFaceEmbeddingClient:
    """Embed text locally with a Hugging Face ``sentence-transformers`` model.

    Runs entirely on your machine: no API key and no network calls after the
    model weights are downloaded once. The model is loaded lazily on the first
    :meth:`embed` call so constructing the scorer stays instant. Pass ``model``
    to choose any sentence-transformers checkpoint, or ``client`` to inject a
    preloaded model (or a stub in tests).
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._client = client

    def embed(self, text: str) -> Sequence[float]:
        """Return the embedding vector for ``text`` from the local model."""
        client = self._client or _default_client(self.model)
        self._client = client
        vector = client.encode(text)
        return [float(value) for value in vector]


def _default_client(model: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        msg = "Install promptlens[huggingface] to use HuggingFaceEmbeddingClient"
        raise RuntimeError(msg) from exc
    return SentenceTransformer(model)
