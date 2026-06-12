"""Local Hugging Face ``sentence-transformers`` embedding client."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class HuggingFaceEmbeddingClient:
    """Embed text locally with a Hugging Face ``sentence-transformers`` model.

    Real semantic embeddings with no API key and no per-call cost: the model
    runs in-process (downloaded from the Hugging Face Hub on first use, then
    cached). The default, ``all-MiniLM-L6-v2``, is small and fast enough to
    score an attribution sweep on CPU.

    The model is loaded lazily so constructing the scorer is free; pass
    ``model`` to use a different sentence-transformers checkpoint, or
    ``encoder`` to inject a preloaded model (or a stub in tests).
    """

    def __init__(self, model: str = _DEFAULT_MODEL, *, encoder: Any | None = None) -> None:
        self.model = model
        self._encoder = encoder

    def embed(self, text: str) -> Sequence[float]:
        """Return the embedding vector for ``text`` from the local model."""
        encoder = self._encoder or self._load_encoder()
        return [float(value) for value in encoder.encode(text)]

    def _load_encoder(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised without optional extra
            msg = "Install promptlens[huggingface] to use HuggingFaceEmbeddingClient"
            raise RuntimeError(msg) from exc
        self._encoder = SentenceTransformer(self.model)
        return self._encoder
