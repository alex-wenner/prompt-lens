"""Embedding clients for the semantic drift scorer.

:class:`~promptlens.scorers.text.EmbeddingScorer` accepts any object matching the
:class:`~promptlens.scorers.text.EmbeddingClient` protocol. This module supplies two
real clients:

* :class:`OpenAIEmbeddingClient` — calls the OpenAI (or compatible) embeddings API.
* :class:`HuggingFaceEmbeddingClient` — runs a ``sentence-transformers`` model locally,
  no API key required. Install with ``pip install sentence-transformers``.
"""

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


class HuggingFaceEmbeddingClient:
    """Embed text locally with a ``sentence-transformers`` model.

    No API key required. The model is downloaded once and cached by
    ``sentence-transformers`` on first use.

    Install the dependency with::

        pip install sentence-transformers

    Or use the bundled extra::

        pip install -e '.[hf]'

    Parameters
    ----------
    model:
        Any model name accepted by ``sentence-transformers``, e.g.
        ``"all-MiniLM-L6-v2"`` (fast, 384-dim) or
        ``"Qwen/Qwen3-Embedding"`` (higher quality, requires more RAM).
        Defaults to ``"all-MiniLM-L6-v2"``.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self.model = model
        self._encoder: Any | None = None

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                msg = (
                    "Install sentence-transformers to use HuggingFaceEmbeddingClient: "
                    "pip install sentence-transformers"
                )
                raise RuntimeError(msg) from exc
            self._encoder = SentenceTransformer(self.model)
        return self._encoder

    def embed(self, text: str) -> Sequence[float]:
        """Return the embedding vector for ``text`` using the local model."""
        encoder = self._get_encoder()
        vector = encoder.encode(text, convert_to_numpy=True)
        return [float(v) for v in vector]
