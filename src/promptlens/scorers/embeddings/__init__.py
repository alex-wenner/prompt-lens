"""Embedding clients for the semantic drift scorer, one provider per module.

:class:`~promptlens.scorers.text.EmbeddingScorer` accepts any object matching
the :class:`~promptlens.scorers.text.EmbeddingClient` protocol. Each module in
this package supplies one provider-backed client:

* :mod:`promptlens.scorers.embeddings.openai` — OpenAI (and OpenAI-compatible)
  embeddings API.
* :mod:`promptlens.scorers.embeddings.huggingface` — local Hugging Face
  ``sentence-transformers`` models; no API key, no network after download.
* :mod:`promptlens.scorers.embeddings.local` — deterministic text-shape
  features for offline smoke runs and tests (not semantic).
"""

from promptlens.scorers.embeddings.huggingface import HuggingFaceEmbeddingClient
from promptlens.scorers.embeddings.local import TextShapeEmbeddingClient
from promptlens.scorers.embeddings.openai import OpenAIEmbeddingClient

__all__ = [
    "HuggingFaceEmbeddingClient",
    "OpenAIEmbeddingClient",
    "TextShapeEmbeddingClient",
]
