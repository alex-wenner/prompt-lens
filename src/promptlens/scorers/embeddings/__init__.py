"""Embedding clients for the semantic drift scorer, one provider per module.

:class:`~promptlens.scorers.text.EmbeddingScorer` accepts any object matching
the :class:`~promptlens.scorers.text.EmbeddingClient` protocol. Each module in
this package supplies one real provider:

* :mod:`promptlens.scorers.embeddings.openai` — hosted OpenAI (or any
  OpenAI-compatible) embeddings API.
* :mod:`promptlens.scorers.embeddings.huggingface` — local Hugging Face
  ``sentence-transformers`` models; semantic, free, and offline once the model
  is downloaded.

New providers get their own module here rather than growing an existing one.
"""

from promptlens.scorers.embeddings.huggingface import HuggingFaceEmbeddingClient
from promptlens.scorers.embeddings.openai import OpenAIEmbeddingClient

__all__ = [
    "HuggingFaceEmbeddingClient",
    "OpenAIEmbeddingClient",
]
