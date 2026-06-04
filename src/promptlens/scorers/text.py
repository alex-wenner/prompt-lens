"""General-purpose output scorers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from promptlens.core.base import CompletionOutput, Scorer


class EmbeddingClient(Protocol):
    def embed(self, text: str) -> Sequence[float]:
        """Return an embedding vector for text."""


class EmbeddingScorer(Scorer):
    """Score output drift as cosine distance between embeddings."""

    def __init__(self, embedding_client: EmbeddingClient) -> None:
        self.embedding_client = embedding_client

    def score(self, baseline: CompletionOutput, candidate: CompletionOutput) -> float:
        return cosine_distance(
            self.embedding_client.embed(baseline.text), self.embedding_client.embed(candidate.text)
        )


class LengthDriftScorer(Scorer):
    """Offline scorer useful for tests and smoke runs."""

    def score(self, baseline: CompletionOutput, candidate: CompletionOutput) -> float:
        baseline_length = max(1, len(baseline.text))
        return min(1.0, abs(len(baseline.text) - len(candidate.text)) / baseline_length)


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        msg = "Embedding vectors must have the same length"
        raise ValueError(msg)
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 1.0
    similarity = dot / (left_norm * right_norm)
    return 1.0 - max(-1.0, min(1.0, similarity))
