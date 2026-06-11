"""Deterministic offline embedding fallback for smoke runs and tests."""

from __future__ import annotations

from collections.abc import Sequence


class TextShapeEmbeddingClient:
    """Deterministic local embedding fallback for offline smoke runs.

    This is **not** a semantic embedding: it derives a few cheap text-shape
    features without contacting a provider, so it is only useful for smoke tests
    and demos. Select it explicitly via the ``embedding-local`` scorer name. For
    real attribution use the ``embedding`` scorer with a provider config.
    """

    def embed(self, text: str) -> Sequence[float]:
        """Return simple text-shape features without making provider calls."""
        char_sum = sum(ord(char) for char in text)
        return (
            float(len(text)),
            float(text.count(" ")),
            float(char_sum % 997),
        )
