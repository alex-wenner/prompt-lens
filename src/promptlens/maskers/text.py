"""Text masking strategies."""

from __future__ import annotations

from collections.abc import Sequence

from promptlens.core.base import Coalition, Feature, Masker, normalize_coalition


class PlaceholderMasker(Masker):
    """Replace absent features with a stable placeholder while preserving order."""

    def __init__(self, placeholder: str = "[...]", separator: str = " ") -> None:
        self.placeholder = placeholder
        self.separator = separator

    def mask(self, features: Sequence[Feature], coalition: Coalition) -> str:
        normalized = normalize_coalition(coalition, len(features))
        parts = [feature.text if keep else self.placeholder for feature, keep in zip(features, normalized)]
        return self.separator.join(parts)
