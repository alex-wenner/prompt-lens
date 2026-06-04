"""Text masking strategies."""

from __future__ import annotations

from collections.abc import Sequence

from promptlens.core.base import Coalition, Feature, Masker, normalize_coalition


class PlaceholderMasker(Masker):
    """Replace absent features with a stable placeholder while preserving order.

    Keeps prompt structure mostly intact, so attribution measures the effect of
    hiding a feature's *content* while signalling that something was there.
    """

    def __init__(self, placeholder: str = "[...]", separator: str = " ") -> None:
        self.placeholder = placeholder
        self.separator = separator

    def mask(self, features: Sequence[Feature], coalition: Coalition) -> str:
        normalized = normalize_coalition(coalition, len(features))
        parts = [
            feature.text if keep else self.placeholder
            for feature, keep in zip(features, normalized, strict=True)
        ]
        return self.separator.join(parts)


class DropMasker(Masker):
    """Omit absent features entirely, collapsing their separators.

    Attribution measures the effect of removing a feature outright, with no
    placeholder hint left behind. Useful when the presence of a placeholder
    would itself perturb the model.
    """

    def __init__(self, separator: str = " ") -> None:
        self.separator = separator

    def mask(self, features: Sequence[Feature], coalition: Coalition) -> str:
        normalized = normalize_coalition(coalition, len(features))
        kept = [
            feature.text
            for feature, keep in zip(features, normalized, strict=True)
            if keep
        ]
        return self.separator.join(kept)


class FillerMasker(Masker):
    """Replace absent features with neutral filler of comparable length.

    Holds prompt length and shape roughly constant so attribution isolates a
    feature's *semantic* content from confounds such as total prompt length.
    """

    def __init__(self, filler_token: str = "x", separator: str = " ") -> None:
        if not filler_token:
            msg = "filler_token must be a non-empty string"
            raise ValueError(msg)
        self.filler_token = filler_token
        self.separator = separator

    def mask(self, features: Sequence[Feature], coalition: Coalition) -> str:
        normalized = normalize_coalition(coalition, len(features))
        parts = [
            feature.text if keep else self._filler_for(feature.text)
            for feature, keep in zip(features, normalized, strict=True)
        ]
        return self.separator.join(parts)

    def _filler_for(self, text: str) -> str:
        repeats = max(1, len(text) // len(self.filler_token))
        return (self.filler_token * repeats)[: len(text)] or self.filler_token
