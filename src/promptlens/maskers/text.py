"""Text masking strategies."""

from __future__ import annotations

from collections.abc import Sequence

from promptlens.core.base import Coalition, Feature, Masker, normalize_coalition


def _spanned_features(
    features: Sequence[Feature], prompt: str | None
) -> list[tuple[int, Feature]] | None:
    """Return (index, feature) pairs ordered by span, or None when splicing is unsound.

    Splicing masks into the original prompt requires every spanned feature to
    have in-bounds, non-overlapping offsets. Features without spans (e.g. the
    appended tools feature) are excluded here and appended by the caller,
    mirroring the join path's treatment of them.
    """
    if prompt is None:
        return None
    spanned = [
        (index, feature)
        for index, feature in enumerate(features)
        if feature.start is not None and feature.end is not None
    ]
    if not spanned:
        return None
    spanned.sort(key=lambda item: item[1].start or 0)
    cursor = 0
    for _, feature in spanned:
        assert feature.start is not None and feature.end is not None
        if feature.start < cursor or feature.end > len(prompt) or feature.start >= feature.end:
            return None
        cursor = feature.end
    return spanned


class PlaceholderMasker(Masker):
    """Replace absent features with a stable placeholder while preserving order.

    Keeps prompt structure mostly intact, so attribution measures the effect of
    hiding a feature's *content* while signalling that something was there.
    When the original ``prompt`` is supplied, masks are spliced into it so all
    surrounding formatting (blank lines, headings, spacing) survives verbatim.
    """

    def __init__(self, placeholder: str = "[...]", separator: str = " ") -> None:
        self.placeholder = placeholder
        self.separator = separator

    def mask(
        self,
        features: Sequence[Feature],
        coalition: Coalition,
        prompt: str | None = None,
    ) -> str:
        normalized = normalize_coalition(coalition, len(features))
        spanned = _spanned_features(features, prompt)
        if spanned is None or prompt is None:
            parts = [
                feature.text if keep else self._masked_span(feature.text)
                for feature, keep in zip(features, normalized, strict=True)
            ]
            return self.separator.join(parts)
        return self._splice(features, normalized, prompt, spanned)

    def _splice(
        self,
        features: Sequence[Feature],
        normalized: Coalition,
        prompt: str,
        spanned: list[tuple[int, Feature]],
    ) -> str:
        pieces: list[str] = []
        cursor = 0
        spanned_indices = set()
        for index, feature in spanned:
            spanned_indices.add(index)
            assert feature.start is not None and feature.end is not None
            pieces.append(prompt[cursor : feature.start])
            original = prompt[feature.start : feature.end]
            pieces.append(original if normalized[index] else self._masked_span(original))
            cursor = feature.end
        pieces.append(prompt[cursor:])
        masked = "".join(pieces)
        trailing = [
            feature.text if normalized[index] else self._masked_span(feature.text)
            for index, feature in enumerate(features)
            if index not in spanned_indices
        ]
        if trailing:
            masked = self.separator.join([masked, *trailing])
        return masked

    def _masked_span(self, original: str) -> str:
        del original
        return self.placeholder


class DropMasker(Masker):
    """Omit absent features entirely, collapsing their separators.

    Attribution measures the effect of removing a feature outright, with no
    placeholder hint left behind. Useful when the presence of a placeholder
    would itself perturb the model. When the original ``prompt`` is supplied,
    masked spans are excised from it and the surviving text keeps its original
    formatting.
    """

    def __init__(self, separator: str = " ") -> None:
        self.separator = separator

    def mask(
        self,
        features: Sequence[Feature],
        coalition: Coalition,
        prompt: str | None = None,
    ) -> str:
        normalized = normalize_coalition(coalition, len(features))
        spanned = _spanned_features(features, prompt)
        if spanned is None or prompt is None:
            kept = [
                feature.text
                for feature, keep in zip(features, normalized, strict=True)
                if keep
            ]
            return self.separator.join(kept)
        pieces: list[str] = []
        cursor = 0
        spanned_indices = set()
        for index, feature in spanned:
            spanned_indices.add(index)
            assert feature.start is not None and feature.end is not None
            pieces.append(prompt[cursor : feature.start])
            if normalized[index]:
                pieces.append(prompt[feature.start : feature.end])
            cursor = feature.end
        pieces.append(prompt[cursor:])
        masked = "".join(pieces)
        trailing = [
            feature.text
            for index, feature in enumerate(features)
            if index not in spanned_indices and normalized[index]
        ]
        if trailing:
            masked = self.separator.join([masked, *trailing])
        return masked


class FillerMasker(PlaceholderMasker):
    """Replace absent features with neutral filler of comparable length.

    Holds prompt length and shape roughly constant so attribution isolates a
    feature's *semantic* content from confounds such as total prompt length.
    When the original ``prompt`` is supplied, the filler exactly matches each
    masked span's length, so the masked prompt keeps the original's length.
    """

    def __init__(self, filler_token: str = "x", separator: str = " ") -> None:
        if not filler_token:
            msg = "filler_token must be a non-empty string"
            raise ValueError(msg)
        super().__init__(separator=separator)
        self.filler_token = filler_token

    def _masked_span(self, original: str) -> str:
        if not original:
            return ""
        token = self.filler_token
        repeats = (len(original) + len(token) - 1) // len(token)
        return (token * repeats)[: len(original)]
