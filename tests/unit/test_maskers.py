from promptlens import Feature
from promptlens.maskers import DropMasker, FillerMasker, PlaceholderMasker

FEATURES = [
    Feature(name="a", text="Alpha one"),
    Feature(name="b", text="Beta"),
]


def test_placeholder_masker_replaces_absent_feature() -> None:
    masker = PlaceholderMasker()
    assert masker.mask(FEATURES, (True, False)) == "Alpha one [...]"


def test_drop_masker_omits_absent_feature() -> None:
    masker = DropMasker()
    assert masker.mask(FEATURES, (True, False)) == "Alpha one"
    assert masker.mask(FEATURES, (False, True)) == "Beta"


def test_filler_masker_preserves_length() -> None:
    masker = FillerMasker(filler_token="x")
    masked = masker.mask(FEATURES, (False, True))
    filler, kept = masked.split(" ", 1)
    assert kept == "Beta"
    assert len(filler) == len("Alpha one")
    assert set(filler) == {"x"}


def test_filler_masker_rejects_empty_token() -> None:
    import pytest

    with pytest.raises(ValueError, match="non-empty"):
        FillerMasker(filler_token="")


def test_filler_masker_multichar_token_preserves_length() -> None:
    features = [Feature(name="a", text="abcde"), Feature(name="b", text="keep")]
    masker = FillerMasker(filler_token="xy")
    masked = masker.mask(features, (False, True))
    filler, kept = masked.split(" ", 1)
    assert kept == "keep"
    assert len(filler) == len("abcde")


def test_filler_masker_handles_empty_feature_text() -> None:
    features = [Feature(name="a", text=""), Feature(name="b", text="keep")]
    masker = FillerMasker(filler_token="xy")
    assert masker.mask(features, (False, True)) == " keep"


def _spanned_features() -> list[Feature]:
    # "Alpha\n\nBeta" — spans skip the blank line separating the paragraphs.
    return [
        Feature(name="a", text="Alpha", start=0, end=5),
        Feature(name="b", text="Beta", start=7, end=11),
    ]


_SPAN_PROMPT = "Alpha\n\nBeta"


def test_placeholder_splice_preserves_original_formatting() -> None:
    masked = PlaceholderMasker().mask(_spanned_features(), (True, False), prompt=_SPAN_PROMPT)
    assert masked == "Alpha\n\n[...]"


def test_drop_splice_excises_span_only() -> None:
    masked = DropMasker().mask(_spanned_features(), (True, False), prompt=_SPAN_PROMPT)
    assert masked == "Alpha\n\n"


def test_filler_splice_keeps_prompt_length() -> None:
    masked = FillerMasker().mask(_spanned_features(), (False, True), prompt=_SPAN_PROMPT)
    assert masked == "xxxxx\n\nBeta"
    assert len(masked) == len(_SPAN_PROMPT)


def test_splice_appends_spanless_features_like_join_path() -> None:
    features = [*_spanned_features(), Feature(name="tools", text="{schema}")]
    masked = PlaceholderMasker().mask(features, (True, True, False), prompt=_SPAN_PROMPT)
    assert masked == "Alpha\n\nBeta [...]"


def test_overlapping_spans_fall_back_to_join() -> None:
    features = [
        Feature(name="a", text="Alpha", start=0, end=8),
        Feature(name="b", text="Beta", start=5, end=11),
    ]
    masked = PlaceholderMasker().mask(features, (True, False), prompt=_SPAN_PROMPT)
    assert masked == "Alpha [...]"


def test_no_prompt_uses_join_path() -> None:
    masked = PlaceholderMasker().mask(_spanned_features(), (True, False))
    assert masked == "Alpha [...]"
