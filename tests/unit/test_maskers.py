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
