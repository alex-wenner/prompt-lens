from promptlens.core import CompletionOutput
from promptlens.scorers import (
    CompositeScorer,
    LengthDriftScorer,
    ToolAccuracyScorer,
    cosine_distance,
)


def test_cosine_distance_identical_vectors_is_zero() -> None:
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == 0.0


def test_tool_accuracy_scores_required_args() -> None:
    scorer = ToolAccuracyScorer(expected_tool="search", required_args=["query", "limit"])
    output = CompletionOutput(
        text="",
        tool_calls=[{"name": "search", "arguments": {"query": "docs", "limit": 5}}],
    )

    assert scorer.score(CompletionOutput(text=""), output) == 1.0


def test_composite_scorer_weights_components() -> None:
    baseline = CompletionOutput(text="aaaa")
    candidate = CompletionOutput(text="aa")
    drift = LengthDriftScorer().score(baseline, candidate)

    composite = CompositeScorer([(LengthDriftScorer(), 0.25), (LengthDriftScorer(), 0.75)])

    assert composite.score(baseline, candidate) == drift


def test_composite_scorer_requires_components() -> None:
    import pytest

    with pytest.raises(ValueError, match="at least one"):
        CompositeScorer([])

