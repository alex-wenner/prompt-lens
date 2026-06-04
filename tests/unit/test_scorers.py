from promptlens.core import CompletionOutput
from promptlens.scorers import ToolAccuracyScorer, cosine_distance


def test_cosine_distance_identical_vectors_is_zero() -> None:
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == 0.0


def test_tool_accuracy_scores_required_args() -> None:
    scorer = ToolAccuracyScorer(expected_tool="search", required_args=["query", "limit"])
    output = CompletionOutput(
        text="",
        tool_calls=[{"name": "search", "arguments": {"query": "docs", "limit": 5}}],
    )

    assert scorer.score(CompletionOutput(text=""), output) == 1.0
