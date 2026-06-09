from promptlens.core import CompletionOutput
from promptlens.scorers import ToolSequenceDriftScorer


def _output(*names: str) -> CompletionOutput:
    return CompletionOutput(text="", tool_calls=[{"name": name} for name in names])


def test_identical_sequences_score_zero() -> None:
    scorer = ToolSequenceDriftScorer()
    assert scorer.score(_output("search", "answer"), _output("search", "answer")) == 0.0


def test_no_tools_on_either_side_scores_zero() -> None:
    assert ToolSequenceDriftScorer().score(_output(), _output()) == 0.0


def test_disjoint_sequences_score_one() -> None:
    assert ToolSequenceDriftScorer().score(_output("a", "b"), _output("c", "d")) == 1.0


def test_dropped_call_scores_partial_drift() -> None:
    score = ToolSequenceDriftScorer().score(_output("search", "answer"), _output("search"))
    assert score == 0.5


def test_reordered_calls_count_as_drift() -> None:
    score = ToolSequenceDriftScorer().score(_output("a", "b"), _output("b", "a"))
    assert 0.0 < score <= 1.0


def test_openai_function_shape_names_are_read() -> None:
    baseline = CompletionOutput(text="", tool_calls=[{"function": {"name": "search"}}])
    candidate = CompletionOutput(text="", tool_calls=[{"name": "search"}])
    assert ToolSequenceDriftScorer().score(baseline, candidate) == 0.0


def test_orientation_is_drift() -> None:
    assert ToolSequenceDriftScorer().orientation == "drift"
