import json

import pytest

from promptlens.core import CompletionOutput
from promptlens.scorers import ToolArgumentDriftScorer, ToolSequenceDriftScorer


def _output(*names: str) -> CompletionOutput:
    return CompletionOutput(text="", tool_calls=[{"name": name} for name in names])


def _call_output(*calls: tuple[str, dict]) -> CompletionOutput:
    return CompletionOutput(
        text="", tool_calls=[{"name": name, "arguments": args} for name, args in calls]
    )


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


def test_argument_scorer_same_tools_same_args_scores_zero() -> None:
    baseline = _call_output(("search", {"query": "refunds"}))
    candidate = _call_output(("search", {"query": "refunds"}))
    assert ToolArgumentDriftScorer().score(baseline, candidate) == 0.0


def test_argument_scorer_no_tools_scores_zero() -> None:
    assert ToolArgumentDriftScorer().score(_output(), _output()) == 0.0


def test_argument_scorer_counts_argument_changes() -> None:
    baseline = _call_output(("search", {"query": "refunds"}))
    candidate = _call_output(("search", {"query": "shipping"}))
    # Single call, one of one params changed: argument_weight * 1.0.
    assert ToolArgumentDriftScorer(argument_weight=0.5).score(baseline, candidate) == 0.5


def test_argument_changes_never_outweigh_tool_changes() -> None:
    baseline = _call_output(("search", {"query": "refunds", "limit": 5}))
    different_args = _call_output(("search", {"query": "shipping", "limit": 1}))
    different_tool = _call_output(("lookup", {"query": "refunds", "limit": 5}))
    scorer = ToolArgumentDriftScorer(argument_weight=0.5)

    assert scorer.score(baseline, different_args) <= 0.5
    assert scorer.score(baseline, different_tool) == 1.0


def test_explicit_none_matches_missing_param_by_default() -> None:
    baseline = _call_output(("search", {"query": "refunds", "limit": None}))
    candidate = _call_output(("search", {"query": "refunds"}))
    assert ToolArgumentDriftScorer().score(baseline, candidate) == 0.0


def test_explicit_none_counts_when_none_is_meaningful() -> None:
    baseline = _call_output(("search", {"query": "refunds", "limit": None}))
    candidate = _call_output(("search", {"query": "refunds"}))
    scorer = ToolArgumentDriftScorer(none_is_missing=False, argument_weight=1.0)
    assert scorer.score(baseline, candidate) == 0.5


def test_zero_weight_param_is_inert() -> None:
    baseline = _call_output(("search", {"query": "refunds", "reason": "a"}))
    candidate = _call_output(("search", {"query": "refunds", "reason": "b"}))
    scorer = ToolArgumentDriftScorer(param_weights={"reason": 0.0})
    assert scorer.score(baseline, candidate) == 0.0


def test_param_weights_shift_drift_toward_critical_params() -> None:
    baseline = _call_output(("transfer", {"account_id": "a-1", "memo": "rent"}))
    candidate = _call_output(("transfer", {"account_id": "a-2", "memo": "rent"}))
    heavy = ToolArgumentDriftScorer(argument_weight=1.0, param_weights={"account_id": 9.0})
    flat = ToolArgumentDriftScorer(argument_weight=1.0)

    assert heavy.score(baseline, candidate) == 0.9
    assert flat.score(baseline, candidate) == 0.5


def test_argument_scorer_reads_openai_json_string_arguments() -> None:
    baseline = CompletionOutput(
        text="",
        tool_calls=[
            {"function": {"name": "search", "arguments": json.dumps({"query": "refunds"})}}
        ],
    )
    candidate = _call_output(("search", {"query": "refunds"}))
    assert ToolArgumentDriftScorer().score(baseline, candidate) == 0.0


def test_argument_scorer_dropped_call_scores_partial_drift() -> None:
    baseline = _call_output(("search", {"query": "x"}), ("answer", {}))
    candidate = _call_output(("search", {"query": "x"}))
    assert ToolArgumentDriftScorer().score(baseline, candidate) == 0.5


def test_argument_scorer_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="argument_weight"):
        ToolArgumentDriftScorer(argument_weight=1.5)
    with pytest.raises(ValueError, match="param_weights"):
        ToolArgumentDriftScorer(param_weights={"q": -1.0})
    with pytest.raises(ValueError, match="default_param_weight"):
        ToolArgumentDriftScorer(default_param_weight=-0.1)


def test_argument_scorer_orientation_is_drift() -> None:
    assert ToolArgumentDriftScorer().orientation == "drift"
