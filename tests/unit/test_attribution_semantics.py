from promptlens import AttributionHarness
from promptlens.core.base import Adapter, CompletionOutput, Scorer, ToolDefinitions
from promptlens.scorers import CompositeScorer, LengthDriftScorer, ToolAccuracyScorer
from promptlens.segmenters import SentenceSegmenter


def test_drift_scorers_default_orientation() -> None:
    assert Scorer.orientation == "drift"
    assert LengthDriftScorer().orientation == "drift"


def test_tool_accuracy_scorer_is_objective() -> None:
    assert ToolAccuracyScorer(expected_tool="search").orientation == "objective"


def test_composite_adopts_shared_orientation() -> None:
    composite = CompositeScorer([(LengthDriftScorer(), 0.5), (LengthDriftScorer(), 0.5)])

    assert composite.orientation == "drift"


def test_composite_rejects_mixed_orientation() -> None:
    import pytest

    with pytest.raises(ValueError, match="share one orientation"):
        CompositeScorer(
            [(LengthDriftScorer(), 0.5), (ToolAccuracyScorer(expected_tool="search"), 0.5)]
        )


class ToolRoutingAdapter(Adapter):
    """Emit the expected tool call only when a trigger sentence survives masking."""

    def __init__(self, trigger: str) -> None:
        self.model = "tool-routing"
        self.trigger = trigger

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        if self.trigger in prompt:
            return CompletionOutput(
                text=prompt,
                tool_calls=[{"name": "search", "arguments": {"query": "docs"}}],
            )
        return CompletionOutput(text=prompt, tool_calls=[])


def _tool_routing_harness() -> AttributionHarness:
    return AttributionHarness(
        adapter=ToolRoutingAdapter(trigger="Use the search tool."),
        segmenter=SentenceSegmenter(),
        scorer=ToolAccuracyScorer(expected_tool="search", required_args=["query"]),
    )


def test_objective_scorer_attributes_drop_not_quality() -> None:
    # Two sentences; only the first drives the correct tool call.
    result = _tool_routing_harness().explain("Use the search tool. Be concise.")

    by_name = {a.feature.name: a for a in result.attributions}
    driver = by_name["sentence_1"]
    bystander = by_name["sentence_2"]

    # Masking the driver collapses the objective from 1.0 to 0.0 -> high attribution.
    assert driver.value == 1.0
    # Masking the bystander keeps the correct tool call -> zero attribution, even
    # though the raw objective stays high. This is the drift-vs-quality fix.
    assert bystander.value == 0.0
    assert driver.value > bystander.value


def test_objective_coalition_score_stays_raw() -> None:
    result = _tool_routing_harness().explain("Use the search tool. Be concise.")

    scores = {tuple(ev.coalition): ev.score for ev in result.evaluations}
    # Masking sentence_2 (bystander) keeps a raw objective of 1.0 (transparent),
    # while masking sentence_1 (driver) drops it to 0.0.
    assert scores[(True, False)] == 1.0
    assert scores[(False, True)] == 0.0


def test_drift_scorer_attribution_uses_raw_score() -> None:
    harness = AttributionHarness(
        adapter=ToolRoutingAdapter(trigger="never matches"),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )

    result = harness.explain("Alpha sentence. Beta sentence.")

    # Drift path is unchanged: two features, real-valued attributions.
    assert len(result.attributions) == 2
    assert all(isinstance(a.value, float) for a in result.attributions)
