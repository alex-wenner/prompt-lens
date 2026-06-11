from collections.abc import Sequence

from promptlens import AttributionHarness, Feature, PromptMutation, PromptMutator
from promptlens.adapters import EchoAdapter
from promptlens.core.base import CompletionOutput, ToolDefinitions
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter


class StaticMutator(PromptMutator):
    def mutate(
        self,
        prompt: str,
        features: Sequence[Feature],
        tools: ToolDefinitions | None = None,
    ) -> list[PromptMutation]:
        return [
            PromptMutation(
                prompt="Alpha rewritten. Beta sentence.",
                feature=features[0],
                metadata={"source": "test"},
            )
        ]


def test_harness_runs_leave_one_out_pipeline() -> None:
    harness = AttributionHarness(
        adapter=EchoAdapter(),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )

    result = harness.explain("Alpha sentence. Beta sentence.")

    assert result.baseline_output.text == "Alpha sentence. Beta sentence."
    assert len(result.attributions) == 2
    assert len(result.evaluations) == 2
    assert result.cost_estimate is not None
    assert result.cost_estimate.evaluations == 2


def test_harness_can_run_supplementary_mutations() -> None:
    harness = AttributionHarness(
        adapter=EchoAdapter(),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
        supplementary_mutator=StaticMutator(),
    )

    result = harness.explain("Alpha sentence. Beta sentence.")

    assert len(result.attributions) == 2
    assert len(result.supplementary_evaluations) == 1
    supplementary = result.supplementary_evaluations[0]
    assert supplementary.kind == "prompt-mutation"
    assert supplementary.feature is not None
    assert supplementary.feature.name == "sentence_1"
    assert supplementary.prompt == "Alpha rewritten. Beta sentence."
    assert result.to_dict()["supplementary_evaluations"][0]["metadata"] == {"source": "test"}


def test_estimate_supports_model_comparisons() -> None:
    harness = AttributionHarness(
        adapter=EchoAdapter(model="openai/gpt-4o"),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )

    _, estimate = harness.estimate("One. Two.", compare_models=["openai/gpt-4o-mini"])

    assert estimate.features == 2
    assert "openai/gpt-4o-mini" in estimate.comparisons


class CountingAdapter(EchoAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def complete_batch(
        self, prompts: Sequence[str], tools: ToolDefinitions | None = None
    ) -> list[CompletionOutput]:
        self.batch_sizes.append(len(prompts))
        return super().complete_batch(prompts, tools=tools)


def test_samples_per_coalition_expands_evaluations() -> None:
    adapter = CountingAdapter()
    harness = AttributionHarness(
        adapter=adapter,
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
        samples_per_coalition=3,
    )

    result = harness.explain("Alpha sentence. Beta sentence.")

    # Two coalitions, each evaluated three times in a single batch call.
    assert adapter.batch_sizes == [6]
    assert len(result.evaluations) == 2
    assert result.cost_estimate is not None
    assert result.cost_estimate.evaluations == 6


def test_samples_per_coalition_must_be_positive() -> None:
    import pytest

    with pytest.raises(ValueError, match="samples_per_coalition"):
        AttributionHarness(
            adapter=EchoAdapter(),
            segmenter=SentenceSegmenter(),
            scorer=LengthDriftScorer(),
            samples_per_coalition=0,
        )
