from promptlens import AttributionHarness
from promptlens.adapters import EchoAdapter
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter


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


def test_estimate_supports_model_comparisons() -> None:
    harness = AttributionHarness(
        adapter=EchoAdapter(model="openai/gpt-4o"),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )

    estimate = harness.estimate("One. Two.", compare_models=["openai/gpt-4o-mini"])

    assert estimate.features == 2
    assert "openai/gpt-4o-mini" in estimate.comparisons
