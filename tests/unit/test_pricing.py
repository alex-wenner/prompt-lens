import pytest

from promptlens import AttributionHarness
from promptlens.adapters import EchoAdapter
from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions, Usage
from promptlens.core.pricing import estimate_cost_from_baseline, resolve_rates
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter


def test_estimate_multiplies_baseline_usage_by_calls() -> None:
    estimate = estimate_cost_from_baseline(
        model="openai/gpt-4o-mini",
        usage=Usage(input_tokens=100, output_tokens=40),
        features=3,
        evaluations=3,
    )

    # Baseline plus three perturbations: every call is priced at measured usage.
    assert estimate.input_tokens == 100 * 4
    assert estimate.output_tokens == 40 * 4
    rates = resolve_rates("openai/gpt-4o-mini")
    assert estimate.input_cost_usd == pytest.approx(400 / 1_000_000 * rates[0])
    assert estimate.output_cost_usd == pytest.approx(160 / 1_000_000 * rates[1])


def test_estimate_compares_models_on_the_same_usage() -> None:
    estimate = estimate_cost_from_baseline(
        model="openai/gpt-4o-mini",
        usage=Usage(input_tokens=10, output_tokens=10),
        features=1,
        evaluations=1,
        compare_models=["ollama/llama3.2", "anthropic/claude-haiku-4-5"],
    )

    assert estimate.comparisons["ollama/llama3.2"] == 0.0
    assert estimate.comparisons["anthropic/claude-haiku-4-5"] > 0.0


def test_resolve_rates_accepts_bare_model_names() -> None:
    assert resolve_rates("gpt-4o-mini") == resolve_rates("openai/gpt-4o-mini")
    assert resolve_rates("totally-unknown-model") == (0.0, 0.0)


def test_harness_estimate_runs_real_baseline_and_returns_it() -> None:
    harness = AttributionHarness(
        adapter=EchoAdapter(),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )

    baseline, estimate = harness.estimate("Alpha sentence here. Beta sentence here.")

    assert baseline.usage is not None
    assert estimate.features == 2
    assert estimate.evaluations == 2
    # Baseline + 2 evaluations, each at the measured baseline usage.
    assert estimate.input_tokens == baseline.usage.input_tokens * 3


def test_harness_estimate_reuses_supplied_baseline() -> None:
    class CountingAdapter(EchoAdapter):
        calls = 0

        def complete(
            self, prompt: str, tools: ToolDefinitions | None = None
        ) -> CompletionOutput:
            type(self).calls += 1
            return super().complete(prompt, tools=tools)

    adapter = CountingAdapter()
    harness = AttributionHarness(
        adapter=adapter,
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )
    baseline, _ = harness.estimate("One. Two.")
    assert CountingAdapter.calls == 1

    harness.estimate("One. Two.", baseline=baseline)
    assert CountingAdapter.calls == 1  # no extra baseline call


def test_harness_estimate_requires_provider_usage() -> None:
    class NoUsageAdapter(Adapter):
        model = "no-usage"

        def complete(
            self, prompt: str, tools: ToolDefinitions | None = None
        ) -> CompletionOutput:
            return CompletionOutput(text=prompt)

    harness = AttributionHarness(
        adapter=NoUsageAdapter(),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )

    with pytest.raises(ValueError, match="usage"):
        harness.estimate("One. Two.")
