"""Baseline-derived cost projection: no tokenizers, no heuristics."""

import pytest

from promptlens import AttributionHarness, CostGateAborted
from promptlens.core.base import Adapter, CompletionOutput, TokenUsage, ToolDefinitions
from promptlens.core.pricing import project_cost, resolve_rates
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter


class _MeteredAdapter(Adapter):
    """Offline adapter that reports fixed provider usage on every call."""

    def __init__(self, model: str = "gpt-4o-mini", input_tokens: int = 100,
                 output_tokens: int = 40) -> None:
        self.model = model
        self.usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
        self.calls = 0

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        self.calls += 1
        return CompletionOutput(text=prompt.upper(), usage=self.usage)


def test_project_cost_multiplies_baseline_usage() -> None:
    estimate = project_cost(
        model="openai/gpt-4o-mini",
        usage=TokenUsage(input_tokens=1000, output_tokens=200),
        features=4,
        evaluations=4,
    )

    assert estimate.baseline_input_tokens == 1000
    assert estimate.input_tokens == 1000 * 5  # baseline + four masked calls
    assert estimate.output_tokens == 200 * 5
    assert estimate.usage_available and estimate.priced
    assert estimate.total_usd == pytest.approx(
        5000 / 1e6 * 0.15 + 1000 / 1e6 * 0.60
    )


def test_project_cost_without_usage_flags_it() -> None:
    estimate = project_cost(
        model="openai/gpt-4o-mini", usage=None, features=2, evaluations=2
    )

    assert not estimate.usage_available
    assert estimate.input_tokens == 0
    assert estimate.total_usd == 0.0


def test_project_cost_unknown_model_is_unpriced() -> None:
    estimate = project_cost(
        model="mystery/model-x",
        usage=TokenUsage(input_tokens=10, output_tokens=10),
        features=1,
        evaluations=1,
    )

    assert not estimate.priced
    assert estimate.total_usd == 0.0


def test_resolve_rates_matches_bare_model_names() -> None:
    # Adapters carry bare model ids; pricing keys are provider/model.
    assert resolve_rates("gpt-4o-mini") == resolve_rates("openai/gpt-4o-mini")
    assert resolve_rates("claude-haiku-4-5") == resolve_rates("anthropic/claude-haiku-4-5")
    assert resolve_rates("never-heard-of-it") is None


def test_project_cost_compare_models() -> None:
    estimate = project_cost(
        model="openai/gpt-4o-mini",
        usage=TokenUsage(input_tokens=1_000_000, output_tokens=0),
        features=1,
        evaluations=0,
        compare_models=["anthropic/claude-haiku-4-5", "unknown/skipped"],
    )

    assert estimate.comparisons == {"anthropic/claude-haiku-4-5": 1.00}


def test_harness_estimate_from_baseline_counts_planned_evaluations() -> None:
    adapter = _MeteredAdapter()
    harness = AttributionHarness(
        adapter=adapter,
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )
    prompt = "Alpha sentence here. Beta sentence here. Gamma sentence here."

    baseline = harness.run_baseline(prompt)
    estimate = harness.estimate_from_baseline(prompt, baseline)

    assert adapter.calls == 1  # estimating costs exactly the baseline call
    assert estimate.features == 3
    assert estimate.evaluations == 3  # leave-one-out, quick scale
    assert estimate.baseline_input_tokens == 100
    assert estimate.input_tokens == 100 * 4


def test_cost_gate_aborts_before_masked_calls() -> None:
    adapter = _MeteredAdapter()
    harness = AttributionHarness(
        adapter=adapter,
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )

    with pytest.raises(CostGateAborted):
        harness.explain("One sentence. Two sentence.", cost_gate=lambda estimate: False)

    assert adapter.calls == 1  # only the baseline ran


def test_cost_gate_approval_reuses_baseline() -> None:
    adapter = _MeteredAdapter()
    harness = AttributionHarness(
        adapter=adapter,
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )
    prompt = "One sentence. Two sentence."
    seen: list[int] = []

    baseline = harness.run_baseline(prompt)
    result = harness.explain(
        prompt,
        baseline=baseline,
        cost_gate=lambda estimate: seen.append(estimate.evaluations) or True,
    )

    assert seen == [2]
    # one baseline + two masked evaluations; the supplied baseline is not re-run
    assert adapter.calls == 3
    assert result.cost_estimate is not None
    assert result.cost_estimate.baseline_input_tokens == 100
