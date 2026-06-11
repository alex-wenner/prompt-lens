"""Pricing from measured baseline usage.

promptlens never guesses token counts with a tokenizer or character heuristic.
Cost estimation runs the **baseline completion for real**, reads the provider's
reported input/output token usage, and multiplies it by the number of
perturbation evaluations the sweep will make. Masked prompts are always subsets
of the baseline prompt, so the projection is a tight upper bound on actual
spend.
"""

from __future__ import annotations

from promptlens.core.base import Usage
from promptlens.core.result import CostEstimate

PRICING_UPDATED = "2026-06-09"
MODEL_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "anthropic/claude-opus-4-8": (5.00, 25.00),
    "anthropic/claude-opus-4-7": (5.00, 25.00),
    "anthropic/claude-opus-4-6": (5.00, 25.00),
    "anthropic/claude-sonnet-4-6": (3.00, 15.00),
    "anthropic/claude-haiku-4-5": (1.00, 5.00),
    "openai/gpt-5.5": (5.00, 30.00),
    "openai/gpt-5.5-pro": (30.00, 180.00),
    "openai/gpt-5.4": (2.50, 15.00),
    "openai/gpt-5.4-mini": (0.75, 4.50),
    "openai/gpt-5.4-nano": (0.20, 1.25),
    "openai/gpt-5.4-pro": (30.00, 180.00),
    "openai/gpt-5.3-codex": (1.75, 14.00),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0": (3.00, 15.00),
    "openai-compatible/local": (0.00, 0.00),
    "ollama/llama3.2": (0.00, 0.00),
}


def resolve_rates(model: str) -> tuple[float, float]:
    """Return (input, output) USD/MTok rates for ``model``.

    Accepts both ``provider/model`` keys and bare model ids; unknown models
    price at zero so the call-count projection still renders.
    """
    if model in MODEL_PRICING_USD_PER_MTOK:
        return MODEL_PRICING_USD_PER_MTOK[model]
    for key, rates in MODEL_PRICING_USD_PER_MTOK.items():
        if key.rsplit("/", 1)[-1] == model:
            return rates
    return (0.0, 0.0)


def estimate_cost_from_baseline(
    *,
    model: str,
    usage: Usage,
    features: int,
    evaluations: int,
    compare_models: list[str] | None = None,
) -> CostEstimate:
    """Project sweep cost from one measured baseline completion.

    ``usage`` is the provider-reported token usage of the real baseline call.
    The sweep sends one masked variant of the same prompt per evaluation, so
    total spend is projected as ``baseline usage x (evaluations + 1)`` — the
    baseline itself plus every perturbation.
    """
    total_calls = evaluations + 1
    input_tokens = usage.input_tokens * total_calls
    output_tokens = usage.output_tokens * total_calls
    input_rate, output_rate = resolve_rates(model)
    comparisons = {
        compare_model: (
            input_tokens / 1_000_000 * resolve_rates(compare_model)[0]
            + output_tokens / 1_000_000 * resolve_rates(compare_model)[1]
        )
        for compare_model in compare_models or []
    }
    return CostEstimate(
        model=model,
        features=features,
        evaluations=evaluations,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=input_tokens / 1_000_000 * input_rate,
        output_cost_usd=output_tokens / 1_000_000 * output_rate,
        pricing_updated=PRICING_UPDATED,
        comparisons=comparisons,
    )
