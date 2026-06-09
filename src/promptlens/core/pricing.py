"""Pricing and token-count estimation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

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
}


class TokenEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int
    output_tokens: int


def estimate_tokens(text: str, expected_output_tokens: int = 300) -> TokenEstimate:
    """Use a conservative character heuristic that works for closed and open-weight models."""
    input_tokens = max(1, (len(text) + 3) // 4)
    return TokenEstimate(input_tokens=input_tokens, output_tokens=expected_output_tokens)


def estimate_cost(
    *,
    model: str,
    prompt: str,
    features: int,
    evaluations: int,
    expected_output_tokens: int = 300,
    compare_models: list[str] | None = None,
) -> CostEstimate:
    """Estimate provider cost for baseline plus coalition evaluations."""
    total_calls = evaluations + 1
    token_estimate = estimate_tokens(prompt, expected_output_tokens=expected_output_tokens)
    input_tokens = token_estimate.input_tokens * total_calls
    output_tokens = token_estimate.output_tokens * total_calls
    input_rate, output_rate = MODEL_PRICING_USD_PER_MTOK.get(model, (0.0, 0.0))
    input_cost = input_tokens / 1_000_000 * input_rate
    output_cost = output_tokens / 1_000_000 * output_rate
    comparisons: dict[str, float] = {}
    for compare_model in compare_models or []:
        compare_input_rate, compare_output_rate = MODEL_PRICING_USD_PER_MTOK.get(
            compare_model, (0.0, 0.0)
        )
        comparisons[compare_model] = (
            input_tokens / 1_000_000 * compare_input_rate
            + output_tokens / 1_000_000 * compare_output_rate
        )
    return CostEstimate(
        model=model,
        features=features,
        evaluations=evaluations,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        pricing_updated=PRICING_UPDATED,
        comparisons=comparisons,
    )
