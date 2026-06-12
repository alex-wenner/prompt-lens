"""Provider pricing and baseline-derived cost projection.

promptlens never guesses token counts with a local tokenizer or a character
heuristic. The harness runs the real baseline completion first, reads the
provider's own metered usage off that response, and projects the sweep as

    projected input  = baseline input tokens  x (evaluations + 1)
    projected output = baseline output tokens x (evaluations + 1)

Every masked prompt is a near-copy of the baseline prompt (one feature hidden),
so the baseline's real usage is the tightest honest per-call proxy available
without running the sweep itself. The CLI shows this projection after the
baseline call and asks before spending the rest.
"""

from __future__ import annotations

from promptlens.core.base import TokenUsage
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


def resolve_rates(model: str) -> tuple[float, float] | None:
    """Return ``(input, output)`` $/MTok for ``model``, or ``None`` if unknown.

    Accepts either the full ``provider/model`` pricing key or a bare model id as
    adapters carry it (``gpt-4o-mini``), matched against the key suffix.
    """
    exact = MODEL_PRICING_USD_PER_MTOK.get(model)
    if exact is not None:
        return exact
    bare = model.strip().lower().rsplit("/", 1)[-1]
    for key, rates in MODEL_PRICING_USD_PER_MTOK.items():
        if key.rsplit("/", 1)[-1] == bare:
            return rates
    return None


def project_cost(
    *,
    model: str,
    usage: TokenUsage | None,
    features: int,
    evaluations: int,
    compare_models: list[str] | None = None,
) -> CostEstimate:
    """Project sweep cost from the baseline completion's real provider usage.

    ``evaluations`` counts the masked-prompt calls still to be made; the
    baseline call itself is added on top. When the adapter reported no usage
    (offline adapters, providers without metering) the estimate carries zero
    tokens and ``usage_available=False`` so renderers can say so instead of
    showing a fake number.
    """
    total_calls = evaluations + 1
    baseline_input = usage.input_tokens if usage else 0
    baseline_output = usage.output_tokens if usage else 0
    input_tokens = baseline_input * total_calls
    output_tokens = baseline_output * total_calls
    rates = resolve_rates(model)
    input_rate, output_rate = rates if rates is not None else (0.0, 0.0)
    comparisons: dict[str, float] = {}
    for compare_model in compare_models or []:
        compare_rates = resolve_rates(compare_model)
        if compare_rates is None:
            continue
        comparisons[compare_model] = (
            input_tokens / 1_000_000 * compare_rates[0]
            + output_tokens / 1_000_000 * compare_rates[1]
        )
    return CostEstimate(
        model=model,
        features=features,
        evaluations=evaluations,
        baseline_input_tokens=baseline_input,
        baseline_output_tokens=baseline_output,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=input_tokens / 1_000_000 * input_rate,
        output_cost_usd=output_tokens / 1_000_000 * output_rate,
        pricing_updated=PRICING_UPDATED,
        comparisons=comparisons,
        usage_available=usage is not None,
        priced=rates is not None,
    )
