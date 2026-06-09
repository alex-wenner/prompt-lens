"""Pricing and token-count estimation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

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

# Models whose tokenizer tiktoken knows. Claude is deliberately excluded:
# tiktoken is OpenAI's tokenizer and undercounts Claude tokens by 15-20% on
# typical text (more on code), so Claude estimates stay on the conservative
# character heuristic; exact Claude counts require the provider's count_tokens
# endpoint, which a dry run should not call.
_TIKTOKEN_MODEL_RE = re.compile(r"^(gpt-|o\d)")


class TokenEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int
    output_tokens: int


def _heuristic_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


@lru_cache(maxsize=8)
def _tiktoken_encoding(model_name: str) -> Any | None:
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Newer OpenAI models may not be in the installed tiktoken's registry;
        # the o200k family is the closest stand-in.
        return tiktoken.get_encoding("o200k_base")


def _bare_model_name(model: str | None) -> str:
    if not model:
        return ""
    return model.strip().lower().rsplit("/", 1)[-1]


def count_tokens(text: str, model: str | None = None) -> tuple[int, str]:
    """Count tokens in ``text``, returning ``(count, counter_name)``.

    Uses ``tiktoken`` when it is installed and ``model`` is an OpenAI-family
    model whose tokenizer it implements. Everything else (Claude, local and
    unknown models, or tiktoken not installed) falls back to a conservative
    character heuristic that works without network calls.
    """
    name = _bare_model_name(model)
    if name and _TIKTOKEN_MODEL_RE.match(name):
        encoding = _tiktoken_encoding(name)
        if encoding is not None:
            return len(encoding.encode(text)), "tiktoken"
    return _heuristic_tokens(text), "heuristic"


def estimate_tokens(text: str, expected_output_tokens: int = 300) -> TokenEstimate:
    """Use a conservative character heuristic that works for closed and open-weight models."""
    return TokenEstimate(
        input_tokens=_heuristic_tokens(text), output_tokens=expected_output_tokens
    )


def estimate_cost(
    *,
    model: str,
    prompt: str,
    features: int,
    evaluations: int,
    expected_output_tokens: int = 300,
    compare_models: list[str] | None = None,
    evaluation_prompts: Sequence[str] | None = None,
) -> CostEstimate:
    """Estimate provider cost for baseline plus coalition evaluations.

    When ``evaluation_prompts`` (the actual masked prompts, one per coalition)
    is supplied, input tokens are counted per perturbation instead of assuming
    every call resends the full prompt — maskers like ``DropMasker`` produce
    shorter prompts, so this tightens the estimate. ``evaluations`` may exceed
    ``len(evaluation_prompts)`` when each coalition is sampled multiple times.
    """
    total_calls = evaluations + 1
    if evaluation_prompts:
        baseline_tokens, counter = count_tokens(prompt, model)
        per_sweep = sum(count_tokens(masked, model)[0] for masked in evaluation_prompts)
        repeats = max(1, round(evaluations / len(evaluation_prompts)))
        input_tokens = baseline_tokens + per_sweep * repeats
    else:
        per_call, counter = count_tokens(prompt, model)
        input_tokens = per_call * total_calls
    output_tokens = expected_output_tokens * total_calls
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
        token_counter=counter,
    )
