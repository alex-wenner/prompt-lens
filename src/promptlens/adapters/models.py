"""Known provider models and their capabilities.

Capability data is sourced from the official provider documentation and was
last reviewed on :data:`CAPABILITIES_REVIEWED`:

* OpenAI models and pricing — https://developers.openai.com/api/docs/models
* Anthropic models — https://docs.anthropic.com/en/docs/about-claude/models/overview

The most important capability tracked here is **logprobs support**, because only
some models return token log probabilities (which :class:`~promptlens.scorers.logprob.LogprobScorer`
relies on). The OpenAI GPT-5 reasoning family — and the older ``o``-series
reasoning models — do not accept the ``logprobs`` parameter on the Chat
Completions API and reject the request, whereas the GPT-4o and GPT-4.1 families
do. Anthropic's Messages API does not expose token log probabilities for any
model, so every Anthropic/Bedrock model is treated as not supporting logprobs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

CAPABILITIES_REVIEWED = "2026-06-06"


class ModelInfo(BaseModel):
    """Static capability metadata for a known provider model."""

    model_config = ConfigDict(frozen=True)

    name: str
    provider: str
    supports_logprobs: bool


# OpenAI Chat Completions models that return token log probabilities.
_OPENAI_LOGPROB_MODELS = (
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
)

# OpenAI reasoning models that reject the ``logprobs`` parameter. This covers the
# current GPT-5 family (including the codex and chat-latest variants) and the
# older ``o``-series reasoning models.
_OPENAI_NON_LOGPROB_MODELS = (
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.4-pro",
    "gpt-5.3-codex",
    "gpt-5-chat-latest",
    "o1",
    "o1-mini",
    "o3",
    "o3-mini",
    "o4-mini",
)

# Current Anthropic models. The Messages API does not expose token logprobs, so
# they are all recorded as not supporting logprobs.
_ANTHROPIC_MODELS = (
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
)


def _build_registry() -> dict[str, ModelInfo]:
    registry: dict[str, ModelInfo] = {}
    for name in _OPENAI_LOGPROB_MODELS:
        registry[name] = ModelInfo(name=name, provider="openai", supports_logprobs=True)
    for name in _OPENAI_NON_LOGPROB_MODELS:
        registry[name] = ModelInfo(name=name, provider="openai", supports_logprobs=False)
    for name in _ANTHROPIC_MODELS:
        registry[name] = ModelInfo(name=name, provider="anthropic", supports_logprobs=False)
    return registry


KNOWN_MODELS: dict[str, ModelInfo] = _build_registry()


def _normalize(model: str) -> str:
    """Strip an optional ``provider/`` prefix and surrounding whitespace."""
    normalized = model.strip().lower()
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[-1]
    return normalized


def lookup_model(model: str) -> ModelInfo | None:
    """Return registry metadata for ``model`` if it is a known model."""
    return KNOWN_MODELS.get(_normalize(model))


def supports_logprobs(model: str) -> bool:
    """Return whether ``model`` returns token log probabilities.

    Known models use their recorded capability. For unknown models a heuristic is
    applied: OpenAI reasoning families (the ``gpt-5`` series and the ``o``-series)
    and any Anthropic/Bedrock Claude model are treated as not supporting logprobs,
    while everything else defaults to supporting them.
    """
    name = _normalize(model)
    info = KNOWN_MODELS.get(name)
    if info is not None:
        return info.supports_logprobs
    if name.startswith("gpt-5") or name.startswith(("o1", "o3", "o4")):
        return False
    if "claude" in name:
        return False
    return True
