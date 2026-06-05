"""CLI component factories."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from promptlens.adapters import (
    AnthropicAdapter,
    BedrockAdapter,
    EchoAdapter,
    OpenAIAdapter,
    OpenAICompatibleAdapter,
)
from promptlens.core import Adapter, Masker, Sampler, Scorer
from promptlens.maskers import DropMasker, FillerMasker, PlaceholderMasker
from promptlens.samplers import LeaveOneOutSampler, RandomCoalitionSampler
from promptlens.scorers import (
    EmbeddingScorer,
    LengthDriftScorer,
    LogprobScorer,
    ToolAccuracyScorer,
)

_DEFAULT_MODELS: dict[str, tuple[str, tuple[str, ...]]] = {
    "echo": ("echo", ("PROMPTLENS_ECHO_MODEL",)),
    "openai": ("gpt-4o-mini", ("OPENAI_MODEL",)),
    "anthropic": ("claude-3-5-haiku-latest", ("ANTHROPIC_MODEL",)),
    "bedrock": (
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        ("AWS_BEDROCK_MODEL_ID", "BEDROCK_MODEL_ID"),
    ),
    "openai-compatible": ("local", ("OPENAI_COMPATIBLE_MODEL", "OPENAI_MODEL")),
}

# Base number of random coalitions at scale "quick"; larger scales multiply it.
_BASE_RANDOM_COALITIONS = 50


class _TextEmbeddingClient:
    """Deterministic local embedding fallback for CLI smoke runs."""

    def embed(self, text: str) -> Sequence[float]:
        """Return simple text-shape features without making provider calls."""
        char_sum = sum(ord(char) for char in text)
        return (
            float(len(text)),
            float(text.count(" ")),
            float(char_sum % 997),
        )


def build_adapter(
    provider: str,
    model: str | None,
    *,
    temperature: float,
    base_url: str | None,
    use_batch_api: bool = False,
    client: Any | None = None,
) -> Adapter:
    """Build a provider adapter without exposing credentials.

    ``use_batch_api`` opts OpenAI and Anthropic adapters into their native batch
    APIs (cheaper, asynchronous). It is ignored by providers without batch
    support (echo, bedrock, openai-compatible).
    """
    provider_key = provider.strip().lower()
    model_id = _resolve_model(provider_key, model)
    if provider_key == "echo":
        return EchoAdapter(model=model_id)
    if provider_key == "openai":
        return OpenAIAdapter(
            model=model_id,
            temperature=temperature,
            use_batch_api=use_batch_api,
            client=client,
        )
    if provider_key == "anthropic":
        return AnthropicAdapter(
            model=model_id,
            temperature=temperature,
            use_batch_api=use_batch_api,
            client=client,
        )
    if provider_key == "bedrock":
        return BedrockAdapter(model=model_id, temperature=temperature, client=client)
    if provider_key == "openai-compatible":
        endpoint = (
            base_url
            or os.environ.get("OPENAI_COMPATIBLE_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        )
        if not endpoint:
            msg = "openai-compatible provider requires --base-url or OPENAI_COMPATIBLE_BASE_URL"
            raise ValueError(msg)
        api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            return OpenAICompatibleAdapter(
                model=model_id,
                base_url=endpoint,
                api_key=api_key,
                temperature=temperature,
                client=client,
            )
        return OpenAICompatibleAdapter(
            model=model_id,
            base_url=endpoint,
            temperature=temperature,
            client=client,
        )
    msg = f"Unsupported provider: {provider}"
    raise ValueError(msg)


def build_masker(name: str) -> Masker:
    """Build a masking strategy for CLI attribution runs."""
    masker_key = name.strip().lower()
    if masker_key == "placeholder":
        return PlaceholderMasker()
    if masker_key == "drop":
        return DropMasker()
    if masker_key == "filler":
        return FillerMasker()
    msg = f"Unsupported masker: {name}"
    raise ValueError(msg)


def build_sampler(name: str, *, scale: str | int) -> Sampler:
    """Build a coalition sampler for CLI attribution runs."""
    sampler_key = name.strip().lower()
    if sampler_key in {"leave-one-out", "loo"}:
        return LeaveOneOutSampler(repeats=_repeats_from_scale(scale))
    if sampler_key in {"random", "random-coalition"}:
        # Reuse the perturbation scale as a coalition-count multiplier so larger
        # scales buy more random coalitions, mirroring how it adds LOO repeats.
        return RandomCoalitionSampler(
            n_coalitions=_BASE_RANDOM_COALITIONS * _repeats_from_scale(scale)
        )
    msg = f"Unsupported sampler: {name}"
    raise ValueError(msg)


def build_scorer(name: str, *, config_path: str | None = None) -> Scorer:
    """Build an output scorer from a CLI name and optional JSON config."""
    scorer_key = name.strip().lower()
    config = _load_config(config_path)
    if scorer_key == "length":
        return LengthDriftScorer()
    if scorer_key == "embedding":
        return EmbeddingScorer(_TextEmbeddingClient())
    if scorer_key == "logprob":
        return LogprobScorer()
    if scorer_key in {"tool-call", "tool-accuracy"}:
        expected_tool = config.get("expected_tool")
        if not isinstance(expected_tool, str) or not expected_tool:
            msg = "tool-call scorer requires scorer config with non-empty expected_tool"
            raise ValueError(msg)
        required_args = _required_args_from_config(config)
        return ToolAccuracyScorer(expected_tool=expected_tool, required_args=required_args)
    msg = f"Unsupported scorer: {name}"
    raise ValueError(msg)


def _resolve_model(provider: str, model: str | None) -> str:
    if provider not in _DEFAULT_MODELS:
        msg = f"Unsupported provider: {provider}"
        raise ValueError(msg)
    default, env_names = _DEFAULT_MODELS[provider]
    if model:
        return model
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value
    return default


def _repeats_from_scale(scale: str | int) -> int:
    if isinstance(scale, int):
        if scale < 1:
            msg = f"scale must be a positive integer, got {scale}"
            raise ValueError(msg)
        return scale
    repeats = {"quick": 1, "standard": 3, "full": 5}.get(scale)
    if repeats is None:
        msg = f"Unsupported perturbation scale: {scale}"
        raise ValueError(msg)
    return repeats


def _required_args_from_config(config: dict[str, Any]) -> list[str]:
    required_args = config.get("required_args", [])
    if not isinstance(required_args, list) or not all(
        isinstance(item, str) for item in required_args
    ):
        msg = "tool-call scorer required_args must be a list of strings"
        raise ValueError(msg)
    return required_args


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"Unable to read scorer config: {path}"
        raise ValueError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"Scorer config must be valid JSON: {exc.msg}"
        raise ValueError(msg) from exc
    if not isinstance(data, Mapping):
        msg = "Scorer config must contain a JSON object"
        raise ValueError(msg)
    return dict(data)
