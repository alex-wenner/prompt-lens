"""CLI component factories."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from promptlens.adapters import (
    AnthropicAdapter,
    BedrockAdapter,
    CopilotAdapter,
    EchoAdapter,
    GeminiAdapter,
    GrokAdapter,
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
    OpenAIEmbeddingClient,
    ToolAccuracyScorer,
)

_DEFAULT_MODELS: dict[str, tuple[str, tuple[str, ...]]] = {
    "echo": ("echo", ("PROMPTLENS_ECHO_MODEL",)),
    "openai": ("gpt-5.4-mini", ("OPENAI_MODEL",)),
    "anthropic": ("claude-haiku-4-5", ("ANTHROPIC_MODEL",)),
    "bedrock": (
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        ("AWS_BEDROCK_MODEL_ID", "BEDROCK_MODEL_ID"),
    ),
    "openai-compatible": ("local", ("OPENAI_COMPATIBLE_MODEL", "OPENAI_MODEL")),
}


@dataclass(frozen=True)
class _SdkProvider:
    """Connection defaults for a provider reached through its official SDK."""

    adapter: Callable[..., Adapter]
    default_model: str
    model_envs: tuple[str, ...]
    api_key_envs: tuple[str, ...]


# Branded providers that each ship a dedicated, official SDK adapter (matching
# OpenAI and Anthropic) rather than going through the generic OpenAI-compatible
# HTTP path. GitHub Copilot is handled separately because its SDK takes a token
# rather than an API key (see ``_build_copilot_adapter``).
_SDK_PROVIDERS: dict[str, _SdkProvider] = {
    "grok": _SdkProvider(
        adapter=GrokAdapter,
        default_model="grok-4",
        model_envs=("XAI_MODEL", "GROK_MODEL"),
        api_key_envs=("XAI_API_KEY", "GROK_API_KEY"),
    ),
    "gemini": _SdkProvider(
        adapter=GeminiAdapter,
        default_model="gemini-3.5-flash",
        model_envs=("GEMINI_MODEL", "GOOGLE_MODEL"),
        api_key_envs=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    ),
}

# Connection defaults for the GitHub Copilot SDK adapter.
_COPILOT_DEFAULT_MODEL = "gpt-5.4"
_COPILOT_MODEL_ENVS = ("COPILOT_MODEL", "GITHUB_COPILOT_MODEL")
_COPILOT_TOKEN_ENVS = ("GITHUB_COPILOT_TOKEN", "COPILOT_API_KEY", "GITHUB_TOKEN")

# Friendly aliases that resolve to a built-in provider key.
_PROVIDER_ALIASES: dict[str, str] = {
    "xai": "grok",
    "google": "gemini",
    "github": "copilot",
    "github-copilot": "copilot",
}

# Base number of random coalitions at scale "quick"; larger scales multiply it.
_BASE_RANDOM_COALITIONS = 50


class _TextEmbeddingClient:
    """Deterministic local embedding fallback for offline CLI smoke runs.

    This is **not** a semantic embedding: it derives a few cheap text-shape
    features without contacting a provider, so it is only useful for smoke tests
    and demos. Select it explicitly via the ``embedding-local`` scorer name. For
    real attribution use the ``embedding`` scorer with a provider config.
    """

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
    support (echo, bedrock, copilot, grok, gemini, openai-compatible).
    """
    provider_key = provider.strip().lower()
    provider_key = _PROVIDER_ALIASES.get(provider_key, provider_key)
    if provider_key == "copilot":
        return _build_copilot_adapter(model, temperature=temperature, client=client)
    if provider_key in _SDK_PROVIDERS:
        return _build_sdk_provider_adapter(
            provider_key, model, temperature=temperature, client=client
        )
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


def _build_copilot_adapter(
    model: str | None,
    *,
    temperature: float,
    client: Any | None,
) -> Adapter:
    """Build the GitHub Copilot adapter backed by the official Copilot SDK."""
    model_id = model or _first_env(_COPILOT_MODEL_ENVS) or _COPILOT_DEFAULT_MODEL
    github_token = _first_env(_COPILOT_TOKEN_ENVS)
    return CopilotAdapter(
        model=model_id,
        temperature=temperature,
        github_token=github_token,
        client=client,
    )


def _build_sdk_provider_adapter(
    provider_key: str,
    model: str | None,
    *,
    temperature: float,
    client: Any | None,
) -> Adapter:
    """Build a branded provider adapter backed by its official SDK (Grok, Gemini)."""
    spec = _SDK_PROVIDERS[provider_key]
    model_id = model or _first_env(spec.model_envs) or spec.default_model
    api_key = _first_env(spec.api_key_envs)
    return spec.adapter(
        model=model_id,
        temperature=temperature,
        api_key=api_key,
        client=client,
    )


def _first_env(env_names: Sequence[str]) -> str | None:
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value
    return None


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
    if scorer_key in {"embedding-local", "text-shape"}:
        # Explicit opt-in to the deterministic text-shape fallback (offline only).
        return EmbeddingScorer(_TextEmbeddingClient())
    if scorer_key == "embedding":
        # Real semantic embeddings require a provider; the offline toy is opt-in
        # under the embedding-local name so plain "embedding" is never mistaken
        # for a semantic scorer.
        return EmbeddingScorer(_build_embedding_client(config))
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


def _build_embedding_client(config: dict[str, Any]) -> Any:
    """Build a provider-backed embedding client from scorer config.

    The ``embedding`` scorer is semantic and therefore needs a provider. Config
    must name a ``provider`` (``openai`` or ``openai-compatible``); for offline
    smoke runs use the ``embedding-local`` scorer instead.
    """
    provider = config.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        msg = (
            "embedding scorer requires scorer config with a 'provider', e.g. "
            '{"provider": "openai", "model": "text-embedding-3-small"}. '
            "Use the 'embedding-local' scorer for an offline deterministic fallback."
        )
        raise ValueError(msg)
    provider_key = provider.strip().lower()
    model = config.get("model")
    if model is not None and not isinstance(model, str):
        msg = "embedding scorer 'model' must be a string"
        raise ValueError(msg)
    if provider_key == "openai":
        return OpenAIEmbeddingClient(model=model or "text-embedding-3-small")
    if provider_key == "openai-compatible":
        base_url = (
            config.get("base_url")
            or os.environ.get("OPENAI_COMPATIBLE_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        )
        if not isinstance(base_url, str) or not base_url:
            msg = (
                "embedding scorer provider 'openai-compatible' requires a 'base_url' "
                "in scorer config or the OPENAI_COMPATIBLE_BASE_URL environment variable"
            )
            raise ValueError(msg)
        return OpenAIEmbeddingClient(model=model or "local", base_url=base_url)
    msg = f"Unsupported embedding scorer provider: {provider}"
    raise ValueError(msg)


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
