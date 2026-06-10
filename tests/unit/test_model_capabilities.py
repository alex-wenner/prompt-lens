from promptlens.adapters.models import (
    KNOWN_MODELS,
    lookup_model,
    supports_logprobs,
    supports_temperature,
)


def test_known_logprob_models_supported() -> None:
    assert supports_logprobs("gpt-4o") is True
    assert supports_logprobs("gpt-4.1-mini") is True
    # Provider-prefixed names are normalized.
    assert supports_logprobs("openai/gpt-4o-mini") is True


def test_reasoning_models_do_not_support_logprobs() -> None:
    assert supports_logprobs("gpt-5.5") is False
    assert supports_logprobs("gpt-5.4-mini") is False
    assert supports_logprobs("gpt-5.3-codex") is False
    assert supports_logprobs("o3-mini") is False


def test_anthropic_models_do_not_support_logprobs() -> None:
    assert supports_logprobs("claude-opus-4-8") is False
    assert supports_logprobs("anthropic/claude-sonnet-4-6") is False


def test_unknown_models_use_heuristic() -> None:
    # Unknown GPT-5 / o-series / Claude variants are treated as reasoning models.
    assert supports_logprobs("gpt-5.9-ultra") is False
    assert supports_logprobs("o3-pro") is False
    assert supports_logprobs("claude-opus-5-0") is False
    # Anything else defaults to supporting logprobs.
    assert supports_logprobs("some-local-model") is True


def test_lookup_model_returns_metadata() -> None:
    info = lookup_model("gpt-5.5")
    assert info is not None
    assert info.provider == "openai"
    assert info.supports_logprobs is False
    assert lookup_model("not-a-real-model") is None


def test_registry_includes_current_flagships() -> None:
    assert "gpt-5.5" in KNOWN_MODELS
    assert "claude-opus-4-8" in KNOWN_MODELS


def test_opus_4_7_plus_do_not_support_temperature() -> None:
    assert supports_temperature("claude-opus-4-8") is False
    assert supports_temperature("anthropic/claude-opus-4-7") is False


def test_other_models_support_temperature() -> None:
    assert supports_temperature("claude-opus-4-6") is True
    assert supports_temperature("claude-sonnet-4-6") is True
    assert supports_temperature("claude-haiku-4-5") is True
    assert supports_temperature("gpt-4o") is True
    # Unknown models keep the historical default of sending temperature.
    assert supports_temperature("some-local-model") is True
