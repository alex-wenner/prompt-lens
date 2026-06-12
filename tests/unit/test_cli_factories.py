import json

import pytest

from promptlens.adapters import (
    AnthropicAdapter,
    CopilotAdapter,
    EchoAdapter,
    GeminiAdapter,
    GrokAdapter,
    OpenAIAdapter,
    OpenAICompatibleAdapter,
)
from promptlens.cli.factories import build_adapter, build_masker, build_sampler, build_scorer
from promptlens.core import CompletionOutput
from promptlens.maskers import DropMasker, FillerMasker, PlaceholderMasker
from promptlens.samplers import LeaveOneOutSampler, RandomCoalitionSampler
from promptlens.scorers import (
    EmbeddingScorer,
    LengthDriftScorer,
    OpenAIEmbeddingClient,
    ToolAccuracyScorer,
    ToolArgumentDriftScorer,
)


def test_build_adapter_grok_uses_sdk_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAI_MODEL", raising=False)
    monkeypatch.delenv("GROK_MODEL", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "secret-key")
    client = object()

    adapter = build_adapter("grok", None, temperature=0.0, base_url=None, client=client)

    assert isinstance(adapter, GrokAdapter)
    assert adapter.model == "grok-4"
    assert adapter.api_key == "secret-key"
    assert adapter._client is client


def test_build_adapter_gemini_alias_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-pro")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    client = object()

    adapter = build_adapter("google", None, temperature=0.3, base_url=None, client=client)

    assert isinstance(adapter, GeminiAdapter)
    assert adapter.model == "gemini-3.1-pro"
    assert adapter.temperature == 0.3
    assert adapter.api_key == "gemini-key"


def test_build_adapter_copilot_uses_sdk_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN", "copilot-token")
    client = object()

    adapter = build_adapter(
        "copilot", "gpt-4.1", temperature=0.0, base_url=None, client=client
    )

    assert isinstance(adapter, CopilotAdapter)
    assert adapter.model == "gpt-4.1"
    assert adapter.github_token == "copilot-token"
    assert adapter._client is client


def test_build_adapter_copilot_alias_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COPILOT_MODEL", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_MODEL", raising=False)
    monkeypatch.setenv("COPILOT_MODEL", "gpt-5.4-mini")

    adapter = build_adapter("github", None, temperature=0.0, base_url=None, client=object())

    assert isinstance(adapter, CopilotAdapter)
    assert adapter.model == "gpt-5.4-mini"


def test_build_adapter_defaults_to_echo() -> None:
    adapter = build_adapter("echo", None, temperature=0.0, base_url=None)

    assert isinstance(adapter, EchoAdapter)
    assert adapter.model == "echo"


def test_build_adapter_uses_injected_openai_client() -> None:
    client = object()
    adapter = build_adapter("openai", "gpt-4o-mini", temperature=0.2, base_url=None, client=client)

    assert isinstance(adapter, OpenAIAdapter)
    assert adapter.model == "gpt-4o-mini"
    assert adapter.temperature == 0.2
    assert adapter._client is client


def test_build_adapter_openai_uses_env_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "env-model")

    adapter = build_adapter("openai", None, temperature=0.0, base_url=None, client=object())

    assert adapter.model == "env-model"


def test_openai_compatible_requires_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="requires --base-url"):
        build_adapter("openai-compatible", "local", temperature=0.0, base_url=None, client=object())


def test_openai_compatible_accepts_injected_client() -> None:
    client = object()
    adapter = build_adapter(
        "openai-compatible",
        "local",
        temperature=0.1,
        base_url="http://localhost:8000/v1",
        client=client,
    )

    assert isinstance(adapter, OpenAICompatibleAdapter)
    assert adapter.base_url == "http://localhost:8000/v1"
    assert adapter._client is client


def test_build_masker_returns_strategy() -> None:
    assert isinstance(build_masker("placeholder"), PlaceholderMasker)
    assert isinstance(build_masker("drop"), DropMasker)
    assert isinstance(build_masker("filler"), FillerMasker)


def test_build_masker_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported masker"):
        build_masker("nope")


def test_build_sampler_converts_scale_to_repeats() -> None:
    sampler = build_sampler("leave-one-out", scale="standard")

    assert isinstance(sampler, LeaveOneOutSampler)
    assert sampler.estimate_evaluations(2) == 6


def test_build_sampler_supports_random() -> None:
    sampler = build_sampler("random", scale="quick")

    assert isinstance(sampler, RandomCoalitionSampler)
    # quick scale (repeat 1) -> 50 coalitions; full (repeat 5) -> 250.
    assert sampler.n_coalitions == 50
    assert build_sampler("random-coalition", scale="full").n_coalitions == 250


def test_build_sampler_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported sampler"):
        build_sampler("nope", scale="quick")


def test_build_adapter_enables_batch_api() -> None:
    openai = build_adapter(
        "openai", "gpt-4o-mini", temperature=0.0, base_url=None, use_batch_api=True, client=object()
    )
    anthropic = build_adapter(
        "anthropic",
        "claude-3-5-haiku-latest",
        temperature=0.0,
        base_url=None,
        use_batch_api=True,
        client=object(),
    )

    assert isinstance(openai, OpenAIAdapter)
    assert openai.use_batch_api is True
    assert isinstance(anthropic, AnthropicAdapter)
    assert anthropic.use_batch_api is True


def test_build_adapter_batch_api_defaults_off() -> None:
    adapter = build_adapter(
        "openai", "gpt-4o-mini", temperature=0.0, base_url=None, client=object()
    )

    assert isinstance(adapter, OpenAIAdapter)
    assert adapter.use_batch_api is False


def test_build_scorer_creates_correct_scorer_types(tmp_path) -> None:
    tool_config = tmp_path / "tool-scorer.json"
    tool_config.write_text(
        json.dumps({"expected_tool": "search", "required_args": ["query"]}),
        encoding="utf-8",
    )

    length = build_scorer("length")
    embedding = build_scorer("embedding-local")
    tool_call = build_scorer("tool-call", config_path=str(tool_config))

    assert isinstance(length, LengthDriftScorer)
    assert isinstance(embedding, EmbeddingScorer)
    assert isinstance(tool_call, ToolAccuracyScorer)
    assert tool_call.score(
        CompletionOutput(text=""),
        CompletionOutput(text="", tool_calls=[{"name": "search", "arguments": {"query": "docs"}}]),
    ) == 1.0


def test_build_scorer_embedding_defaults_to_local_huggingface() -> None:
    from promptlens.scorers import HuggingFaceEmbeddingClient

    scorer = build_scorer("embedding")

    assert isinstance(scorer, EmbeddingScorer)
    assert isinstance(scorer.embedding_client, HuggingFaceEmbeddingClient)
    # Constructing the scorer must not load the model; that happens lazily.
    assert scorer.embedding_client._encoder is None


def test_build_scorer_embedding_builds_openai_client(tmp_path) -> None:
    config = tmp_path / "embedding.json"
    config.write_text(
        json.dumps({"provider": "openai", "model": "text-embedding-3-small"}),
        encoding="utf-8",
    )

    scorer = build_scorer("embedding", config_path=str(config))

    assert isinstance(scorer, EmbeddingScorer)
    assert isinstance(scorer.embedding_client, OpenAIEmbeddingClient)
    assert scorer.embedding_client.model == "text-embedding-3-small"


def test_build_scorer_embedding_openai_compatible_requires_base_url(tmp_path) -> None:
    config = tmp_path / "embedding.json"
    config.write_text(json.dumps({"provider": "openai-compatible"}), encoding="utf-8")

    with pytest.raises(ValueError, match="base_url"):
        build_scorer("embedding", config_path=str(config))


def test_build_scorer_rejects_unknown_embedding_provider(tmp_path) -> None:
    config = tmp_path / "embedding.json"
    config.write_text(json.dumps({"provider": "mystery"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported embedding scorer provider"):
        build_scorer("embedding", config_path=str(config))


def test_tool_call_scorer_requires_config() -> None:
    with pytest.raises(ValueError, match="expected_tool"):
        build_scorer("tool-call")


def test_scorer_config_must_be_object(tmp_path) -> None:
    config = tmp_path / "bad.json"
    config.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        build_scorer("length", config_path=str(config))


def test_build_adapter_ollama_defaults_to_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    client = object()

    adapter = build_adapter("ollama", None, temperature=0.0, base_url=None, client=client)

    assert isinstance(adapter, OpenAICompatibleAdapter)
    assert adapter.model == "llama3.2"
    assert adapter.base_url == "http://localhost:11434/v1"
    assert adapter._client is client


def test_build_adapter_local_alias_normalizes_ollama_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.5:11434")

    adapter = build_adapter("local", None, temperature=0.0, base_url=None, client=object())

    assert isinstance(adapter, OpenAICompatibleAdapter)
    assert adapter.model == "qwen3"
    assert adapter.base_url == "http://10.0.0.5:11434/v1"


def test_build_scorer_tool_args_defaults() -> None:
    scorer = build_scorer("tool-args")

    assert isinstance(scorer, ToolArgumentDriftScorer)
    assert scorer.argument_weight == 0.5
    assert scorer.none_is_missing is True


def test_build_scorer_tool_args_reads_config(tmp_path) -> None:
    config = tmp_path / "tool-args.json"
    config.write_text(
        json.dumps(
            {
                "argument_weight": 1.0,
                "param_weights": {"account_id": 9, "reason": 0},
                "default_param_weight": 2,
                "none_is_missing": False,
            }
        ),
        encoding="utf-8",
    )

    scorer = build_scorer("tool-arguments", config_path=str(config))

    assert isinstance(scorer, ToolArgumentDriftScorer)
    assert scorer.argument_weight == 1.0
    assert scorer.param_weights == {"account_id": 9.0, "reason": 0.0}
    assert scorer.default_param_weight == 2.0
    assert scorer.none_is_missing is False


def test_build_scorer_tool_args_rejects_bad_config(tmp_path) -> None:
    config = tmp_path / "tool-args.json"
    config.write_text(json.dumps({"param_weights": {"q": "heavy"}}), encoding="utf-8")

    with pytest.raises(ValueError, match="param_weights"):
        build_scorer("tool-args", config_path=str(config))
