import json

import pytest

from promptlens.adapters import EchoAdapter, OpenAIAdapter, OpenAICompatibleAdapter
from promptlens.cli.factories import build_adapter, build_sampler, build_scorer
from promptlens.core import CompletionOutput
from promptlens.samplers import LeaveOneOutSampler
from promptlens.scorers import EmbeddingScorer, LengthDriftScorer, ToolAccuracyScorer


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


def test_build_sampler_converts_scale_to_repeats() -> None:
    sampler = build_sampler("leave-one-out", scale="standard")

    assert isinstance(sampler, LeaveOneOutSampler)
    assert sampler.estimate_evaluations(2) == 6


def test_build_scorer_creates_correct_scorer_types(tmp_path) -> None:
    tool_config = tmp_path / "tool-scorer.json"
    tool_config.write_text(
        json.dumps({"expected_tool": "search", "required_args": ["query"]}),
        encoding="utf-8",
    )

    length = build_scorer("length")
    embedding = build_scorer("embedding")
    tool_call = build_scorer("tool-call", config_path=str(tool_config))

    assert isinstance(length, LengthDriftScorer)
    assert isinstance(embedding, EmbeddingScorer)
    assert isinstance(tool_call, ToolAccuracyScorer)
    assert tool_call.score(
        CompletionOutput(text=""),
        CompletionOutput(text="", tool_calls=[{"name": "search", "arguments": {"query": "docs"}}]),
    ) == 1.0


def test_tool_call_scorer_requires_config() -> None:
    with pytest.raises(ValueError, match="expected_tool"):
        build_scorer("tool-call")


def test_scorer_config_must_be_object(tmp_path) -> None:
    config = tmp_path / "bad.json"
    config.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        build_scorer("length", config_path=str(config))
