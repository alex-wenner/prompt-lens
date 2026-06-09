import sys
from types import SimpleNamespace

from promptlens import AttributionHarness
from promptlens.adapters import EchoAdapter
from promptlens.core import pricing
from promptlens.core.pricing import count_tokens, estimate_cost
from promptlens.maskers import DropMasker
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter


def test_heuristic_used_for_claude_models() -> None:
    # tiktoken is OpenAI's tokenizer and undercounts Claude tokens, so Claude
    # estimates must stay on the conservative heuristic even when installed.
    count, counter = count_tokens("hello world", "anthropic/claude-opus-4-8")
    assert counter == "heuristic"
    assert count == len("hello world") // 4 + 1


def test_heuristic_used_for_unknown_models() -> None:
    _, counter = count_tokens("hello", "some-local-model")
    assert counter == "heuristic"


def test_tiktoken_used_for_openai_models_when_installed(monkeypatch) -> None:
    class _FakeEncoding:
        def encode(self, text: str) -> list[int]:
            return [0] * len(text.split())

    fake_tiktoken = SimpleNamespace(
        encoding_for_model=lambda name: _FakeEncoding(),
        get_encoding=lambda name: _FakeEncoding(),
    )
    monkeypatch.setitem(sys.modules, "tiktoken", fake_tiktoken)
    pricing._tiktoken_encoding.cache_clear()
    try:
        count, counter = count_tokens("one two three", "openai/gpt-4o-mini")
    finally:
        pricing._tiktoken_encoding.cache_clear()

    assert counter == "tiktoken"
    assert count == 3


def test_estimate_counts_each_masked_prompt() -> None:
    prompts = ["aaaa bbbb", "aaaa", "bbbb"]
    estimate = estimate_cost(
        model="openai-compatible/local",
        prompt=prompts[0],
        features=2,
        evaluations=2,
        expected_output_tokens=10,
        evaluation_prompts=prompts[1:],
    )

    expected = sum((len(p) + 3) // 4 for p in prompts)
    assert estimate.input_tokens == expected
    assert estimate.output_tokens == 10 * 3  # baseline + two evaluations
    assert estimate.token_counter == "heuristic"


def test_estimate_scales_with_samples_per_coalition() -> None:
    estimate = estimate_cost(
        model="openai-compatible/local",
        prompt="aaaa bbbb",
        features=2,
        evaluations=4,  # two masked prompts, two samples each
        evaluation_prompts=["aaaa", "bbbb"],
    )

    per_sweep = ((len("aaaa") + 3) // 4) * 2
    baseline = (len("aaaa bbbb") + 3) // 4
    assert estimate.input_tokens == baseline + per_sweep * 2


def test_harness_estimate_reflects_masker_choice() -> None:
    prompt = "Alpha sentence here. Beta sentence here."
    drop = AttributionHarness(
        adapter=EchoAdapter(),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
        masker=DropMasker(),
    ).estimate(prompt)
    placeholder = AttributionHarness(
        adapter=EchoAdapter(),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    ).estimate(prompt)

    # Dropping features outright sends shorter prompts than placeholder masking.
    assert drop.input_tokens < placeholder.input_tokens


class _CountingClient:
    """Fake Anthropic client exposing the count_tokens metering endpoint."""

    def __init__(self) -> None:
        self.counted: list[str] = []
        self.messages = SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(content=[]),
            count_tokens=self._count_tokens,
        )

    def _count_tokens(self, *, model: str, messages: list, **kwargs) -> SimpleNamespace:
        text = messages[0]["content"]
        self.counted.append(text)
        return SimpleNamespace(input_tokens=len(text.split()))


def test_exact_tokens_uses_anthropic_count_tokens_endpoint() -> None:
    from promptlens.adapters import AnthropicAdapter

    client = _CountingClient()
    harness = AttributionHarness(
        adapter=AnthropicAdapter(model="claude-sonnet-4-6", client=client),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )

    estimate = harness.estimate("alpha one two. beta three.", exact_tokens=True)

    assert estimate.token_counter == "provider"
    # Baseline prompt plus one masked prompt per feature, each counted exactly.
    assert len(client.counted) == 3
    assert estimate.input_tokens == sum(len(text.split()) for text in client.counted)


def test_exact_tokens_falls_back_for_adapters_without_counter() -> None:
    harness = AttributionHarness(
        adapter=EchoAdapter(),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )

    estimate = harness.estimate("alpha one. beta two.", exact_tokens=True)

    assert estimate.token_counter == "heuristic"
