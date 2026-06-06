from promptlens.core import CompletionOutput
from promptlens.scorers import (
    CompositeScorer,
    EmbeddingScorer,
    LengthDriftScorer,
    OpenAIEmbeddingClient,
    ToolAccuracyScorer,
    cosine_distance,
)


class _StubEmbeddings:
    """Minimal stand-in for the OpenAI embeddings resource."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, str]] = []

    def create(self, *, model: str, input: str):  # noqa: A002 - mirrors SDK kwarg
        self.calls.append((model, input))
        vector = self.vectors[input]
        data = [type("Item", (), {"embedding": vector})()]
        return type("Response", (), {"data": data})()


class _StubClient:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.embeddings = _StubEmbeddings(vectors)



def test_cosine_distance_identical_vectors_is_zero() -> None:
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == 0.0


def test_tool_accuracy_scores_required_args() -> None:
    scorer = ToolAccuracyScorer(expected_tool="search", required_args=["query", "limit"])
    output = CompletionOutput(
        text="",
        tool_calls=[{"name": "search", "arguments": {"query": "docs", "limit": 5}}],
    )

    assert scorer.score(CompletionOutput(text=""), output) == 1.0


def test_composite_scorer_weights_components() -> None:
    baseline = CompletionOutput(text="aaaa")
    candidate = CompletionOutput(text="aa")
    drift = LengthDriftScorer().score(baseline, candidate)

    composite = CompositeScorer([(LengthDriftScorer(), 0.25), (LengthDriftScorer(), 0.75)])

    assert composite.score(baseline, candidate) == drift


def test_composite_scorer_sums_distinct_scorers() -> None:
    class ConstantScorer(LengthDriftScorer):
        def __init__(self, value: float) -> None:
            self.value = value

        def score(self, baseline: CompletionOutput, candidate: CompletionOutput) -> float:
            return self.value

    composite = CompositeScorer([(ConstantScorer(2.0), 0.5), (ConstantScorer(10.0), 0.1)])

    # 0.5 * 2.0 + 0.1 * 10.0 == 2.0
    assert composite.score(CompletionOutput(text=""), CompletionOutput(text="")) == 2.0


def test_composite_scorer_requires_components() -> None:
    import pytest

    with pytest.raises(ValueError, match="at least one"):
        CompositeScorer([])


def test_openai_embedding_client_uses_injected_client() -> None:
    client = _StubClient({"hello": [1.0, 0.0], "world": [0.0, 1.0]})
    embed_client = OpenAIEmbeddingClient(model="text-embedding-3-small", client=client)

    assert list(embed_client.embed("hello")) == [1.0, 0.0]
    assert client.embeddings.calls == [("text-embedding-3-small", "hello")]


def test_embedding_scorer_uses_provider_embeddings() -> None:
    client = _StubClient({"baseline text": [1.0, 0.0], "drifted text": [0.0, 1.0]})
    scorer = EmbeddingScorer(OpenAIEmbeddingClient(client=client))

    distance = scorer.score(
        CompletionOutput(text="baseline text"),
        CompletionOutput(text="drifted text"),
    )

    # Orthogonal vectors -> cosine similarity 0 -> distance 1.0.
    assert distance == 1.0


