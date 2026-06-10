import threading

from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions


class _BarrierAdapter(Adapter):
    """Completes only when two calls are in flight simultaneously."""

    def __init__(self, parties: int) -> None:
        self.model = "barrier"
        self.barrier = threading.Barrier(parties, timeout=5)
        self.max_concurrency = parties

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        self.barrier.wait()  # deadlocks (then times out) if calls were serial
        return CompletionOutput(text=prompt)


class _RecordingAdapter(Adapter):
    def __init__(self, max_concurrency: int) -> None:
        self.model = "recording"
        self.max_concurrency = max_concurrency
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        with self._lock:
            self.calls.append(prompt)
        return CompletionOutput(text=prompt.upper())


def test_batch_runs_concurrently_when_enabled() -> None:
    adapter = _BarrierAdapter(parties=2)
    outputs = adapter.complete_batch(["a", "b"])
    assert [output.text for output in outputs] == ["a", "b"]


def test_batch_preserves_prompt_order() -> None:
    adapter = _RecordingAdapter(max_concurrency=4)
    prompts = [f"p{i}" for i in range(10)]
    outputs = adapter.complete_batch(prompts)
    assert [output.text for output in outputs] == [p.upper() for p in prompts]
    assert sorted(adapter.calls) == sorted(prompts)


def test_batch_stays_serial_by_default() -> None:
    adapter = _RecordingAdapter(max_concurrency=1)
    outputs = adapter.complete_batch(["x", "y"])
    assert [output.text for output in outputs] == ["X", "Y"]


def test_repeated_prompts_are_not_deduplicated() -> None:
    # samples_per_coalition intentionally re-sends identical prompts to sample
    # a non-deterministic provider's distribution; dedupe would break that.
    adapter = _RecordingAdapter(max_concurrency=4)
    outputs = adapter.complete_batch(["same", "same", "same"])
    assert len(outputs) == 3
    assert adapter.calls.count("same") == 3
