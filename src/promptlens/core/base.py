"""Core interfaces for promptlens attribution components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

Coalition = tuple[bool, ...]
ToolDefinitions = list[dict[str, Any]]


@dataclass(frozen=True)
class Feature:
    """An attributable prompt or tool feature."""

    name: str
    text: str
    start: int | None = None
    end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompletionOutput:
    """Normalized model output returned by adapters."""

    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    logprobs: list[float] | None = None
    raw: Any | None = None


class Adapter(ABC):
    """Thin provider wrapper for a model completion API."""

    model: str

    @abstractmethod
    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        """Return text output and optional provider-specific metadata."""

    def complete_batch(
        self, prompts: Sequence[str], tools: ToolDefinitions | None = None
    ) -> list[CompletionOutput]:
        """Default batch path; provider adapters may override with native batching."""
        return [self.complete(prompt, tools=tools) for prompt in prompts]


class Segmenter(ABC):
    """Splits a prompt into attributable features."""

    @abstractmethod
    def segment(self, prompt: str, tools: ToolDefinitions | None = None) -> list[Feature]:
        """Return named, ordered features with optional span metadata."""


class Masker(ABC):
    """Reconstructs prompts from a coalition mask."""

    @abstractmethod
    def mask(self, features: Sequence[Feature], coalition: Coalition) -> str:
        """Return a prompt where features with coalition=False are masked."""


class Scorer(ABC):
    """Collapses model outputs into a scalar attribution signal."""

    @abstractmethod
    def score(self, baseline: CompletionOutput, candidate: CompletionOutput) -> float:
        """Return a scalar score. Larger means the masked output drifted more."""


class Sampler(ABC):
    """Generates coalition masks to evaluate."""

    @abstractmethod
    def sample(self, n_features: int) -> Iterator[Coalition]:
        """Yield binary coalition masks to evaluate."""

    @abstractmethod
    def estimate_evaluations(self, n_features: int) -> int:
        """Return the expected number of non-baseline evaluations."""


def normalize_coalition(coalition: Iterable[bool], n_features: int) -> Coalition:
    """Validate and normalize a coalition iterable."""
    normalized = tuple(bool(value) for value in coalition)
    if len(normalized) != n_features:
        msg = f"Expected coalition length {n_features}, got {len(normalized)}"
        raise ValueError(msg)
    return normalized
