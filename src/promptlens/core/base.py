"""Core interfaces for promptlens attribution components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from promptlens.core.tools import (
    Tool,
    ToolDefinitions,
    ToolLike,
    ToolParameter,
    coerce_tools,
    normalize_tool,
    tool,
)

Coalition = tuple[bool, ...]


class Feature(BaseModel):
    """An attributable prompt or tool feature."""

    model_config = ConfigDict(frozen=True)

    name: str
    text: str
    start: int | None = None
    end: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompletionOutput(BaseModel):
    """Normalized model output returned by adapters."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    text: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    logprobs: list[float] | None = None
    raw: Any | None = None


class PromptMutation(BaseModel):
    """A supplementary prompt variant generated outside attribution scoring."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    feature: Feature | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    """Collapses model outputs into a scalar attribution signal.

    Scorers come in two orientations, declared via the ``orientation`` class
    attribute, because "a higher score" means opposite things for each:

    * ``"drift"`` (default): higher means the candidate output moved *further*
      from the baseline. The score is already an attribution signal, so the
      harness uses it directly: masking an influential feature produces a large
      drift.
    * ``"objective"``: higher means the candidate did the desired thing *better*
      (e.g. selected the expected tool). These scorers measure task quality and
      typically ignore the baseline, so a raw value is **not** a drift signal.
      The harness converts it to attribution by measuring how far the objective
      *drops* when a feature is masked, relative to the baseline objective.

    Keeping the two orientations distinct avoids the conceptual error of treating
    "the masked prompt still did the right thing" as "this feature mattered a lot".
    """

    orientation: str = "drift"

    @abstractmethod
    def score(self, baseline: CompletionOutput, candidate: CompletionOutput) -> float:
        """Return a scalar score.

        For ``orientation == "drift"`` scorers, larger means the candidate output
        drifted more from ``baseline``. For ``orientation == "objective"`` scorers,
        larger means ``candidate`` better achieved the task objective.
        """


class Sampler(ABC):
    """Generates coalition masks to evaluate."""

    @abstractmethod
    def sample(self, n_features: int) -> Iterator[Coalition]:
        """Yield binary coalition masks to evaluate."""

    @abstractmethod
    def estimate_evaluations(self, n_features: int) -> int:
        """Return the expected number of non-baseline evaluations."""


class PromptMutator(ABC):
    """Generates supplementary prompt variants for robustness analysis."""

    @abstractmethod
    def mutate(
        self,
        prompt: str,
        features: Sequence[Feature],
        tools: ToolDefinitions | None = None,
    ) -> list[PromptMutation]:
        """Return prompt variants to evaluate alongside attribution results."""


class PromptOptimizer(ABC):
    """Proposes an improved prompt from completed attribution evidence."""

    @abstractmethod
    def optimize(self, prompt: str, result: Any) -> Any:
        """Return an OptimizationResult proposing a rewrite of ``prompt``.

        ``result`` is an :class:`~promptlens.core.result.AttributionResult`; it is
        typed as ``Any`` here to avoid a circular import between base and result.
        """


def normalize_coalition(coalition: Iterable[bool], n_features: int) -> Coalition:
    """Validate and normalize a coalition iterable."""
    normalized = tuple(bool(value) for value in coalition)
    if len(normalized) != n_features:
        msg = f"Expected coalition length {n_features}, got {len(normalized)}"
        raise ValueError(msg)
    return normalized


__all__ = [
    "Adapter",
    "Coalition",
    "CompletionOutput",
    "Feature",
    "Masker",
    "PromptMutation",
    "PromptMutator",
    "PromptOptimizer",
    "Sampler",
    "Scorer",
    "Segmenter",
    "Tool",
    "ToolDefinitions",
    "ToolLike",
    "ToolParameter",
    "coerce_tools",
    "normalize_coalition",
    "normalize_tool",
    "tool",
]
