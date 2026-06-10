"""Result and cost data structures."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.table import Table
from rich.text import Text

from promptlens.core.base import Coalition, CompletionOutput, Feature

_MAX_BAR_WIDTH = 20


def _plain(text: str) -> Text:
    """Render model/prompt-derived text literally.

    Dynamic text flows from prompts and model outputs; rendered as a plain
    string, rich would parse bracketed tokens like ``[blocker]`` or
    ``[section_2]`` as console markup and silently delete them from tables.
    """
    return Text(text)


class CostEstimate(BaseModel):
    """Estimated spend before provider calls are made."""

    model_config = ConfigDict(frozen=True)

    model: str
    features: int
    evaluations: int
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    pricing_updated: str
    comparisons: dict[str, float] = Field(default_factory=dict)
    token_counter: str = "heuristic"

    @property
    def total_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["total_usd"] = self.total_usd
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def print(self) -> None:
        table = Table(title="CostEstimate")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("model", self.model)
        table.add_row("features", str(self.features))
        table.add_row("evaluations", str(self.evaluations))
        table.add_row("input tokens", f"{self.input_tokens} ({self.token_counter})")
        table.add_row("output tokens", str(self.output_tokens))
        table.add_row("input cost", f"${self.input_cost_usd:.6f}")
        table.add_row("output cost", f"${self.output_cost_usd:.6f}")
        table.add_row("total", f"${self.total_usd:.6f}")
        for model, total in self.comparisons.items():
            table.add_row(f"compare {model}", f"${total:.6f}")
        Console().print(table)


class CoalitionEvaluation(BaseModel):
    """A single masked-prompt evaluation."""

    model_config = ConfigDict(frozen=True)

    coalition: Coalition
    prompt: str
    output: CompletionOutput
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "coalition": list(self.coalition),
            "prompt": self.prompt,
            "output": {
                "text": self.output.text,
                "tool_calls": self.output.tool_calls,
                "logprobs": self.output.logprobs,
            },
            "score": self.score,
        }


class FeatureAttribution(BaseModel):
    """Attribution score for one feature."""

    model_config = ConfigDict(frozen=True)

    feature: Feature
    value: float
    stderr: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature.model_dump(),
            "value": self.value,
            "stderr": self.stderr,
        }


class SupplementaryEvaluation(BaseModel):
    """A non-attribution prompt variant evaluation."""

    model_config = ConfigDict(frozen=True)

    kind: str
    prompt: str
    output: CompletionOutput
    score: float
    feature: Feature | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "feature": self.feature.model_dump() if self.feature else None,
            "prompt": self.prompt,
            "output": {
                "text": self.output.text,
                "tool_calls": self.output.tool_calls,
                "logprobs": self.output.logprobs,
            },
            "score": self.score,
            "metadata": self.metadata,
        }


class Synopsis(BaseModel):
    """An LLM-written narrative summary of attribution evidence."""

    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "model": self.model, "metadata": self.metadata}


class AttributionResult(BaseModel):
    """Rich attribution output for SDK and CLI consumers."""

    model_config = ConfigDict(frozen=True)

    baseline_output: CompletionOutput
    attributions: list[FeatureAttribution]
    evaluations: list[CoalitionEvaluation]
    cost_estimate: CostEstimate | None = None
    supplementary_evaluations: list[SupplementaryEvaluation] = Field(default_factory=list)
    synopsis: Synopsis | None = None

    def with_synopsis(self, synopsis: Synopsis) -> AttributionResult:
        """Return a copy of this result with an attached synopsis."""
        return self.model_copy(update={"synopsis": synopsis})

    def drift_highlights(self, limit: int = 3) -> list[dict[str, Any]]:
        """Return the highest-drift coalition evaluations with the features they masked.

        Each entry names the features that were removed, the resulting score,
        and the output the model produced without them — the concrete "what
        actually changed" evidence behind the attribution numbers.
        """
        names = [attribution.feature.name for attribution in self.attributions]
        ordered = sorted(self.evaluations, key=lambda item: item.score, reverse=True)
        return [
            {
                "removed": [
                    names[index]
                    for index, included in enumerate(evaluation.coalition)
                    if not included and index < len(names)
                ],
                "score": evaluation.score,
                "output_text": evaluation.output.text,
            }
            for evaluation in ordered[:limit]
        ]

    def ranked(self) -> list[tuple[FeatureAttribution, float]]:
        """Return attributions sorted by importance with each one's normalized share.

        Shares are computed over the positive attribution mass so the visible
        weights sum to 1.0 (100%) and rank features by how much masking them
        moved the output.
        """
        total = sum(max(0.0, attribution.value) for attribution in self.attributions)
        ordered = sorted(self.attributions, key=lambda item: item.value, reverse=True)
        return [
            (attribution, (max(0.0, attribution.value) / total) if total > 0 else 0.0)
            for attribution in ordered
        ]

    def to_dict(self) -> dict[str, Any]:
        shares = {id(attribution): share for attribution, share in self.ranked()}
        return {
            "baseline_output": {
                "text": self.baseline_output.text,
                "tool_calls": self.baseline_output.tool_calls,
                "logprobs": self.baseline_output.logprobs,
            },
            "attributions": [
                {**attribution.to_dict(), "share": shares[id(attribution)]}
                for attribution in self.attributions
            ],
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
            "supplementary_evaluations": [
                evaluation.to_dict() for evaluation in self.supplementary_evaluations
            ],
            "cost_estimate": self.cost_estimate.to_dict() if self.cost_estimate else None,
            "drift_highlights": self.drift_highlights(),
            "synopsis": self.synopsis.to_dict() if self.synopsis else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def print(self) -> None:
        table = Table(title="promptlens Attribution")
        table.add_column("Feature")
        table.add_column("Value", justify="right")
        table.add_column("Share", justify="right")
        table.add_column("Weight")
        table.add_column("Text")
        for attribution, share in self.ranked():
            bar = "█" * round(share * _MAX_BAR_WIDTH)
            table.add_row(
                _plain(attribution.feature.name),
                f"{attribution.value:.4f}",
                f"{share * 100:.1f}%",
                bar,
                _plain(attribution.feature.text.replace("\n", " ")[:80]),
            )
        Console().print(table)
        highlights = self.drift_highlights()
        if highlights:
            highlight_table = Table(title="Largest output drifts")
            highlight_table.add_column("Removed features")
            highlight_table.add_column("Score", justify="right")
            highlight_table.add_column("Output without them")
            for highlight in highlights:
                highlight_table.add_row(
                    _plain(", ".join(highlight["removed"]) or "(none)"),
                    f"{highlight['score']:.4f}",
                    _plain(highlight["output_text"].replace("\n", " ")[:80]),
                )
            Console().print(highlight_table)
        if self.supplementary_evaluations:
            supplementary_table = Table(title="Supplementary prompt mutations")
            supplementary_table.add_column("Kind")
            supplementary_table.add_column("Feature")
            supplementary_table.add_column("Score", justify="right")
            supplementary_table.add_column("Prompt")
            for evaluation in self.supplementary_evaluations:
                supplementary_table.add_row(
                    _plain(evaluation.kind),
                    _plain(evaluation.feature.name if evaluation.feature else ""),
                    f"{evaluation.score:.4f}",
                    _plain(evaluation.prompt.replace("\n", " ")[:80]),
                )
            Console().print(supplementary_table)
        if self.synopsis:
            synopsis_table = Table(
                title=_plain(f"Synopsis ({self.synopsis.model})"), show_lines=True
            )
            synopsis_table.add_column("Summary")
            synopsis_table.add_row(_plain(self.synopsis.text))
            Console().print(synopsis_table)


class DrilldownRefinement(BaseModel):
    """Fine-grained attribution within one coarse feature of a drill-down run."""

    model_config = ConfigDict(frozen=True)

    feature: Feature
    result: AttributionResult

    def to_dict(self) -> dict[str, Any]:
        return {"feature": self.feature.model_dump(), "result": self.result.to_dict()}


class DrilldownResult(BaseModel):
    """Coarse-to-fine attribution: a section overview plus refined hot spots.

    Produced by :func:`promptlens.core.drilldown.explain_drilldown`. The
    ``overview`` attributes the prompt at coarse granularity (sections or
    paragraphs); each refinement re-attributes the sentences of one
    high-attribution coarse feature while the rest of the prompt stays intact.
    ``provider_calls_used`` versus ``flat_sweep_provider_calls`` shows what the
    two-stage pass saved over masking every sentence of the prompt one at a
    time.
    """

    model_config = ConfigDict(frozen=True)

    overview: AttributionResult
    refinements: list[DrilldownRefinement]
    provider_calls_used: int
    flat_sweep_provider_calls: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "overview": self.overview.to_dict(),
            "refinements": [refinement.to_dict() for refinement in self.refinements],
            "provider_calls_used": self.provider_calls_used,
            "flat_sweep_provider_calls": self.flat_sweep_provider_calls,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def print(self) -> None:
        self.overview.print()
        for refinement in self.refinements:
            table = Table(title=_plain(f"Refined: {refinement.feature.name}"))
            table.add_column("Feature")
            table.add_column("Value", justify="right")
            table.add_column("Share", justify="right")
            table.add_column("Weight")
            table.add_column("Text")
            for attribution, share in refinement.result.ranked():
                bar = "█" * round(share * _MAX_BAR_WIDTH)
                table.add_row(
                    _plain(attribution.feature.name),
                    f"{attribution.value:.4f}",
                    f"{share * 100:.1f}%",
                    bar,
                    _plain(attribution.feature.text.replace("\n", " ")[:80]),
                )
            Console().print(table)
        Console().print(
            f"Drill-down used {self.provider_calls_used} provider calls vs "
            f"~{self.flat_sweep_provider_calls} for a flat sentence sweep."
        )


class PerQuestionAttribution(BaseModel):
    """Per-question attribution over a fixed multi-question task.

    Produced by :func:`promptlens.adapters.explain_per_question`. Each question
    in the task gets its own complete :class:`AttributionResult`, so a feature's
    importance can be compared *across* questions — "this instruction carries
    refund questions but is dead weight for shipping questions". Per-question
    scoring is statistically well-posed because the questions are fixed inputs
    rather than model-generated turns: answer *i* always corresponds to
    question *i*, in every coalition.
    """

    model_config = ConfigDict(frozen=True)

    questions: list[str]
    results: list[AttributionResult]

    def share_matrix(self) -> dict[str, list[float]]:
        """Map each feature name to its normalized share per question."""
        matrix: dict[str, list[float]] = {}
        for column, result in enumerate(self.results):
            for attribution, share in result.ranked():
                row = matrix.setdefault(
                    attribution.feature.name, [0.0] * len(self.results)
                )
                row[column] = share
        return matrix

    def to_dict(self) -> dict[str, Any]:
        return {
            "questions": self.questions,
            "share_matrix": self.share_matrix(),
            "results": [result.to_dict() for result in self.results],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def print(self) -> None:
        table = Table(title="promptlens Attribution by question")
        table.add_column("Feature")
        for question in self.questions:
            label = question.replace("\n", " ")
            table.add_column(
                _plain(label[:40] + ("…" if len(label) > 40 else "")), justify="right"
            )
        matrix = self.share_matrix()
        ordered = sorted(
            matrix.items(), key=lambda item: max(item[1], default=0.0), reverse=True
        )
        for feature_name, shares in ordered:
            table.add_row(_plain(feature_name), *[f"{share * 100:.1f}%" for share in shares])
        Console().print(table)


class OptimizationResult(BaseModel):
    """An LLM-proposed prompt rewrite derived from attribution evidence."""

    model_config = ConfigDict(frozen=True)

    original_prompt: str
    proposed_prompt: str
    rationale: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "proposed_prompt": self.proposed_prompt,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def print(self) -> None:
        table = Table(title="promptlens Optimization", show_lines=True)
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("original prompt", _plain(self.original_prompt))
        table.add_row("proposed prompt", _plain(self.proposed_prompt))
        if self.rationale:
            table.add_row("rationale", _plain(self.rationale))
        Console().print(table)
