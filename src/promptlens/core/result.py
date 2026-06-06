"""Result and cost data structures."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.table import Table

from promptlens.core.base import Coalition, CompletionOutput, Feature

_MAX_BAR_WIDTH = 20


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
        table.add_row("input tokens", str(self.input_tokens))
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


class AttributionResult(BaseModel):
    """Rich attribution output for SDK and CLI consumers."""

    model_config = ConfigDict(frozen=True)

    baseline_output: CompletionOutput
    attributions: list[FeatureAttribution]
    evaluations: list[CoalitionEvaluation]
    cost_estimate: CostEstimate | None = None
    supplementary_evaluations: list[SupplementaryEvaluation] = Field(default_factory=list)

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
                attribution.feature.name,
                f"{attribution.value:.4f}",
                f"{share * 100:.1f}%",
                bar,
                attribution.feature.text.replace("\n", " ")[:80],
            )
        Console().print(table)
        if self.supplementary_evaluations:
            supplementary_table = Table(title="Supplementary prompt mutations")
            supplementary_table.add_column("Kind")
            supplementary_table.add_column("Feature")
            supplementary_table.add_column("Score", justify="right")
            supplementary_table.add_column("Prompt")
            for evaluation in self.supplementary_evaluations:
                supplementary_table.add_row(
                    evaluation.kind,
                    evaluation.feature.name if evaluation.feature else "",
                    f"{evaluation.score:.4f}",
                    evaluation.prompt.replace("\n", " ")[:80],
                )
            Console().print(supplementary_table)


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
        table.add_row("original prompt", self.original_prompt)
        table.add_row("proposed prompt", self.proposed_prompt)
        if self.rationale:
            table.add_row("rationale", self.rationale)
        Console().print(table)
