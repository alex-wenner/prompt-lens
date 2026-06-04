"""Result and cost data structures."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

from promptlens.core.base import Coalition, CompletionOutput, Feature


@dataclass(frozen=True)
class CostEstimate:
    """Estimated spend before provider calls are made."""

    model: str
    features: int
    evaluations: int
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    pricing_updated: str
    comparisons: dict[str, float] = field(default_factory=dict)

    @property
    def total_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
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


@dataclass(frozen=True)
class CoalitionEvaluation:
    """A single masked-prompt evaluation."""

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


@dataclass(frozen=True)
class FeatureAttribution:
    """Attribution score for one feature."""

    feature: Feature
    value: float
    stderr: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": asdict(self.feature),
            "value": self.value,
            "stderr": self.stderr,
        }


@dataclass(frozen=True)
class AttributionResult:
    """Rich attribution output for SDK and CLI consumers."""

    baseline_output: CompletionOutput
    attributions: list[FeatureAttribution]
    evaluations: list[CoalitionEvaluation]
    cost_estimate: CostEstimate | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_output": {
                "text": self.baseline_output.text,
                "tool_calls": self.baseline_output.tool_calls,
                "logprobs": self.baseline_output.logprobs,
            },
            "attributions": [attribution.to_dict() for attribution in self.attributions],
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
            "cost_estimate": self.cost_estimate.to_dict() if self.cost_estimate else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def print(self) -> None:
        table = Table(title="promptlens Attribution")
        table.add_column("Feature")
        table.add_column("Value", justify="right")
        table.add_column("Text")
        for attribution in self.attributions:
            table.add_row(
                attribution.feature.name,
                f"{attribution.value:.4f}",
                attribution.feature.text.replace("\n", " ")[:80],
            )
        Console().print(table)
