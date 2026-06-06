"""Tool schema segmenter for OpenAI, Anthropic, and compatible formats."""

from __future__ import annotations

from typing import Literal

from promptlens.core.base import Feature, Segmenter, ToolDefinitions, ToolLike, normalize_tool

Granularity = Literal["tool", "field", "parameter"]


class ToolSegmenter(Segmenter):
    """Segment tool definitions for docstring and parameter attribution."""

    def __init__(self, granularity: Granularity = "tool") -> None:
        self.granularity = granularity

    def segment(self, prompt: str, tools: ToolDefinitions | None = None) -> list[Feature]:
        features = [Feature(name="prompt", text=prompt, start=0, end=len(prompt))] if prompt else []
        for tool_index, tool in enumerate(tools or []):
            features.extend(self._segment_tool(tool_index, tool))
        return features

    def _segment_tool(self, tool_index: int, tool: ToolLike) -> list[Feature]:
        normalized = normalize_tool(tool)
        tool_name = str(normalized.get("name", f"tool_{tool_index + 1}"))
        if self.granularity == "tool":
            return [
                Feature(
                    name=f"tool:{tool_name}",
                    text=str(tool),
                    metadata={"kind": "tool", "tool": tool},
                )
            ]
        features = [
            Feature(
                name=f"tool:{tool_name}:description",
                text=str(normalized.get("description", "")),
                metadata={"kind": "tool_description", "tool": tool_name},
            )
        ]
        parameters = normalized.get("parameters", {})
        if self.granularity == "field":
            features.append(
                Feature(
                    name=f"tool:{tool_name}:parameters",
                    text=str(parameters),
                    metadata={"kind": "tool_parameters", "tool": tool_name},
                )
            )
            return features
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        for parameter_name, schema in properties.items():
            features.append(
                Feature(
                    name=f"tool:{tool_name}:parameter:{parameter_name}",
                    text=str(schema),
                    metadata={
                        "kind": "tool_parameter",
                        "tool": tool_name,
                        "parameter": parameter_name,
                    },
                )
            )
        return features
