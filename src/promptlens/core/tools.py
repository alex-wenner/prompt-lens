"""User-facing tool definitions and per-provider coercion.

Users describe a tool once, in provider-neutral terms, either by constructing a
:class:`Tool` directly or by decorating a Python function with :func:`tool`.
Adapters then coerce that single definition into the schema their provider
expects (OpenAI ``function`` blocks, Anthropic ``input_schema``, Bedrock
``toolSpec``, Gemini ``function_declarations``). Passing a raw ``dict`` is still
supported as an escape hatch and is forwarded to the provider unchanged.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any, get_args, get_type_hints

from pydantic import BaseModel, ConfigDict, Field

# JSON-Schema type names keyed by the Python annotation they map to.
_PYTHON_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class ToolParameter(BaseModel):
    """A single tool parameter, described in provider-neutral terms."""

    model_config = ConfigDict(frozen=True)

    type: str = "string"
    description: str = ""
    enum: list[Any] | None = None
    items: dict[str, Any] | None = None
    required: bool = True

    def to_schema(self) -> dict[str, Any]:
        """Return the JSON-Schema fragment describing this parameter."""
        schema: dict[str, Any] = {"type": self.type}
        if self.description:
            schema["description"] = self.description
        if self.enum is not None:
            schema["enum"] = list(self.enum)
        if self.items is not None:
            schema["items"] = dict(self.items)
        return schema


class Tool(BaseModel):
    """A provider-neutral tool definition.

    Construct one directly or via the :func:`tool` decorator, then let an adapter
    coerce it into the schema its provider expects via :func:`coerce_tools`.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    parameters: dict[str, ToolParameter] = Field(default_factory=dict)

    def json_schema(self) -> dict[str, Any]:
        """Return the JSON-Schema ``object`` describing this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                name: parameter.to_schema() for name, parameter in self.parameters.items()
            },
            "required": [
                name for name, parameter in self.parameters.items() if parameter.required
            ],
        }

    def to_openai(self) -> dict[str, Any]:
        """Coerce into an OpenAI (and OpenAI-compatible) ``function`` tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema(),
            },
        }

    def to_anthropic(self) -> dict[str, Any]:
        """Coerce into an Anthropic Messages API tool."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.json_schema(),
        }

    def to_bedrock(self) -> dict[str, Any]:
        """Coerce into an Amazon Bedrock Converse ``toolSpec``."""
        return {
            "toolSpec": {
                "name": self.name,
                "description": self.description,
                "inputSchema": {"json": self.json_schema()},
            }
        }

    def to_gemini(self) -> dict[str, Any]:
        """Coerce into a Google Gemini ``function_declarations`` entry."""
        return {
            "function_declarations": [
                {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.json_schema(),
                }
            ]
        }


# A tool may be supplied as a structured :class:`Tool` or, as an escape hatch, as
# a raw provider-shaped mapping that is forwarded unchanged.
ToolLike = Tool | dict[str, Any]
ToolDefinitions = list[ToolLike]

# Maps a provider key to the :class:`Tool` coercion method that produces its shape.
_PROVIDER_COERCERS: dict[str, str] = {
    "openai": "to_openai",
    "openai-compatible": "to_openai",
    "grok": "to_openai",
    "anthropic": "to_anthropic",
    "bedrock": "to_bedrock",
    "gemini": "to_gemini",
}


def coerce_tools(
    tools: Sequence[ToolLike] | None, provider: str
) -> list[dict[str, Any]] | None:
    """Coerce user tool definitions into ``provider``'s expected schema.

    :class:`Tool` instances are converted via their matching ``to_*`` method;
    raw ``dict`` definitions are passed through unchanged so callers can still
    hand a provider exactly the payload they want.
    """
    if tools is None:
        return None
    method = _PROVIDER_COERCERS.get(provider, "to_openai")
    coerced: list[dict[str, Any]] = []
    for definition in tools:
        if isinstance(definition, Tool):
            coercer: Callable[[], dict[str, Any]] = getattr(definition, method)
            coerced.append(coercer())
        else:
            coerced.append(definition)
    return coerced


def normalize_tool(definition: ToolLike) -> dict[str, Any]:
    """Return ``{name, description, parameters}`` for segmentation of any tool shape.

    Handles structured :class:`Tool` instances as well as the OpenAI, Anthropic,
    and bare mapping shapes that may be supplied as raw dicts.
    """
    if isinstance(definition, Tool):
        return {
            "name": definition.name,
            "description": definition.description,
            "parameters": definition.json_schema(),
        }
    if "function" in definition and isinstance(definition["function"], dict):
        return dict(definition["function"])
    input_schema = definition.get("input_schema")
    if input_schema is not None:
        return {
            "name": definition.get("name"),
            "description": definition.get("description", ""),
            "parameters": input_schema,
        }
    return dict(definition)


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Build a :class:`Tool` from a function's signature, hints, and docstring.

    Use it as ``@tool`` or ``@tool(name=..., description=...)``. Parameter
    descriptions are read from :data:`typing.Annotated` metadata, for example
    ``order_reference: Annotated[str, "The customer's order ID"]``.
    """

    def decorate(target: Callable[..., Any]) -> Tool:
        return _tool_from_function(target, name=name, description=description)

    if func is not None:
        return decorate(func)
    return decorate


def _tool_from_function(
    func: Callable[..., Any], *, name: str | None, description: str | None
) -> Tool:
    signature = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)
    parameters: dict[str, ToolParameter] = {}
    for parameter_name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(parameter_name, str)
        base, parameter_description = _split_annotation(annotation)
        parameters[parameter_name] = ToolParameter(
            type=_PYTHON_TO_JSON_TYPE.get(base, "string"),
            description=parameter_description,
            required=parameter.default is inspect.Parameter.empty,
        )
    doc = description if description is not None else (inspect.getdoc(func) or "")
    return Tool(name=name or func.__name__, description=doc.strip(), parameters=parameters)


def _split_annotation(annotation: Any) -> tuple[Any, str]:
    """Split an annotation into its base type and any ``Annotated`` string description."""
    if hasattr(annotation, "__metadata__"):
        args = get_args(annotation)
        base = args[0] if args else str
        description = next((item for item in args[1:] if isinstance(item, str)), "")
        return base, description
    return annotation, ""


__all__ = [
    "Tool",
    "ToolDefinitions",
    "ToolLike",
    "ToolParameter",
    "coerce_tools",
    "normalize_tool",
    "tool",
]
