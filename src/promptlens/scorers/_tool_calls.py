"""Shared helpers for reading normalized tool calls across provider shapes."""

from __future__ import annotations

import json
from typing import Any


def tool_call_name(call: dict[str, Any]) -> str:
    """Read a tool call's name from Anthropic/Bedrock or OpenAI function shapes."""
    return str(call.get("name") or call.get("function", {}).get("name") or "")


def tool_call_arguments(call: dict[str, Any]) -> Any:
    """Read a tool call's arguments, decoding OpenAI's JSON-string encoding."""
    raw = call.get("arguments")
    if raw is None:
        raw = call.get("input")
    if raw is None:
        raw = call.get("function", {}).get("arguments")
    return parse_arguments(raw)


def parse_arguments(arguments: Any) -> Any:
    """Decode tool-call arguments, which OpenAI delivers as a JSON string."""
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments
