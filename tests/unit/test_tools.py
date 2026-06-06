from typing import Annotated

from promptlens import Tool, ToolParameter, coerce_tools, tool
from promptlens.segmenters import ToolSegmenter


def test_tool_decorator_builds_definition_from_signature() -> None:
    @tool
    def lookup_order(
        order_reference: Annotated[str, "Identifier for the customer's existing purchase."],
        include_history: bool = False,
    ) -> str:
        """Look up the status of an existing customer order."""

    assert isinstance(lookup_order, Tool)
    assert lookup_order.name == "lookup_order"
    assert lookup_order.description == "Look up the status of an existing customer order."
    order_reference = lookup_order.parameters["order_reference"]
    assert order_reference.type == "string"
    assert order_reference.description == "Identifier for the customer's existing purchase."
    assert order_reference.required is True
    # Defaulted parameters are optional and excluded from the required list.
    assert lookup_order.parameters["include_history"].required is False
    assert lookup_order.json_schema()["required"] == ["order_reference"]


def test_tool_decorator_accepts_overrides() -> None:
    @tool(name="search", description="Find products")
    def search_catalog(query: str) -> str:
        """Ignored when description is provided."""

    assert search_catalog.name == "search"
    assert search_catalog.description == "Find products"
    assert search_catalog.parameters["query"].type == "string"


def test_coerce_tools_maps_each_provider_shape() -> None:
    search = Tool(
        name="search",
        description="Search the web",
        parameters={"query": ToolParameter(description="Search query")},
    )

    assert coerce_tools([search], "openai") == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required": ["query"],
                },
            },
        }
    ]
    assert coerce_tools([search], "anthropic")[0]["input_schema"]["properties"] == {
        "query": {"type": "string", "description": "Search query"}
    }
    assert coerce_tools([search], "bedrock")[0]["toolSpec"]["name"] == "search"
    assert coerce_tools([search], "gemini")[0]["function_declarations"][0]["name"] == "search"
    # Grok rides the OpenAI-compatible function shape.
    assert coerce_tools([search], "grok") == coerce_tools([search], "openai")


def test_coerce_tools_passes_raw_dicts_through_unchanged() -> None:
    raw = {"type": "function", "function": {"name": "custom"}}
    assert coerce_tools([raw], "anthropic") == [raw]


def test_coerce_tools_none_is_none() -> None:
    assert coerce_tools(None, "openai") is None


def test_tool_segmenter_handles_structured_tool() -> None:
    search = Tool(
        name="search",
        description="Search the web",
        parameters={"query": ToolParameter(description="Search query")},
    )

    features = ToolSegmenter(granularity="parameter").segment("Find docs", tools=[search])

    assert [feature.name for feature in features] == [
        "prompt",
        "tool:search:description",
        "tool:search:parameter:query",
    ]
