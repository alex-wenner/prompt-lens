from promptlens.segmenters import SentenceSegmenter, ToolSegmenter


def test_sentence_segmenter_tracks_sentences_and_tools() -> None:
    tools = [{"name": "search", "description": "Search docs"}]
    features = SentenceSegmenter().segment("First sentence. Second sentence!", tools=tools)

    assert [feature.name for feature in features] == ["sentence_1", "sentence_2", "tools"]
    assert features[0].text == "First sentence."
    assert features[2].metadata["kind"] == "tools"


def test_tool_segmenter_handles_openai_parameter_granularity() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                },
            },
        }
    ]

    features = ToolSegmenter(granularity="parameter").segment("Find docs", tools=tools)

    assert [feature.name for feature in features] == [
        "prompt",
        "tool:search:description",
        "tool:search:parameter:query",
    ]
