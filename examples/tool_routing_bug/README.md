# Tool-routing bug: find the description that breaks tool selection

**Problem:** an agent keeps calling the wrong tool, and you don't know which
part of its instructions is responsible.

A support agent has two tools (`lookup_order` and `search_catalog`). It should
answer *"where is my recent purchase?"* by calling `lookup_order`, but it only
routes correctly when it can still see the order-id hint in the
`order_reference` description. promptlens pinpoints that one sentence using the
objective `ToolAccuracyScorer`, and a before/after task metric proves the fix.

## Run it

```bash
OPENAI_API_KEY=sk-... python examples/tool_routing_bug/run.py
```

This makes **real provider calls** (about a dozen, with real tool schemas).
The default provider is `openai`; pick another with
`PROMPTLENS_EXAMPLE_PROVIDER=anthropic|gemini|grok|bedrock|copilot|ollama|openai-compatible`
and override the model with `PROMPTLENS_EXAMPLE_MODEL`. With no credential set,
the script exits with setup instructions (see [`_shared.py`](../_shared.py)).

The script prints the tool definitions first via `render_tools(TOOLS)`, so the
attribution table is read against exactly what the model could call.

## Example output

(output from a gpt-5.4-mini run; your numbers will vary)

```text
Provider: openai · model: gpt-5.4-mini
╭───────────────────── Tools the model sees (2) ─────────────────────╮
│ lookup_order  —  Look up the status of an existing customer order. │
│  parameter        type    required  description                    │
│  order_reference  string    yes     Identifier for the customer's  │
│                                     existing purchase.             │
│                                                                    │
│ search_catalog  —  Search the product catalog for items to buy.    │
│  parameter  type    required  description                          │
│  query      string    yes     What the customer wants to buy.      │
╰────────────────────────────────────────────────────────────────────╯
Attribution over the healthy prompt (objective: tool accuracy):

                          promptlens Attribution
┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Feature    ┃  Value ┃ Share ┃ Weight             ┃ Text                  ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ sentence_3 │ 0.9000 │ 90.0% │ ██████████████████ │ The order_reference … │
│ sentence_2 │ 0.1000 │ 10.0% │ ██                 │ Call lookup_order wh… │
│ sentence_1 │ 0.0000 │  0.0% │                    │ You are a support ag… │
│ sentence_4 │ 0.0000 │  0.0% │                    │ Call search_catalog … │
│ sentence_5 │ 0.0000 │  0.0% │                    │ User message: 'Where… │
└────────────┴────────┴───────┴────────────────────┴───────────────────────┘
                          Largest output drifts
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Removed features ┃  Score ┃ Tool calls without them┃ Output without them ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ sentence_1       │ 1.0000 │ lookup_order(order_re… │ Let me check that … │
│ sentence_4       │ 1.0000 │ lookup_order(order_re… │ Let me check that … │
│ sentence_2       │ 0.9000 │ lookup_order(order_re… │ Looking up your or… │
└──────────────────┴────────┴────────────────────────┴─────────────────────┘

Most load-bearing feature: sentence_3 (90% of attribution mass)
  -> The order_reference parameter is the customer's order ID, for example #1234.

Diagnosis confirmed by the task metric:
  tool accuracy with the misleading description: 0.00
  tool accuracy after restoring the description: 1.00

Lens, not oracle: attribution located the sentence that drives routing; the
before/after accuracy is what proves the fix.
```

Masking the order-id description is the only change that collapses tool
accuracy, so it dominates the attribution mass; the role blurb and the
`search_catalog` rule are dead weight *for this task*.

## Why this uses the objective scorer

`ToolAccuracyScorer` is an `objective` (task-quality) scorer, not a drift
scorer. The harness attributes a feature by how far the objective *drops* when
the feature is masked, so a sentence whose removal still yields the correct tool
call correctly receives near-zero attribution. See
[`docs/detailed-guide.md`](../../docs/detailed-guide.md) for the drift-vs-objective
distinction.

## Lens, not oracle

Attribution tells you *where* to look — it located the load-bearing sentence.
The before/after tool-accuracy number is what actually proves the edit worked.
Always confirm an attribution-driven change with a task-level metric.
