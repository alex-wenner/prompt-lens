# Tool-routing bug: find the description that breaks tool selection

**Problem:** an agent keeps calling the wrong tool, and you don't know which
part of its instructions is responsible.

This example runs a support agent with two tools (`lookup_order` and
`search_catalog`). The agent should answer *"where is my recent purchase?"* by
calling `lookup_order`, but it only routes correctly when it can still see the
order-id hint in the `order_reference` description. promptlens pinpoints that one
sentence using the objective `ToolAccuracyScorer`, and a before/after task
metric proves the fix.

## Run it

```bash
python examples/tool_routing_bug/run.py
```

This calls a **real provider** with real tool schemas. Export `OPENAI_API_KEY`
(or `ANTHROPIC_API_KEY` plus `PROMPTLENS_EXAMPLE_PROVIDER=anthropic`) to choose
the model; `PROMPTLENS_EXAMPLE_MODEL` overrides the default model. The run
prints the full tool table the model sees (names, docstrings, parameter names,
types, and descriptions), the model's tool calls, and — after executing the
chosen tool against a stub backend — the model's final answer to the tool
result, so the whole agent loop is visible.

## Example output

Attribution over the healthy prompt ranks the order-id sentence at the top
because masking it is the only change that collapses tool accuracy (exact
numbers vary by model, but it should surface that sentence):

```text
                          Tools exposed to the model
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Tool           ┃ Description                          ┃ Parameters                         ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ lookup_order   │ Look up the status of an existing    │ order_reference: string (required) │
│                │ customer order.                      │   Identifier for the customer's    │
│                │                                      │   existing purchase.               │
├────────────────┼──────────────────────────────────────┼────────────────────────────────────┤
│ search_catalog │ Search the product catalog for items │ query: string (required)           │
│                │ to buy.                              │   What the customer wants to buy.  │
└────────────────┴──────────────────────────────────────┴────────────────────────────────────┘
    Baseline routing decision — tool calls
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Tool         ┃ Arguments                    ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ lookup_order │ {"order_reference": "#1234"} │
└──────────────┴──────────────────────────────┘
╭──────────────── Model's answer after the tool result ─────────────────╮
│ Your order #1234 shipped two days ago via UPS and should arrive       │
│ tomorrow. Tracking number: 1Z999AA10123456784.                        │
╰────────────────────────── usage: 77 in / 26 out ──────────────────────╯

                          promptlens Attribution
┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Feature    ┃  Value ┃  Share ┃ Weight               ┃ Text                                  ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ sentence_3 │ 1.0000 │ 100.0% │ ████████████████████ │ The order_reference parameter is the  │
│            │        │        │                      │ customer's order ID, for example      │
│            │        │        │                      │ #1234.                                │
│ sentence_1 │ 0.0000 │   0.0% │                      │ You are a support agent with two      │
│            │        │        │                      │ tools.                                │
│ sentence_2 │ 0.0000 │   0.0% │                      │ Call lookup_order when the user asks  │
│            │        │        │                      │ about an existing purchase.           │
│ sentence_4 │ 0.0000 │   0.0% │                      │ Call search_catalog only when the     │
│            │        │        │                      │ user wants to buy a new product.      │
└────────────┴────────┴────────┴──────────────────────┴───────────────────────────────────────┘

Most load-bearing feature: sentence_3 (100% of attribution mass)
  -> The order_reference parameter is the customer's order ID, for example #1234.

Diagnosis confirmed by the task metric:
  tool accuracy with the misleading description: 0.00
  tool accuracy after restoring the description: 1.00
```

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
