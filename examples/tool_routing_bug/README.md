# Tool-routing bug: find the description that breaks tool selection

**Problem:** an agent keeps calling the wrong tool, and you don't know which
part of its instructions is responsible.

This example simulates a support agent with two tools (`lookup_order` and
`search_catalog`). The agent should answer *"where is my recent purchase?"* by
calling `lookup_order`, but it only routes correctly when it can still see the
order-id hint in the `order_reference` description. promptlens pinpoints that one
sentence using the objective `ToolAccuracyScorer`, and a before/after task
metric proves the fix.

## Run it

```bash
python examples/tool_routing_bug/run.py
```

By default this calls a **real provider** with real tool schemas. Set
`OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` plus
`PROMPTLENS_EXAMPLE_PROVIDER=anthropic`) to choose the model;
`PROMPTLENS_EXAMPLE_MODEL` overrides the default model. With no credential set it
falls back to a small deterministic simulated adapter, so it still runs offline
and as a CI smoke test.

## What you should see

With the offline fallback, attribution over the healthy prompt ranks the
order-id sentence at the top because masking it is the only change that collapses
tool accuracy (a real model will vary, but should still surface that sentence):

| Feature      | Share  | Meaning                                         |
| ------------ | ------ | ----------------------------------------------- |
| `sentence_3` | 100.0% | The `order_reference` order-id description       |
| `sentence_1` | 0.0%   | "You are a support agent…" (dead weight)        |
| `sentence_2` | 0.0%   | "Call lookup_order when…" (dead weight here)    |
| `sentence_4` | 0.0%   | "Call search_catalog only when…" (dead weight)  |

Then the task metric confirms the diagnosis:

```
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
