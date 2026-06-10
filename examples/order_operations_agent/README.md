# Order-operations agent: drill into a production-sized prompt

The flagship example: a realistic instruction set, not a toy. "Atlas" is an
order-operations agent for a B2B distributor; [`instructions.md`](instructions.md)
is its eight-section SOP — business objects (orders, RMAs, credit memos), a
refund policy with a dollar threshold, returns workflow, tool usage rules, an
escalation matrix, tone rules, and a JSON output contract. The ticket under
test: a **$182.40 refund on a damaged order**.

## What it demonstrates

| Capability | Where |
| ---------- | ----- |
| **Drill-down attribution** (`explain_drilldown`) | Sections first, then only the top 2 sections sentence-by-sentence — ~20 provider calls instead of ~29 for a flat sentence sweep, with the gap growing as prompts grow |
| **Argument-weighted tool drift** (`ToolArgumentDriftScorer`) | The order_id audit rule changes *only tool arguments*, never tool choice — invisible to plain sequence drift; the free-text `summary` param is weighted to `0.0` so rephrasing never counts |
| **Composite scoring** (`CompositeScorer`) | 0.7 trajectory drift + 0.3 length drift, so the JSON output contract registers too |
| **Markdown section segmentation** | The coarse pass that makes drill-down cheap |

## What it finds

- `# Refund policy` carries half the attribution mass; drilling in pins it to
  one sentence: *"Refunds above $100 must be escalated to a human reviewer."*
- `# Tool usage rules` splits evenly between *call lookup_order first* (drops a
  call when masked) and *include the order_id on every tool call* (pure
  argument drift — the tool-args scorer's reason to exist).
- The role blurb, business-object glossary, and tone rules score **0%** for
  this ticket — dead weight *for this task*, which is exactly the per-task
  nuance attribution is for.

## Run it

```bash
python examples/order_operations_agent/run.py
```

Runs against a real provider when `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` is set
(see [`_realprovider.py`](../_realprovider.py)); otherwise uses a deterministic
offline agent whose trajectory is governed by the same policy sentences a real
model would key on.
