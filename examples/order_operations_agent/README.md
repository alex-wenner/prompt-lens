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

This makes **real provider calls** — export `OPENAI_API_KEY` (or pick another
provider with `PROMPTLENS_EXAMPLE_PROVIDER`; see [`_shared.py`](../_shared.py)).
The run prints the full tool table the model sees (names, docstrings, parameter
names, types, and descriptions), the baseline trajectory with its tool calls,
and — after executing the called tools against stub backends — the model's
final reply to the tool results, so the whole agent loop is visible.

## Example output

```text
                          Tools exposed to the model (excerpt)
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Tool              ┃ Description                       ┃ Parameters                     ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ issue_refund      │ Refund the customer for an order, │ order_id: string (required)    │
│                   │ within policy limits.             │   Order to refund.             │
│                   │                                   │ amount: number (required)      │
│                   │                                   │   Refund amount in USD.        │
├───────────────────┼───────────────────────────────────┼────────────────────────────────┤
│ escalate_to_human │ Hand the ticket to a human        │ order_id, reason_code, summary │
│                   │ reviewer.                         │                                │
└───────────────────┴───────────────────────────────────┴────────────────────────────────┘
                                Baseline trajectory — tool calls
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Tool              ┃ Arguments                                                           ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ lookup_order      │ {"order_id": "ORD-7421"}                                            │
│ create_rma        │ {"order_id": "ORD-7421", "reason_code": "damaged_in_transit"}       │
│ escalate_to_human │ {"order_id": "ORD-7421", "reason_code": "refund_over_limit",        │
│                   │  "summary": "Refund of $182.40 exceeds the $100 direct-refund       │
│                   │  limit."}                                                           │
└───────────────────┴─────────────────────────────────────────────────────────────────────┘
╭──────────────────── Model's reply after the tool results ─────────────────────╮
│ {"status": "pending_review", "action_taken": "escalated_to_reviewer",         │
│  "next_step": "Reply to Dana with the outcome."}                              │
╰────────────────────────── usage: 470 in / 28 out ─────────────────────────────╯

                     promptlens Attribution (coarse pass, excerpt)
┏━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Feature   ┃  Value ┃ Share ┃ Weight     ┃ Text                                         ┃
┡━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ section_3 │ 0.5026 │ 50.6% │ ██████████ │ # Refund policy …                            │
│ section_5 │ 0.3500 │ 35.2% │ ███████    │ # Tool usage rules …                         │
│ section_8 │ 0.1410 │ 14.2% │ ███        │ # Output contract …                          │
│ section_1 │ 0.0000 │  0.0% │            │ # Role …                                     │
│ section_2 │ 0.0000 │  0.0% │            │ # Business objects …                         │
└───────────┴────────┴───────┴────────────┴──────────────────────────────────────────────┘
                              Refined: section_3 (excerpt)
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Feature              ┃  Value ┃ Share ┃ Weight      ┃ Text                             ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ section_3.sentence_2 │ 0.2692 │ 53.6% │ ███████████ │ Refunds above $100 must be       │
│                      │        │       │             │ escalated to a human reviewer …  │
│ section_3.sentence_3 │ 0.2333 │ 46.4% │ █████████   │ Damaged-item claims require an   │
│                      │        │       │             │ RMA before any refund is issued. │
└──────────────────────┴────────┴───────┴─────────────┴──────────────────────────────────┘

Provider calls: 20 with drill-down vs ~30 for a flat sentence sweep.
```
