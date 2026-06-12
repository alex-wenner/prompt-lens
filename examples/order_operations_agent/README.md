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
| **Drill-down attribution** (`explain_drilldown`) | Sections first, then only the top 2 sections sentence-by-sentence — ~17 provider calls instead of ~29 for a flat sentence sweep, with the gap growing as prompts grow |
| **Argument-weighted tool drift** (`ToolArgumentDriftScorer`) | The order_id audit rule changes *only tool arguments*, never tool choice — invisible to plain sequence drift; the free-text `summary` param is weighted to `0.0` so rephrasing never counts |
| **Composite scoring** (`CompositeScorer`) | 0.7 trajectory drift + 0.3 length drift, so the JSON output contract registers too |
| **Markdown section segmentation** | The coarse pass that makes drill-down cheap |

## Run it

```bash
OPENAI_API_KEY=sk-... python examples/order_operations_agent/run.py
```

This makes **real provider calls** (~17 with drill-down). The default provider
is `openai`; pick another with `PROMPTLENS_EXAMPLE_PROVIDER` and override the
model with `PROMPTLENS_EXAMPLE_MODEL` (see [`_shared.py`](../_shared.py)). The
script prints the four tool definitions via `render_tools(TOOLS)` before the
sweep.

## Example output

(output from a gpt-5.4-mini run, abridged; your numbers will vary)

```text
Provider: openai · model: gpt-5.4-mini
╭──────────────────── Tools the model sees (4) ────────────────────╮
│ lookup_order  —  Fetch an order's status, line items, and        │
│ payment state.                                                   │
│  parameter  type    required  description                        │
│  order_id   string    yes     Order identifier, for example      │
│                               ORD-7421.                          │
│                                                                  │
│ create_rma  —  Open a return-merchandise authorization …         │
│ issue_refund  —  Refund the customer for an order …              │
│ escalate_to_human  —  Hand the ticket to a human reviewer.       │
╰──────────────────────────────────────────────────────────────────╯
Coarse-to-fine attribution over the Atlas instruction set:

                          promptlens Attribution
┏━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Feature   ┃  Value ┃ Share ┃ Weight     ┃ Text                          ┃
┡━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ section_3 │ 0.4210 │ 48.6% │ ██████████ │ # Refund policy  Refunds at … │
│ section_5 │ 0.2705 │ 31.2% │ ██████     │ # Tool usage rules  Always c… │
│ section_8 │ 0.1042 │ 12.0% │ ██         │ # Output contract  After you… │
│ section_6 │ 0.0710 │  8.2% │ ██         │ # Escalation matrix  Escalat… │
│ section_1 │ 0.0000 │  0.0% │            │ # Role  You are "Atlas", the… │
│ section_2 │ 0.0000 │  0.0% │            │ # Business objects  An Order… │
│ section_4 │ 0.0000 │  0.0% │            │ # Returns workflow  Create a… │
│ section_7 │ 0.0000 │  0.0% │            │ # Tone and communication  Be… │
└───────────┴────────┴───────┴────────────┴───────────────────────────────┘
                          Largest output drifts
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Removed features ┃  Score ┃ Tool calls without them  ┃ Output without t… ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ section_3        │ 0.4210 │ lookup_order(order_id="… │ {"status": "deli… │
│                  │        │ → issue_refund(order_id… │                   │
│ section_5        │ 0.2705 │ create_rma(order_id="OR… │ {"status": "deli… │
│ section_8        │ 0.1042 │ lookup_order(order_id="… │ I verified order… │
└──────────────────┴────────┴──────────────────────────┴───────────────────┘
                          Refined: section_3
┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Feature    ┃  Value ┃ Share ┃ Weight         ┃ Text                     ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ sentence_2 │ 0.4105 │ 68.4% │ ██████████████ │ Refunds above $100 must… │
│ sentence_3 │ 0.1602 │ 26.7% │ █████          │ Damaged-item claims req… │
│ sentence_1 │ 0.0295 │  4.9% │ █              │ Refunds at or below $10… │
│ sentence_4 │ 0.0000 │  0.0% │                │ Never issue a refund on… │
└────────────┴────────┴───────┴────────────────┴──────────────────────────┘
                          Refined: section_5
┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Feature    ┃  Value ┃ Share ┃ Weight     ┃ Text                        ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ sentence_1 │ 0.2950 │ 52.1% │ ██████████ │ Always call lookup_order f… │
│ sentence_2 │ 0.2483 │ 43.9% │ █████████  │ Include the order_id on ev… │
│ sentence_4 │ 0.0227 │  4.0% │ █          │ If a tool returns an error… │
│ sentence_3 │ 0.0000 │  0.0% │            │ Do not call the same tool … │
└────────────┴────────┴───────┴────────────┴─────────────────────────────┘
Drill-down used 17 provider calls vs ~29 for a flat sentence sweep.

What drill-down found:
  # Refund policy -> Refunds above $100 must be escalated to a human reviewer, regardless of account tier.
  # Tool usage rules -> Always call lookup_order first to verify the order status before taking any action.

Provider calls: 17 with drill-down vs ~29 for a flat sentence sweep.

Lens, not oracle: drill-down finds the sentences that drive the trajectory;
confirm policy edits with a task-level metric before shipping.
```

## What it finds

- `# Refund policy` carries about half the attribution mass; drilling in pins
  it to one sentence: *"Refunds above $100 must be escalated to a human
  reviewer."* — mask it and the agent refunds $182.40 directly instead of
  escalating.
- `# Tool usage rules` splits between *call lookup_order first* (drops a call
  when masked) and *include the order_id on every tool call* (pure argument
  drift — the tool-args scorer's reason to exist).
- The role blurb, business-object glossary, and tone rules score **0%** for
  this ticket — dead weight *for this task*, which is exactly the per-task
  nuance attribution is for.
